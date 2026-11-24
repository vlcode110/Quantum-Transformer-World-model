"""Run the budget-matched comparison: both arms, equal budgets, several seeds.

The existing runs cannot support a performance claim. They span 2,500 to 80,000
environment steps because several were interrupted, so the arms are not
comparable, and the run tag does not encode the step count — meaning a rerun
silently overwrites a run made at a different budget. This script fixes both:
every run uses the same budget, existing runs are backed up before anything is
overwritten, and completed runs are skipped so an interrupted sweep resumes
rather than restarting.

    python run_matched.py --dry-run          show the plan, run nothing
    caffeinate -i python run_matched.py      the real thing
    caffeinate -i python run_matched.py --jobs 6 --seeds 0 1 2 3 4 5 6 7

Parallelism: PennyLane's backprop is CPU-bound, so each process is pinned to a
single thread and several are run concurrently. That is faster in wall clock
than letting one process grab every core, because the circuit evaluations do not
parallelise well across threads. Pick `--jobs` at or just below your physical
core count; the default of 4 is deliberately conservative so the machine stays
usable.

Safe to run alongside a remote-device job: those are network-bound and use
essentially no CPU.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
LOGS = os.path.join(HERE, "logs")


def tag_for(quantum, seed, cfg_bits="b8x16h10"):
    """Mirrors Config.tag(). Kept in one place so the completion check and the
    backup logic cannot drift from what training actually writes."""
    arm = "quantum_q6L4" if quantum else "classical_ctrl"
    return f"cartpole_{arm}_full_{cfg_bits}_seed{seed}"


def completed(tag, steps):
    """A run counts as done only if it finished at the budget we asked for."""
    path = os.path.join(RUNS, tag, "result.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return False
    return int(d.get("env_steps", 0)) >= steps


def backup_existing(steps, dry):
    """Move aside any run that exists at a different budget.

    The tag omits the step count, so without this a 40,000-step run and an
    80,000-step run of the same seed occupy the same directory and the second
    destroys the first. The old runs are not useful for the comparison, but
    overwriting data silently is how results become unreproducible.
    """
    if not os.path.isdir(RUNS):
        return None
    stale = []
    for name in sorted(os.listdir(RUNS)):
        path = os.path.join(RUNS, name, "result.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                got = int(json.load(f).get("env_steps", 0))
        except Exception:
            continue
        if got < steps:
            stale.append((name, got))
    if not stale:
        return None
    dest = os.path.join(HERE, f"runs_backup_{datetime.now():%Y%m%d_%H%M}")
    print(f"  {len(stale)} run(s) exist at a smaller budget and would be "
          f"overwritten:")
    for name, got in stale:
        print(f"      {name}  ({got:,} steps)")
    print(f"  backing them up to {os.path.basename(dest)}/")
    if not dry:
        os.makedirs(dest, exist_ok=True)
        for name, _ in stale:
            shutil.move(os.path.join(RUNS, name), os.path.join(dest, name))
    return dest


def launch(quantum, seed, steps, env):
    tag = tag_for(quantum, seed)
    os.makedirs(LOGS, exist_ok=True)
    log = open(os.path.join(LOGS, tag + ".log"), "w")
    cmd = [sys.executable, "train_agent.py",
           "--task", "cartpole",
           "--quantum", "true" if quantum else "false",
           "--seed", str(seed),
           "--steps", str(steps)]
    p = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                         env=env)
    p._tag, p._log = tag, log
    return p


def preflight():
    """Confirm this interpreter can actually run a training job.

    `sys.executable` is whatever Python launched this script, which is not
    necessarily the project environment — running it from a conda base env
    launched eight processes that each died on an import error sixty seconds
    later. One subprocess here costs a second and turns that into a clear
    message before anything starts.
    """
    probe = ("import torch, pennylane, numpy; "
             "print(torch.__version__, pennylane.__version__, numpy.__version__)")
    r = subprocess.run([sys.executable, "-c", probe],
                       capture_output=True, text=True, cwd=HERE)
    print(f"  interpreter  {sys.executable}")
    if r.returncode == 0:
        tv, pv, nv = r.stdout.split()
        print(f"  torch {tv}, pennylane {pv}, numpy {nv}\n")
        return True
    last = [l for l in r.stderr.strip().splitlines() if l.strip()][-1]
    print(f"\n  This interpreter cannot run a training job:\n    {last}\n")
    print("  The project environment is not active. Activate it first:\n")
    print("      source ~/.venvs/qtwm/bin/activate")
    print("      cd ~/Documents/PhD/qtwm/Phase_2_Closed_Loop")
    print("      caffeinate -i python run_matched.py --jobs 4\n")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # One thread per process. Several single-threaded runs finish sooner than
    # the same runs each contending for every core.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

    plan = [(q, s) for s in args.seeds for q in (False, True)]
    todo = [(q, s) for q, s in plan if not completed(tag_for(q, s), args.steps)]
    done = len(plan) - len(todo)

    print(f"budget {args.steps:,} env steps, seeds {args.seeds}, "
          f"{args.jobs} concurrent\n")
    if not preflight() and not args.dry_run:
        sys.exit(1)
    backup_existing(args.steps, args.dry_run)
    print(f"\n  {len(plan)} runs planned, {done} already complete, "
          f"{len(todo)} to run")
    # classical is roughly 0.07 s/step, quantum roughly 0.3 s/step uncontended
    est = sum((0.30 if q else 0.07) * args.steps for q, _ in todo) / 3600
    print(f"  ~{est:.0f} CPU-hours total, so roughly "
          f"{est / args.jobs:.0f}-{est / args.jobs * 1.4:.0f} h wall clock "
          f"at {args.jobs} jobs")
    print(f"  (estimates from the least-contended existing runs; treat as "
          f"indicative)")

    if args.dry_run:
        print("\n  plan:")
        for q, s in todo:
            print(f"      {tag_for(q, s)}")
        return

    if not todo:
        print("\n  Nothing to do.")
        return

    print(f"\n  logs -> {os.path.relpath(LOGS, HERE)}/\n")
    running, queue, t0 = [], list(todo), time.time()
    while queue or running:
        while queue and len(running) < args.jobs:
            q, s = queue.pop(0)
            p = launch(q, s, args.steps, env)
            running.append(p)
            print(f"  [{time.time()-t0:7.0f}s] start  {p._tag}")
        time.sleep(5)
        for p in running[:]:
            if p.poll() is not None:
                running.remove(p)
                p._log.close()
                ok = "done " if p.returncode == 0 else f"FAILED({p.returncode})"
                print(f"  [{time.time()-t0:7.0f}s] {ok} {p._tag}"
                      f"   {len(queue)} queued, {len(running)} running")

    print(f"\n  All runs finished in {(time.time()-t0)/3600:.1f} h.")
    print("  Next:  python analyze.py")


if __name__ == "__main__":
    main()
