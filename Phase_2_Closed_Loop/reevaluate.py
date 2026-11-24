"""Re-evaluate saved agents with enough episodes to be worth quoting.

`final_eval_return` in each result.json comes from 5 episodes. The evaluation
standard deviations in these runs are 60-100, so each of those numbers carries
an error of roughly +/-30 to +/-45 — comparable to several of the differences
the comparison rests on. Reporting a 104-point mean difference built from
numbers that individually wobble by 40 is not defensible.

This reruns the evaluation from the saved checkpoints with many more episodes.
It is pure inference: no training, no environment interaction beyond the
evaluation itself, a few minutes for all ten agents. It uses train_agent.py's
own `evaluate()` so the measurement is identical to the one taken during
training, differing only in the episode count.

    python reevaluate.py                     50 episodes, every run
    python reevaluate.py --episodes 100
    python reevaluate.py --runs runs/cartpole_quantum_q6L4_full_b8x16h10_seed0

Writes reevaluation.json and prints the paired comparison.
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Everything that defines a comparable condition. `quantum` and `seed` are
# deliberately absent: those are the two axes the comparison varies over.
# If a setting is added to config.py, add it here as well, or two different
# conditions will be averaged together into one number.
CONDITION_FIELDS = (
    "task", "partial_obs", "prior_arch", "prior_context", "d_model",
    "n_heads", "n_layers", "quantum_layer_position", "prior_hidden",
    "n_qubits", "q_layers", "entangler", "readout", "encoding",
    "batch_size", "seq_len", "horizon", "train_every",
)


def cfg_steps(cfg):
    from config import resolve_steps
    return resolve_steps(cfg)


#: Parameter count below which a prior is the non-attentional MLP. The two
#: families are ~18.7k and ~85k, so anything in between is unambiguous.
ARCH_SPLIT = 40_000


def arch_of(prior_params, cfg):
    """Architecture inferred from SIZE, not from a possibly-absent key.

    Runs made before `prior_arch` existed have no such key in their
    config.json, so reconstructing a Config from that file leaves the
    field at its CURRENT default -- which is now "transformer". Those runs
    self-report as transformers with K=8 when they are in fact the
    non-attentional MLP prior, and grouping on cfg alone would silently
    merge a genuine K=8 transformer run with them.

    The parameter count cannot lie about this: ~18.7k is the MLP prior,
    ~85k the transformer.
    """
    if prior_params is None:
        return str(getattr(cfg, "prior_arch", "?")) + "?"
    return "mlp" if prior_params < ARCH_SPLIT else "transformer"


def condition_key(cfg, prior_params=None):
    """Identify a comparable condition.

    The architecture enters as a BUCKET, never as the raw parameter count.
    An earlier version keyed on `prior_params` directly, which looked
    equivalent and was not: the quantum prior has 84,770 parameters and
    its capacity-matched classical twin 85,494. Keying on the exact count
    therefore separated the two ARMS of every comparison into different
    groups, so no group contained a pair and every condition reported
    "0 complete pairs". The arms are supposed to differ slightly in size
    -- that is what capacity-matched means -- so the key must be coarse
    enough to ignore it and fine enough to separate 18.7k from 85k.
    """
    base = tuple((f, getattr(cfg, f, None)) for f in CONDITION_FIELDS)
    return base + (("arch_class", arch_of(prior_params, cfg)),)


def condition_label(key):
    d = dict(key)
    arch = d.get("arch_class", "?")
    ctx = f"K{d.get('prior_context')}" if arch.startswith("transformer") else ""
    return (f"{d.get('task')}/"
            f"{'partial' if d.get('partial_obs') else 'full'}/"
            f"{arch}{ctx}"
            f"/ph{d.get('prior_hidden')}/ent{d.get('entangler')}"
            f"/ro{d.get('readout')}")


def paired_report(rows, label, include_incomplete):
    """rows: list of the per-run dicts for ONE condition."""
    pairs = {}
    for r in rows:
        if not include_incomplete and not r["complete"]:
            continue
        pairs.setdefault(r["seed"], {})[r["arm"]] = r["mean"]
    seeds = sorted(s for s in pairs if len(pairs[s]) == 2)
    print(f"\n  {label}")
    if len(seeds) < 3:
        have = sorted(pairs)
        print(f"    only {len(seeds)} complete pair(s) "
              f"(seeds present: {have}) — nothing to report yet")
        return
    q = np.array([pairs[s]["quantum"] for s in seeds])
    c = np.array([pairs[s]["classical"] for s in seeds])
    d = q - c
    n = len(d)
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, n, (20000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    null = np.abs((signs * np.abs(d)).mean(axis=1))
    p = float((null >= abs(d.mean()) - 1e-15).mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    print(f"    seeds {seeds}")
    print(f"    quantum {q.mean():.1f}   classical {c.mean():.1f}")
    print(f"    difference {d.mean():+.1f}   95% CI [{lo:+.1f}, {hi:+.1f}]   "
          f"exact permutation p = {p:.3f}")
    print(f"    quantum ahead in {int((q > c).sum())}/{n} seeds   "
          f"sd of paired difference {sd:.1f}")
    print(f"    smallest attainable two-sided p at n={n} is {2 / 2**n:.3f}")
    if sd == sd and abs(d.mean()) > 0:
        need = (2.80 * sd / abs(d.mean())) ** 2
        print(f"    seeds for 80% power at this effect and dispersion: "
              f"~{int(np.ceil(need))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000,
                    help="evaluation seed offset; distinct from training seeds "
                         "so the agents are not scored on episodes they saw")
    ap.add_argument("--out", default="reevaluation.json")
    ap.add_argument("--include_incomplete", action="store_true",
                    help="include runs that have not reached their step "
                         "budget. Diagnostic only — never for a quoted number.")
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("torch is missing — activate the project environment:\n"
                 "    source ~/.venvs/qtwm/bin/activate")
    import quantum_rssm
    import train_agent as ta
    from imagination_quality import load_agent

    runs = args.runs or sorted(glob.glob(os.path.join(HERE, "runs", "*")))
    # Normalise before anything uses the path as an identity.
    #
    # A shell glob of the form runs/*foo*/ expands WITH a trailing slash,
    # and os.path.basename("runs/x/") is the empty string. Every result
    # then lands under the same key, the dict keeps only the last one, and
    # the paired comparison reports one condition containing one seed --
    # after having correctly scored all 29 agents. The output looks like a
    # grouping failure and is really a path-handling failure.
    runs = [os.path.normpath(r) for r in runs]
    runs = [r for r in runs if os.path.exists(os.path.join(r, "checkpoint.pt"))]
    if not runs:
        sys.exit("no checkpoints found under runs/")
    names = [os.path.basename(r) for r in runs]
    if len(set(names)) != len(names) or any(not n for n in names):
        sys.exit("run paths do not give unique non-empty names; "
                 "pass directories rather than globs with trailing slashes")

    print(f"re-evaluating {len(runs)} agents over {args.episodes} episodes "
          f"(was 5)\n")
    out = {}
    for run in runs:
        name = os.path.basename(os.path.normpath(run))
        assert name and name not in out, f"duplicate or empty run name: {name!r}"
        try:
            agent, cfg, env, env_id, discrete, step = load_agent(
                run, ta, quantum_rssm)
            mean, std = ta.evaluate(agent, env_id, cfg, discrete,
                                    args.episodes, seed=args.seed)
        except Exception as e:
            print(f"  {name:<50} skipped ({type(e).__name__}: {e})")
            continue
        # standard error of the mean, which is what the comparison needs
        sem = std / np.sqrt(args.episodes)
        old, prior_params = None, None
        rp = os.path.join(run, "result.json")
        for src in (rp, os.path.join(run, "model_info.json")):
            if not os.path.exists(src):
                continue
            try:
                blob = json.load(open(src))
            except Exception:
                continue
            if old is None:
                old = blob.get("final_eval_return", old)
            if prior_params is None:
                prior_params = blob.get("prior_params")
        # ---- completeness and condition, both read from cfg, never the name.
        # An earlier version derived the seed and the arm by substring-matching
        # the directory name and keyed the paired comparison on (seed, arm)
        # alone. Every condition therefore collided: full-observation runs,
        # partial-observation runs and the head-width and entangler ablations
        # all mapped to the same key, and whichever directory sorted last won.
        # It happened to select the partial-observation runs, because "po"
        # sorts after "full" and after every ablation suffix, so the printed
        # comparison was the intended one by coincidence rather than by
        # construction. It also silently included runs that were still
        # training, scoring half-trained checkpoints against finished
        # baselines. Both are fixed by keying on the config and refusing
        # incomplete runs.
        target = cfg_steps(cfg)
        done = os.path.exists(rp) and step is not None and step >= target
        out[name] = {"mean": mean, "std": std, "sem": float(sem),
                     "episodes": args.episodes, "step": step,
                     "target_steps": target, "complete": bool(done),
                     "condition": condition_key(cfg, prior_params),
                     "prior_params": prior_params,
                     "arch": arch_of(prior_params, cfg),
                     "arm": "quantum" if getattr(cfg, "quantum", False)
                            else "classical",
                     "seed": int(getattr(cfg, "seed", -1)),
                     "previous_5_episode": old}
        delta = f"{mean - old:+.1f}" if old is not None else "—"
        flag = "" if done else f"  INCOMPLETE {step}/{target}"
        print(f"  {name:<50} {mean:>7.1f} +/- {sem:>5.1f} (sem)   "
              f"was {old if old is not None else float('nan'):>7.1f}  "
              f"change {delta}{flag}")

    with open(os.path.join(HERE, args.out), "w") as f:
        json.dump(out, f, indent=1)

    # ------------------------------------------- paired comparison, per condition
    groups = {}
    for name, d in out.items():
        if d["seed"] < 0:
            continue
        groups.setdefault(d["condition"], []).append(d)

    n_incomplete = sum(1 for d in out.values() if not d["complete"])
    print(f"\n{'=' * 72}")
    print(f"PAIRED COMPARISONS — {len(groups)} condition(s), "
          f"{n_incomplete} incomplete run(s) excluded")
    print("Only runs that reached their full step budget are included. A "
          "half-trained\ncheckpoint scored against a finished baseline is not "
          "a measurement.")
    print("=" * 72)
    for key in sorted(groups, key=condition_label):
        paired_report(groups[key], condition_label(key),
                      include_incomplete=args.include_incomplete)

    if n_incomplete and not args.include_incomplete:
        print("\n  excluded as incomplete:")
        for name, d in sorted(out.items()):
            if not d["complete"]:
                print(f"    {name}  ({d['step']}/{d['target_steps']} steps)")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
