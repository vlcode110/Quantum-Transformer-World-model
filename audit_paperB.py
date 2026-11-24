"""Audit every Phase 2 number Paper B depends on, against raw result.json.

    python audit_paperB.py

Reads only. Prints OK or MISMATCH per claim and exits non-zero if any
claim fails, so it can be run before each submission.

The arithmetic in the paper has already been checked independently (the
tables recompute exactly). What this script checks is the layer beneath
that, which arithmetic cannot: that the runs behind each cell really are
the runs the paper says they are. That is where every error in this
project has actually lived -- five collisions, three of which changed a
reported number, none of which an arithmetic check could have caught.
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "Phase_2_Closed_Loop", "runs")

# The paper's Table 3, transcribed. If a cell here disagrees with the
# .tex, fix one of them -- do not assume which.
PAPER_PARTIAL_Q = {0: 420.9, 1: 375.2, 2: 478.7, 3: 313.5,
                   4: 368.9, 5: 298.9, 6: 131.5, 7: 443.9}
PAPER_PARTIAL_C = {0: 111.0, 1: 312.6, 2: 469.7, 3: 195.0,
                   4: 150.8, 5: 138.0, 6: 205.0, 7: 96.9}
PAPER_FULL_Q = {0: 495.6, 1: 459.1, 2: 196.0, 3: 396.1, 4: 205.3, 5: 233.6}
PAPER_FULL_C = {0: 228.5, 1: 391.9, 2: 139.3, 3: 469.8, 4: 184.8, 5: 426.2}

QUANTUM_PRIOR_PARAMS = 84770
CLASSICAL_PRIOR_PARAMS = 85494
STEP_BUDGET = 80000

failures = []


def check(ok, msg):
    print(("  OK   " if ok else "  FAIL ") + msg)
    if not ok:
        failures.append(msg)


def rd(path, attempts=15):
    """iCloud intermittently refuses opens; retry rather than report a
    missing file as a missing run."""
    last = None
    for i in range(attempts):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            last = e
            time.sleep(0.3 + 0.2 * i)
    print(f"  [io] could not read {path}: {type(last).__name__}")
    return None


def main():
    dirs = sorted(glob.glob(os.path.join(RUNS, "*archtransformer_K4*", "")))
    print(f"\nfound {len(dirs)} transformer-K4 run directories\n")

    runs = []
    for d in dirs:
        r = rd(os.path.join(d, "result.json"))
        if r is None:
            check(False, f"unreadable or missing result.json: "
                         f"{os.path.basename(d.rstrip(os.sep))}")
            continue
        r["_dir"] = os.path.basename(d.rstrip(os.sep))
        runs.append(r)

    # ---------------------------------------------------- cell membership
    print("1. Cell membership and completeness")
    cells = {}
    for r in runs:
        if r["env_steps"] != STEP_BUDGET:
            continue                      # smoke tests are not in the paper
        key = ("partial" if r["partial_obs"] else "full",
               "Q" if r["quantum"] else "C")
        cells.setdefault(key, {})
        seed = r["seed"]
        check(seed not in cells[key],
              f"{key} seed {seed} appears once (duplicate would mean a "
              f"collision)")
        cells[key][seed] = r

    for key, expect in ((("partial", "Q"), 8), (("partial", "C"), 8),
                        (("full", "Q"), 6), (("full", "C"), 6)):
        got = cells.get(key, {})
        check(len(got) == expect,
              f"{key[0]}/{key[1]} has {len(got)} complete runs, paper uses "
              f"{expect}")
        check(sorted(got) == list(range(expect)),
              f"{key[0]}/{key[1]} seeds are {sorted(got)}, expected "
              f"{list(range(expect))}")

    # ------------------------------------------------------- architecture
    print("\n2. Architecture: both arms are the transformer prior, not the MLP")
    for key, want in ((("partial", "Q"), QUANTUM_PRIOR_PARAMS),
                      (("partial", "C"), CLASSICAL_PRIOR_PARAMS),
                      (("full", "Q"), QUANTUM_PRIOR_PARAMS),
                      (("full", "C"), CLASSICAL_PRIOR_PARAMS)):
        got = {r.get("prior_params") for r in cells.get(key, {}).values()}
        check(got == {want},
              f"{key[0]}/{key[1]} prior_params = {got or 'none'}, expected "
              f"{{{want}}}")

    # ------------------------------------------- settings held constant
    print("\n3. Every setting except arm and seed is identical within a cell")
    for key, members in sorted(cells.items()):
        varying = {}
        for r in members.values():
            for k, v in r.items():
                if k in ("seed", "quantum", "final_eval_return",
                         "final_eval_std", "solved_at_step", "wall_clock_s",
                         "run_dir", "_dir", "gate_met", "prior",
                         "prior_params", "prior_params_replaced",
                         "quantum_params", "total_params"):
                    continue
                varying.setdefault(k, set()).add(
                    json.dumps(v, sort_keys=True, default=str))
        bad = {k: v for k, v in varying.items() if len(v) > 1}
        check(not bad,
              f"{key[0]}/{key[1]}: settings constant across seeds"
              + (f" -- VARYING: {sorted(bad)}" if bad else ""))

    # ---------------------------------------- partial_obs really differs
    print("\n4. The two observability conditions really differ")
    po = {r["partial_obs"] for r in cells.get(("partial", "Q"), {}).values()}
    fu = {r["partial_obs"] for r in cells.get(("full", "Q"), {}).values()}
    check(po == {True} and fu == {False},
          f"partial_obs is {po} in the partial cell and {fu} in the full cell")

    # -------------------------------------- paper values trace to a run
    print("\n5. Every value in Table 3 has a corresponding completed run")
    for cond, arm, table in (("partial", "Q", PAPER_PARTIAL_Q),
                             ("partial", "C", PAPER_PARTIAL_C),
                             ("full", "Q", PAPER_FULL_Q),
                             ("full", "C", PAPER_FULL_C)):
        members = cells.get((cond, arm), {})
        missing = [s for s in table if s not in members]
        check(not missing,
              f"{cond}/{arm}: all {len(table)} paper seeds have runs"
              + (f" -- MISSING {missing}" if missing else ""))

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED -- do not submit until resolved:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED.")
    print("Note: Table 3 reports 50-episode re-evaluations, which are NOT in")
    print("result.json (that holds the 20-episode training-time figure). This")
    print("script verifies the runs behind each cell; reevaluation_tf.json")
    print("holds the scores themselves.")
    sys.exit(0)


if __name__ == "__main__":
    main()
