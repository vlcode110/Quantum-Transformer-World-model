"""Audit every number in Paper A against the raw run records.

    python audit_paperA.py

Reads only. Exits non-zero if any check fails.

WHY THIS EXISTS
---------------
Paper A argues that aggregation errors change published numbers. If
Paper A contains one, it refutes itself. Its three data loaders in
make_figures.py each key on a partial configuration, which is the exact
pattern that manufactured a 29% effect in the companion study:

  load_phase3()          key = (observables, n_spins), filtered to
                         d_model == 16. Not in the key: n_qubits,
                         q_layers, entangler, encoding, readout,
                         context, n_layers, dataset_version, task.

  load_qasa()            key = (signal, seed, mode). comparison.csv
                         carries no configuration columns at all, so a
                         collision is undetectable from the file --- and
                         a SECOND file, comparison_brickwork.csv, exists
                         alongside it.

  load_phase1_cartpole() filters rows where d_model is None, then keys
                         on (seed, model). ablation.jsonl has no
                         d_model field whatsoever, so that filter
                         selects EVERY row including every ablation
                         arm.

Each check below targets one of those. A clean run means the numbers in
the paper come from single, well-defined conditions. It does not mean
the loaders are safe --- it means they happened not to collide.
"""
import csv
import glob
import itertools
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------- paths
# Prefer the copy inside this repository, fall back to a sibling of
# ~/Documents. The companion code was originally kept outside this tree
# and is now vendored in, so a fresh clone is self-contained; the
# fallback keeps older local checkouts working.
def _locate(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", name),            # repo root
                 os.path.join(here, "..", "..", "..", name),
                 os.path.join(here, "..", "..", "..", "..", name)):
        cand = os.path.abspath(cand)
        if os.path.isdir(cand):
            return cand
    return os.path.abspath(os.path.join(here, "..", name))


HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
QASA = _locate("QASA-repro")
PH1 = os.path.join(DOCS, "PhD", "qtwm", "Phase_1_Scaffold_test")
PH3 = os.path.join(DOCS, "PhD", "qtwm", "Phase_3_Quantum_Native")

failures, warnings_ = [], []


def ok(cond, msg):
    print(("  OK    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def warn(msg):
    print("  WARN  " + msg)
    warnings_.append(msg)


def rd(path, attempts=12):
    last = None
    for i in range(attempts):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            last = e
            time.sleep(0.25 + 0.15 * i)
    warn(f"could not read {path} ({type(last).__name__})")
    return None


def rows_csv(path, attempts=12):
    last = None
    for i in range(attempts):
        try:
            with open(path, newline="") as f:
                return list(csv.DictReader(f))
        except FileNotFoundError:
            return None
        except Exception as e:
            last = e
            time.sleep(0.25 + 0.15 * i)
    warn(f"could not read {path} ({type(last).__name__})")
    return None


# ------------------------------------------------------------ statistics
def stats(q, c):
    q, c = np.asarray(q, float), np.asarray(c, float)
    lr = np.log(q / c)
    n = len(lr)
    gm = float(np.exp(lr.mean()))
    rng = np.random.default_rng(0)
    boot = np.exp(lr[rng.integers(0, n, (20000, n))].mean(axis=1))
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    if n <= 22:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        null = np.abs((signs * np.abs(lr)).mean(axis=1))
        p = float((null >= abs(lr.mean()) - 1e-15).mean())
    else:
        rng2 = np.random.default_rng(1)
        s = rng2.choice([-1.0, 1.0], size=(200000, n))
        p = float((np.abs((s * np.abs(lr)).mean(1)) >= abs(lr.mean())
                   - 1e-15).mean())
    return dict(n=n, ratio=gm, ci=[lo, hi], quantum_ahead=int((q < c).sum()),
                perm_p=p, sd=float(lr.std(ddof=1)) if n > 1 else float("nan"))


# =====================================================================
print("\n" + "=" * 70)
print("1. PHASE 3 (Ising) -- full-configuration collision check")
print("=" * 70)
KEY3 = ("task", "observables", "n_spins", "n_qubits", "q_layers", "entangler",
        "encoding", "readout", "context", "d_model", "n_layers",
        "latent_mode", "latent_dim", "dataset_version")

p3 = []
for p in sorted(glob.glob(os.path.join(PH3, "runs", "*", "result.json"))):
    r = rd(p)
    if r:
        r["_dir"] = os.path.basename(os.path.dirname(p))
        p3.append(r)
print(f"  {len(p3)} Phase 3 runs read")

missing = sorted({f for f in KEY3 for r in p3 if f not in r})
if missing:
    warn(f"fields absent from Phase 3 result.json, so unverifiable: {missing}")

slots = defaultdict(list)
for r in p3:
    if r.get("d_model") != 16:
        continue
    key = tuple(r.get(f, "<absent>") for f in KEY3) + (r["model"], r["seed"])
    slots[key].append(r)
dupes = {k: v for k, v in slots.items() if len(v) > 1}
ok(not dupes, f"no two d_model=16 runs share a full configuration key "
              f"({len(slots)} distinct slots)")
for k, v in list(dupes.items())[:5]:
    vals = [x["test_rollout"]["1"] for x in v]
    print(f"        collision: {[x['_dir'] for x in v]}")
    print(f"        values:    {vals}")

# the paper's loader key, for comparison
loose = defaultdict(set)
for r in p3:
    if r.get("d_model") != 16:
        continue
    loose[(r["observables"], r["n_spins"], r["model"], r["seed"])].add(
        tuple(r.get(f, "<absent>") for f in KEY3))
ambiguous = {k: v for k, v in loose.items() if len(v) > 1}
ok(not ambiguous,
   "the paper's (observables, n_spins, model, seed) key maps to one "
   "configuration")
for k, v in list(ambiguous.items())[:5]:
    print(f"        {k} spans {len(v)} configurations")

# =====================================================================
print("\n" + "=" * 70)
print("2. QASA forecasting -- which file, and which entangler")
print("=" * 70)
main = os.path.join(QASA, "results", "comparison.csv")
alt = os.path.join(QASA, "results", "comparison_brickwork.csv")
rm_, ra = rows_csv(main), rows_csv(alt)
print(f"  comparison.csv            {len(rm_) if rm_ else 0} rows")
print(f"  comparison_brickwork.csv  {len(ra) if ra else 0} rows"
      if ra is not None else "  comparison_brickwork.csv  ABSENT")

ok(rm_ is not None, "comparison.csv is readable (the paper's source)")

if rm_:
    cols = set(rm_[0])
    ok("entangler" not in cols or len({r.get("entangler") for r in rm_}) == 1,
       "comparison.csv does not mix entanglers")
    if "entangler" not in cols:
        warn("comparison.csv records NO entangler column, so the circuit "
             "behind the forecasting rows cannot be verified from the file. "
             "Paper A Section 3 states brickwork.")
    # duplicate (signal, seed, mode) rows
    seen = defaultdict(list)
    for r in rm_:
        seen[(r["signal"], int(r["seed"]), r["mode"])].append(
            float(r["test_mse"]))
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    ok(not dup, f"no duplicate (signal, seed, mode) rows in comparison.csv "
                f"({len(seen)} slots)")
    for k, v in list(dup.items())[:5]:
        print(f"        {k} -> {v}")

if rm_ and ra:
    a = {(r["signal"], int(r["seed"]), r["mode"]): float(r["test_mse"])
         for r in rm_}
    b = {(r["signal"], int(r["seed"]), r["mode"]): float(r["test_mse"])
         for r in ra}
    shared = sorted(set(a) & set(b))
    if shared:
        diff = [abs(a[k] - b[k]) for k in shared]
        same = sum(1 for d in diff if d < 1e-12)
        print(f"  {len(shared)} slots present in BOTH files; "
              f"{same} identical, {len(shared) - same} differ")
        if same < len(shared):
            warn("the two comparison files disagree on shared slots. "
                 "Which one the paper reports changes the forecasting "
                 "numbers, and Section 3 must name the entangler used.")

# =====================================================================
print("\n" + "=" * 70)
print("3. PHASE 1 CartPole -- does the loader's filter select anything?")
print("=" * 70)
ab_path = os.path.join(PH1, "results", "ablation.jsonl")
ab = None
for i in range(12):
    try:
        with open(ab_path) as f:
            ab = [json.loads(l) for l in f if l.strip()]
        break
    except FileNotFoundError:
        break
    except Exception:
        time.sleep(0.3)
if ab is None:
    warn(f"could not read {ab_path}")
else:
    print(f"  {len(ab)} rows in ablation.jsonl")
    has_dm = sum(1 for r in ab if "d_model" in r)
    ok(has_dm > 0,
       f"rows carry a d_model field ({has_dm}/{len(ab)}); the loader filters "
       f"on `r.get('d_model') is None`, which selects everything when the "
       f"field is absent")
    axes = defaultdict(set)
    for r in ab:
        if r.get("d_model") is None:
            axes[(r["seed"], r["model"])].add(
                (r.get("ablation_axis"), str(r.get("ablation_value"))))
    multi = {k: v for k, v in axes.items() if len(v) > 1}
    ok(not multi,
       f"each (seed, model) selected by the loader maps to one ablation arm "
       f"({len(axes)} slots)")
    for k, v in list(multi.items())[:6]:
        print(f"        seed {k[0]} {k[1]} spans arms {sorted(v)}")

# =====================================================================
print("\n" + "=" * 70)
print("4. Recompute Table 3 from raw records")
print("=" * 70)
fd = rd(os.path.join(HERE, "figures", "figure_data.json"))
if not fd:
    warn("figure_data.json unreadable; cannot cross-check the published table")
else:
    published = {e["setting"]: e for e in fd["fig3_forest"]}
    # Ising settings, recomputed with the strict key
    strict = defaultdict(lambda: defaultdict(dict))
    for r in p3:
        if r.get("d_model") != 16:
            continue
        strict[(r["observables"], r["n_spins"])][r["seed"]][r["model"]] = \
            r["test_rollout"]["1"]
    label = {("z", 2): "Ising chain, 2 spins", ("z", 4): "Ising chain, 4 spins",
             ("z", 6): "Ising chain, 6 spins",
             ("zx", 4): "Ising chain, 4 spins ($zx$)"}
    for key, name in label.items():
        d = strict.get(key)
        if not d:
            warn(f"{name}: no runs found")
            continue
        seeds = [s for s in sorted(d) if len(d[s]) == 2]
        if len(seeds) < 2:
            warn(f"{name}: fewer than 2 paired seeds")
            continue
        got = stats([d[s]["quantum_transformer"] for s in seeds],
                    [d[s]["classical_transformer"] for s in seeds])
        pub = published.get(name)
        if not pub:
            warn(f"{name}: not in figure_data.json")
            continue
        same_n = got["n"] == pub["n"]
        close = abs(got["ratio"] - pub["ratio"]) < 1e-6
        ok(same_n, f"{name}: n recomputes as {got['n']}, published {pub['n']}"
                   + ("" if same_n else "  <-- MORE SEEDS NOW EXIST"))
        if same_n:
            ok(close, f"{name}: ratio {got['ratio']:.6f} vs published "
                      f"{pub['ratio']:.6f}")
        else:
            print(f"        recomputed ratio at n={got['n']}: "
                  f"{got['ratio']:.4f}  CI [{got['ci'][0]:.4f}, "
                  f"{got['ci'][1]:.4f}]  p={got['perm_p']:.3f}  "
                  f"sd={got['sd']:.4f}")

# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
if warnings_:
    print(f"\n{len(warnings_)} warning(s) -- not failures, but read them:")
    for w in warnings_:
        print(f"  - {w}")
if failures:
    print(f"\n{len(failures)} CHECK(S) FAILED. Do not submit until resolved:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll checks passed." if not warnings_ else
      "\nNo failures. Resolve the warnings before submission.")
sys.exit(0)
