"""Paired test: does removing entanglement change offline prediction?

    python ablation_entanglement.py

analyze.py pairs the quantum arm against the CLASSICAL twin, so it has
no block for this comparison -- both arms here are quantum, differing
only in whether the circuit contains two-qubit gates. Parameter count is
identical (28 either way), so the comparison is otherwise exact.

Reads result.json only. Seconds.
"""
import glob
import itertools
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = dict(task="cartpole", d_model=32, context=4, n_qubits=6,
           n_layers=2, q_layers=4, model="quantum_transformer")


def rd(p, attempts=15):
    last = None
    for i in range(attempts):
        try:
            return json.load(open(p))
        except FileNotFoundError:
            return None
        except Exception as e:
            last = e
            time.sleep(0.3 + 0.2 * i)
    print(f"  [io] could not read {os.path.basename(os.path.dirname(p))}: "
          f"{type(last).__name__}")
    return None


def paired(a, b, label):
    """Exact sign-flip permutation test on paired log-ratios."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    lr = np.log(a / b)
    n = len(lr)
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    null = np.abs((signs * np.abs(lr)).mean(axis=1))
    p = float((null >= abs(lr.mean()) - 1e-15).mean())
    gm = float(np.exp(lr.mean()))
    rng = np.random.default_rng(0)
    boot = np.exp(lr[rng.integers(0, n, (40000, n))].mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    worse = int((a > b).sum())
    print(f"  {label}")
    print(f"    geometric-mean ratio none/brickwork = {gm:.4f}  "
          f"95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"    exact permutation p = {p:.4f}   "
          f"(smallest attainable at n={n} is {2 / 2**n:.4f})")
    print(f"    entanglement-free arm worse in {worse}/{n} seeds")
    # equivalence framing: what difference could this sample size detect?
    sd = float(lr.std(ddof=1))
    mde = 2.80 * sd / np.sqrt(n)
    print(f"    sd of paired log-ratio {sd:.4f}; at n={n} this design could "
          f"detect a\n      ratio of {np.exp(mde):.3f} or larger at 80% power. "
          f"Differences smaller\n      than that are not resolvable here.")
    return gm, p


def main():
    arms = {"brickwork": {}, "none": {}}
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*", "result.json"))):
        r = rd(p)
        if not r:
            continue
        if any(r.get(k) != v for k, v in CFG.items()):
            continue
        ent = r.get("entangler")
        if ent in arms:
            arms[ent][r["seed"]] = r["test_rollout"]

    shared = sorted(set(arms["brickwork"]) & set(arms["none"]))
    print(f"\nentanglement ablation, {CFG['task']}, d_model={CFG['d_model']}, "
          f"K={CFG['context']}, n_qubits={CFG['n_qubits']}")
    print(f"seeds with both arms: {shared}")
    print(f"  brickwork seeds: {sorted(arms['brickwork'])}")
    print(f"  none seeds:      {sorted(arms['none'])}\n")
    if len(shared) < 3:
        raise SystemExit("fewer than 3 paired seeds; nothing to test")

    for k in ("1", "5", "10", "20"):
        if not all(k in arms["none"][s] and k in arms["brickwork"][s]
                   for s in shared):
            continue
        paired([arms["none"][s][k] for s in shared],
               [arms["brickwork"][s][k] for s in shared],
               f"{k}-step open-loop error")
        print()

    print("Reading: a ratio near 1 with a p-value near 1 means the two-qubit")
    print("gates contribute nothing measurable at this sample size. That is a")
    print("statement about THIS circuit on THIS task, not about entanglement in")
    print("general -- and it is the outcome Bowles et al. (arXiv:2403.07059)")
    print("report as common. Note the minimum detectable ratio above before")
    print("calling it equivalence: a null here bounds the effect, it does not")
    print("exclude one.")


if __name__ == "__main__":
    main()
