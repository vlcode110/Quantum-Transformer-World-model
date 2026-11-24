"""The surrogate sweep: does the token-mixing block bind at all?

The experiment was built to ask whether a classical map over the circuit's own
frequency spectrum matches the circuit. It answers that, but the sweep also
answers a prior question that turns out to matter more: whether the component
being compared constrains performance in the first place.

If swinging the block's capacity over a wide range moves the error less than
changing the random seed does, then nothing placed in that slot can be
distinguished from anything else, and a quantum-versus-classical comparison at
that position is uninformative however many seeds are run. That is a saturation
problem at the level of the component rather than the model, and it is cheap to
test for.

    python analyze_surrogate.py

Reads QASA-repro/results/fourier_surrogate.csv, writes figures/fig5_surrogate.*
and figures/surrogate_data.json.
"""
import csv
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
SRC = os.path.join(_locate("QASA-repro"), "results", "fourier_surrogate.csv")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 160, "savefig.bbox": "tight", "font.size": 8.5,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "legend.fontsize": 7.5})
INK, GOLD, RED, GREY = "#1f3864", "#bf8f00", "#c0392b", "#7f7f7f"


def load(signal="mackey_glass", min_epochs=10):
    """Full runs only: a smoke run appends to the same file."""
    rows = []
    for r in csv.DictReader(open(SRC)):
        if r["signal"] != signal:
            continue
        if int(r.get("epochs") or 0) < min_epochs or int(r.get("limit") or 0):
            continue
        rows.append(r)
    by = defaultdict(dict)
    meta = {}
    for r in rows:
        key = r["mode"] if r["mode"] != "fourier" else f"fourier{r['n_freq']}"
        by[int(r["seed"])][key] = float(r["test_mse"])
        meta[key] = int(r["core_params"])
    seeds = sorted(s for s in by if "quantum" in by[s])
    return by, meta, seeds


def paired(by, seeds, a, b, n_boot=20000):
    xa = np.array([by[s][a] for s in seeds])
    xb = np.array([by[s][b] for s in seeds])
    lr = np.log(xa / xb)
    n = len(lr)
    rng = np.random.default_rng(0)
    bs = np.exp(lr[rng.integers(0, n, (n_boot, n))].mean(axis=1))
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    null = np.abs((signs * np.abs(lr)).mean(axis=1))
    return {"ratio": float(np.exp(lr.mean())),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p": float((null >= abs(lr.mean()) - 1e-15).mean()),
            "wins": int((xa < xb).sum()), "n": n}


def main():
    by, meta, seeds = load()
    if len(seeds) < 3:
        raise SystemExit(f"[surrogate] only {len(seeds)} seeds; run more first.")
    print(f"seeds: {seeds}")

    fourier_keys = sorted((k for k in meta if k.startswith("fourier")),
                          key=lambda k: meta[k])
    out = {"seeds": seeds, "core_params": meta, "vs_quantum": {}}

    print("\nPaired against the quantum PQC "
          f"({meta['quantum']} core parameters)")
    print(f"{'arm':>12}{'core':>7}{'ratio':>9}{'  95% CI':>19}{'p':>8}{'wins':>7}")
    for k in fourier_keys + ["classical"]:
        if k not in meta:
            continue
        d = paired(by, seeds, k, "quantum")
        out["vs_quantum"][k] = d
        print(f"{k:>12}{meta[k]:>7}{d['ratio']:>9.3f}   "
              f"[{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]{d['p']:>8.3f}"
              f"{d['wins']:>5}/{d['n']}")

    # ---- does the block's capacity bind at all?
    cps = np.array([meta[k] for k in fourier_keys], float)
    means = np.array([np.mean([by[s][k] for s in seeds]) for k in fourier_keys])
    slope = float(np.polyfit(np.log(cps), np.log(means), 1)[0])
    sweep_sd = float(np.log(means).std(ddof=1))
    qseed_sd = float(np.log([by[s]["quantum"] for s in seeds]).std(ddof=1))
    out["capacity"] = {"core_params": cps.tolist(), "mean_mse": means.tolist(),
                       "loglog_slope": slope, "sweep_spread_log": sweep_sd,
                       "quantum_seed_spread_log": qseed_sd,
                       "param_range": float(cps.max() / cps.min()),
                       "best_over_worst": float(means.min() / means.max())}

    print(f"\nCapacity of the block, swept {cps.max()/cps.min():.0f}x "
          f"({cps.min():.0f} to {cps.max():.0f} parameters)")
    print(f"  log-log slope                 {slope:+.4f}   (0 = does not bind)")
    print(f"  best / worst across the sweep {means.min()/means.max():.3f}")
    print(f"  spread across the sweep       {sweep_sd:.3f}  (sd of log mean)")
    print(f"  spread across seeds, quantum  {qseed_sd:.3f}  (sd of log)")
    verdict = sweep_sd < qseed_sd
    print(f"\n  {'SEED noise exceeds CAPACITY effect' if verdict else 'capacity binds'}"
          f" -> the block {'does NOT' if verdict else 'does'} constrain performance here.")
    out["capacity"]["block_binds"] = not verdict

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    ax = axes[0]
    qv = np.array([by[s]["quantum"] for s in seeds])
    lo, hi = np.percentile(qv, [10, 90])
    ax.axhspan(lo, hi, color=INK, alpha=0.13, lw=0,
               label="quantum circuit, 10–90% over seeds")
    ax.axhline(qv.mean(), color=INK, lw=1.6, label=f"quantum ({meta['quantum']} params)")
    per_seed = np.array([[by[s][k] for k in fourier_keys] for s in seeds])
    for row in per_seed:
        ax.plot(cps, row, "-", color=GREY, lw=0.7, alpha=0.5)
    ax.plot(cps, means, "o-", color=GOLD, lw=1.9, ms=4.5,
            label="classical Fourier surrogate")
    if "classical" in meta:
        cvals = np.array([by[s]["classical"] for s in seeds])
        ax.plot([meta["classical"]], [cvals.mean()], "s", color=RED, ms=6,
                label=f"classical bottleneck ({meta['classical']} params)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("trainable parameters in the token-mixing block")
    ax.set_ylabel("test MSE")
    ax.set_title("Nothing in this slot matters")
    ax.legend(loc="upper left", fontsize=6.5)

    ax = axes[1]
    ks = fourier_keys + (["classical"] if "classical" in meta else [])
    ys = np.arange(len(ks))[::-1]
    for y, k in zip(ys, ks):
        d = out["vs_quantum"][k]
        ax.plot(d["ci"], [y, y], "-", color=INK, lw=1.4, alpha=0.85)
        ax.plot([d["ratio"]], [y], "o", color=INK, ms=5)
    ax.axvline(1.0, color=RED, ls="--", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{k} ({meta[k]}p)" for k in ks], fontsize=7)
    ax.set_xlabel("error relative to the quantum circuit")
    ax.set_title("Every arm overlaps parity")
    ax.grid(axis="y", alpha=0)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig5_surrogate.{ext}"))
    plt.close(fig)

    with open(os.path.join(FIG, "surrogate_data.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {FIG}/fig5_surrogate.png")


if __name__ == "__main__":
    main()
