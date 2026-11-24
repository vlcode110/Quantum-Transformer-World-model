"""Figures for Paper A, computed from the raw run records only.

Every number in this script is recomputed from comparison.csv, ablation.jsonl
and the Phase 3 result.json files. Nothing is copied from a console log or from
an earlier report, because two figures quoted during this project turned out to
be wrong and the paper's whole argument is that this happens.

    python make_figures.py

Writes PNG and PDF into ./figures and a machine-readable dump of every number
plotted into ./figures/figure_data.json, so the numbers in the text can be
checked against the numbers in the plots.
"""
import csv
import glob
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
DOCS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # ~/Documents
QASA = _locate("QASA-repro")
PH1 = os.path.join(DOCS, "PhD", "qtwm", "Phase_1_Scaffold_test")
PH3 = os.path.join(DOCS, "PhD", "qtwm", "Phase_3_Quantum_Native")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160, "savefig.bbox": "tight", "font.size": 8.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.frameon": False, "legend.fontsize": 7.5,
})
INK = "#1f3864"
GOLD = "#bf8f00"
RED = "#c0392b"
GREY = "#7f7f7f"

DUMP = {}


# ----------------------------------------------------------------- statistics
def geo_ratio(q, c):
    return float(np.exp(np.mean(np.log(np.asarray(q) / np.asarray(c)))))


def boot_ci(q, c, n_boot=20000, seed=0):
    """Bootstrap interval on the geometric-mean ratio.

    Geometric because ratios are multiplicative: 0.90 and 1.11 are equal and
    opposite, and an arithmetic mean does not say so.
    """
    r = np.log(np.asarray(q) / np.asarray(c))
    n = len(r)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    means = r[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.exp(lo)), float(np.exp(hi))


def perm_p(q, c, cap=20):
    """Paired permutation test on log-ratios; exact by sign enumeration."""
    r = np.log(np.asarray(q) / np.asarray(c))
    n = len(r)
    if n < 2:
        return float("nan")
    if n <= cap:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
        null = (signs * np.abs(r)).mean(axis=1)
    else:
        rng = np.random.default_rng(0)
        null = (rng.choice([-1.0, 1.0], size=(200000, n)) * np.abs(r)).mean(axis=1)
    return float((np.abs(null) >= abs(r.mean()) - 1e-15).mean())


def sign_p(q, c):
    n = len(q)
    wins = int((np.asarray(q) < np.asarray(c)).sum())
    pmf = lambda k: math.comb(n, k) * 0.5 ** n
    o = pmf(wins)
    return wins, float(min(sum(pmf(k) for k in range(n + 1) if pmf(k) <= o + 1e-15), 1.0))


# --------------------------------------------------------------- data loading
def load_qasa():
    """signal -> (seeds, quantum errors, classical errors), seed-ordered."""
    path = os.path.join(QASA, "results", "comparison.csv")
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by[r["signal"]][int(r["seed"])][r["mode"]] = float(r["test_mse"])
    out = {}
    for sig, seeds in by.items():
        ok = [s for s in sorted(seeds) if len(seeds[s]) == 2]
        out[sig] = (ok,
                    np.array([seeds[s]["quantum"] for s in ok]),
                    np.array([seeds[s]["classical"] for s in ok]))
    return out


def load_phase3():
    """(observables, n_spins) -> (seeds, quantum, classical) at k=1, d_model 16."""
    by = defaultdict(lambda: defaultdict(dict))
    for p in glob.glob(os.path.join(PH3, "runs", "*", "result.json")):
        r = json.load(open(p))
        if r["d_model"] != 16:
            continue
        key = (r["observables"], r["n_spins"])
        by[key][r["seed"]][r["model"]] = r["test_rollout"]["1"]
    out = {}
    for key, seeds in by.items():
        ok = [s for s in sorted(seeds) if len(seeds[s]) == 2]
        if ok:
            out[key] = (ok,
                        np.array([seeds[s]["quantum_transformer"] for s in ok]),
                        np.array([seeds[s]["classical_transformer"] for s in ok]))
    return out


def load_phase1_cartpole():
    path = os.path.join(PH1, "results", "ablation.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("d_model") is None]   # the main arm
    by = defaultdict(dict)
    for r in rows:
        by[r["seed"]][r["model"]] = r["test_rollout"]["1"]
    ok = [s for s in sorted(by) if len(by[s]) == 2]
    return (ok,
            np.array([by[s]["quantum_transformer"] for s in ok]),
            np.array([by[s]["classical_transformer"] for s in ok]))


def load_gradients():
    path = os.path.join(QASA, "results", "gradient_variance.csv")
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(dict)
    for r in rows:
        by[(r["entangler"], r["cost"], int(r["layers"]))][int(r["qubits"])] = \
            float(r["grad_var"])
    return by


# ------------------------------------------------- figure 1: the seed collapse
def figure_seed_collapse(qasa, ph3):
    """The paper's central figure: the estimate you would have reported had you
    stopped after n seeds. Both curves walk toward parity."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # ---- panel A: Mackey-Glass, cumulative estimate
    ax = axes[0]
    seeds, q, c = qasa["mackey_glass"]
    ns, est, los, his = [], [], [], []
    for n in range(3, len(q) + 1):
        ns.append(n)
        est.append(geo_ratio(q[:n], c[:n]))
        lo, hi = boot_ci(q[:n], c[:n])
        los.append(lo); his.append(hi)
    ax.fill_between(ns, los, his, color=INK, alpha=0.15, lw=0)
    ax.plot(ns, est, "-", color=INK, lw=1.8)
    ax.axhline(1.0, color=RED, ls="--", lw=1)
    ax.axvline(13, color=GREY, ls=":", lw=1)
    ax.annotate("originally\nreported here", xy=(13, 0.70),
                xytext=(15.5, 0.63), fontsize=7, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.7))
    ax.set_xlabel("seeds included")
    ax.set_ylabel("quantum error / classical error")
    ax.set_title("Chaotic forecasting (Mackey-Glass)")
    DUMP["fig1a_mackey_glass"] = {"n": ns, "estimate": est, "ci_lo": los, "ci_hi": his}

    # ---- panel B: Phase 3, separation between observable arms
    ax = axes[1]
    zs, zxs = ph3[("z", 4)], ph3[("zx", 4)]
    n_max = min(len(zs[0]), len(zxs[0]))
    ns, sep, zr, zxr = [], [], [], []
    for n in range(3, n_max + 1):
        a = geo_ratio(zs[1][:n], zs[2][:n])
        b = geo_ratio(zxs[1][:n], zxs[2][:n])
        ns.append(n); zr.append(a); zxr.append(b); sep.append((b - a) * 100)
    ax.plot(ns, zr, "o-", color=INK, lw=1.6, ms=3.5,
            label="non-Markovian ($z$ only)")
    ax.plot(ns, zxr, "s-", color=GOLD, lw=1.6, ms=3.5,
            label="nearer-Markovian ($z$ and $x$)")
    ax.axhline(1.0, color=RED, ls="--", lw=1)
    ax.set_xlabel("seeds per arm")
    ax.set_ylabel("quantum error / classical error")
    ax.set_title("Quantum-native dynamics (4-spin Ising)")
    ax.legend(loc="lower right")
    DUMP["fig1b_phase3"] = {"n": ns, "z": zr, "zx": zxr, "separation_pts": sep}

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig1_seed_collapse.{ext}"))
    plt.close(fig)
    print("  fig1_seed_collapse")


# ------------------------------------------ figure 2: the entangler confound
def figure_entangler(grads):
    """Gradient variance against width, split by cost locality and entangler.

    The local-cost rows are the point: brickwork reproduces the predicted
    absence of decay at shallow depth, the sequential ring cascade does not.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    styles = {"brickwork": (INK, "o", "-"), "ring_cascade": (GOLD, "s", "--")}
    label = {"brickwork": "brickwork (depth 2)",
             "ring_cascade": "ring cascade (depth $\\propto n$)"}
    dump = {}
    for ax, cost in zip(axes, ["local", "global"]):
        for ent, (col, mk, ls) in styles.items():
            key = (ent, cost, 2)
            if key not in grads:
                continue
            qs = sorted(grads[key])
            v = np.array([grads[key][q] for q in qs])
            slope = np.polyfit(qs, np.log(v), 1)[0]
            ax.plot(qs, v, mk + ls, color=col, lw=1.6, ms=4,
                    label=f"{label[ent]}, $\\times${np.exp(slope):.2f}/qubit")
            dump[f"{cost}_{ent}"] = {"qubits": qs, "grad_var": v.tolist(),
                                     "decay_per_qubit": float(np.exp(slope))}
        ax.set_yscale("log")
        ax.set_xlabel("qubits")
        ax.set_title(f"{cost} cost function, 2 entangling layers")
        ax.legend(loc="lower left")
    axes[0].set_ylabel("gradient variance")
    DUMP["fig2_entangler"] = dump
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig2_entangler.{ext}"))
    plt.close(fig)
    print("  fig2_entangler")


# --------------------------------------------- figure 3: every comparison made
def figure_forest(qasa, ph3, ph1):
    """One interval per comparison run in the project. Nothing clears parity."""
    items = []
    s, q, c = ph1
    items.append(("Latent dynamics, CartPole", q, c))
    for n in (2, 4, 6):
        if ("z", n) in ph3:
            _, q, c = ph3[("z", n)]
            items.append((f"Ising chain, {n} spins", q, c))
    if ("zx", 4) in ph3:
        _, q, c = ph3[("zx", 4)]
        items.append(("Ising chain, 4 spins ($zx$)", q, c))
    for sig, name in (("sine", "Sine"), ("noisy_sine", "Noisy sine"),
                      ("mackey_glass", "Mackey-Glass")):
        if sig in qasa:
            _, q, c = qasa[sig]
            items.append((f"Forecasting, {name.lower()}", q, c))

    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(items) + 1.0))
    rows = []
    for i, (name, q, c) in enumerate(items):
        y = len(items) - i
        r = geo_ratio(q, c)
        lo, hi = boot_ci(q, c)
        wins, sp = sign_p(q, c)
        pp = perm_p(q, c)
        ax.plot([lo, hi], [y, y], "-", color=INK, lw=1.4, alpha=0.8)
        ax.plot([r], [y], "o", color=INK, ms=5)
        ax.text(2.25, y, f"{r:.3f}  [{lo:.2f}, {hi:.2f}]   $n$={len(q)}",
                va="center", fontsize=7, color="#333333")
        rows.append({"setting": name, "n": len(q), "ratio": r,
                     "ci": [lo, hi], "quantum_ahead": wins,
                     "sign_p": sp, "perm_p": pp})
    ax.axvline(1.0, color=RED, ls="--", lw=1)
    ax.set_yticks(range(1, len(items) + 1))
    ax.set_yticklabels([n for n, _, _ in items][::-1])
    ax.set_xlim(0.45, 3.3)
    ax.set_xlabel("quantum error / classical error  (below 1 favours quantum)")
    ax.set_title("Every quantum-classical comparison in the project")
    ax.grid(axis="y", alpha=0)
    DUMP["fig3_forest"] = rows
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig3_forest.{ext}"))
    plt.close(fig)
    print("  fig3_forest")
    return rows


def main():
    print("loading…")
    qasa = load_qasa()
    ph3 = load_phase3()
    ph1 = load_phase1_cartpole()
    grads = load_gradients()
    print(f"  qasa signals: {sorted(qasa)}")
    print(f"  phase3 cells: {sorted(ph3)}")
    print(f"  phase1 cartpole seeds: {ph1[0]}")
    print("figures…")
    figure_seed_collapse(qasa, ph3)
    figure_entangler(grads)
    rows = figure_forest(qasa, ph3, ph1)

    with open(os.path.join(FIG, "figure_data.json"), "w") as f:
        json.dump(DUMP, f, indent=1)
    print(f"\n-> {FIG}")
    print("\nForest plot contents:")
    for r in rows:
        print(f"  {r['setting']:<34} n={r['n']:>2}  ratio {r['ratio']:.3f}  "
              f"CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  "
              f"ahead {r['quantum_ahead']}/{r['n']}  perm p={r['perm_p']:.3f}")


if __name__ == "__main__":
    main()
