"""How much evidence is a small-n quantum-vs-classical comparison?

The claim that these studies are underpowered should be a measurement, not an
assertion backed by two anecdotes. This script turns it into one, using the
project's own paired runs as empirical populations rather than assuming
normality or inventing a variance.

Three questions, all answered by resampling:

  A. SAMPLING VARIABILITY — if the true effect is whatever our largest sample
     says it is, how much does the number a researcher reports move around as a
     function of how many seeds they ran?

  B. FALSE POSITIVES — if there were NO effect, how often would a study at n
     seeds report an advantage of a given size, and how often would it clear
     p < 0.05? The null is built by exchanging the quantum and classical labels
     within each seed, which is the exact null the paired test assumes, so this
     needs no distributional assumption at all.

  C. POWER — given the dispersion actually observed, how many seeds are needed
     to detect a true effect of 10%, 20%, 30% at 80% power?

    python power_analysis.py

Writes figures/fig4_power.{png,pdf} and figures/power_data.json.
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
DOCS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
QASA = _locate("QASA-repro")
PH3 = os.path.join(DOCS, "PhD", "qtwm", "Phase_3_Quantum_Native")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160, "savefig.bbox": "tight", "font.size": 8.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7.5,
})
INK, GOLD, RED, GREY = "#1f3864", "#bf8f00", "#c0392b", "#7f7f7f"
RNG = np.random.default_rng(20261002)
N_REP = 4000          # resamples per (n, condition)


# --------------------------------------------------------------- the datasets
def load_populations(min_seeds=10):
    """Each population is a vector of per-seed log-ratios log(quantum/classical).

    The log-ratio is the natural unit: ratios are multiplicative, so 0.90 and
    1.11 are equal and opposite effects and an arithmetic mean would not say so.

    min_seeds=10 selects the populations large enough to support a power
    analysis; min_seeds=3 returns every setting, which is what the dispersion
    table uses in order to show how uninformative the small ones are.
    """
    pops = {}

    rows = list(csv.DictReader(open(os.path.join(QASA, "results", "comparison.csv"))))
    by = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by[r["signal"]][int(r["seed"])][r["mode"]] = float(r["test_mse"])
    nice = {"mackey_glass": "Mackey-Glass", "sine": "Sine",
            "noisy_sine": "Noisy sine"}
    for sig in sorted(by):
        seeds = [s for s in sorted(by[sig]) if len(by[sig][s]) == 2]
        if len(seeds) >= min_seeds:
            pops[f"{nice.get(sig, sig)} ({len(seeds)} seeds)"] = np.array(
                [math.log(by[sig][s]["quantum"] / by[sig][s]["classical"])
                 for s in seeds])

    p3 = defaultdict(lambda: defaultdict(dict))
    for p in glob.glob(os.path.join(PH3, "runs", "*", "result.json")):
        r = json.load(open(p))
        if r["d_model"] != 16:
            continue
        p3[(r["observables"], r["n_spins"])][r["seed"]][r["model"]] = \
            r["test_rollout"]["1"]
    for (obs, n_spins), v in sorted(p3.items()):
        seeds = [s for s in sorted(v) if len(v[s]) == 2]
        if len(seeds) >= min_seeds:
            label = f"Ising {n_spins}-spin, ${obs}$"
            pops[f"{label} ({len(seeds)} seeds)"] = np.array(
                [math.log(v[s]["quantum_transformer"] /
                          v[s]["classical_transformer"]) for s in seeds])
    return pops


# ------------------------------------------------------------------- the test
_SIGN_CACHE = {}


def sign_matrix(n, n_perm=512):
    """Sign assignments for the paired permutation test.

    Exact enumeration while 2^n stays small, sampled above that. Cached and
    shared across every replicate, which is what makes the whole analysis
    tractable: the test is then a single matrix product rather than a loop.
    """
    key = (n, n_perm)
    if key not in _SIGN_CACHE:
        if 2 ** n <= n_perm * 2:
            s = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        else:
            s = RNG.choice([-1.0, 1.0], size=(n_perm, n))
        _SIGN_CACHE[key] = s
    return _SIGN_CACHE[key]


def perm_p_batch(draws):
    """Two-sided paired permutation p-values for a whole batch at once.

    draws is (reps, n) of log-ratios; returns (reps,). The null distribution of
    the mean under sign exchange is |draws| @ signs.T / n, so every replicate
    and every permutation is handled by one matmul.
    """
    reps, n = draws.shape
    signs = sign_matrix(n)
    obs = np.abs(draws.mean(axis=1))                      # (reps,)
    null = np.abs(np.abs(draws) @ signs.T) / n            # (reps, n_perm)
    return (null >= obs[:, None] - 1e-15).mean(axis=1)


def perm_p_fast(r):
    return float(perm_p_batch(np.asarray(r, float)[None, :])[0])


# ------------------------------------------------- A. sampling variability
def sampling_variability(pop, ns):
    """Subsample without replacement: the estimate you would have published."""
    out = {}
    for n in ns:
        if n > len(pop):
            continue
        est = np.array([RNG.choice(pop, n, replace=False).mean()
                        for _ in range(N_REP)])
        out[n] = {"p05": float(np.exp(np.percentile(est, 5))),
                  "p50": float(np.exp(np.percentile(est, 50))),
                  "p95": float(np.exp(np.percentile(est, 95)))}
    return out


# ----------------------------------------------------- B. false positives
def false_positives(pop, ns, thresholds=(0.95, 0.90, 0.85, 0.80)):
    """Under the paired null, how often does a study at n seeds 'find' an effect?

    The null is constructed by randomising the sign of each log-ratio, which is
    exactly the exchangeability the paired test assumes: it keeps the real
    dispersion of these models and removes only the direction.
    """
    out = {}
    resid = pop - pop.mean()           # centre: no true effect
    for n in ns:
        draws = (RNG.choice(resid, (N_REP, n), replace=True)
                 * RNG.choice([-1.0, 1.0], (N_REP, n)))
        ratios = np.exp(draws.mean(axis=1))
        ps = perm_p_batch(draws)
        out[n] = {
            "p_sig": float((ps < 0.05).mean()),
            **{f"p_ratio_below_{t}": float((ratios < t).mean())
               for t in thresholds},
            # reporting a gain AND clearing significance: the combination that
            # actually reaches print
            **{f"p_joint_{t}": float(((ratios < t) & (ps < 0.05)).mean())
               for t in thresholds},
        }
    return out


# --------------------------------------------------------------- C. power
def power_curve(pop, ns, effects=(0.90, 0.80, 0.70)):
    """Required n to detect a true effect, given the dispersion observed here.

    Residuals are resampled from the real data and a true effect is added, so
    the dispersion is the models' own rather than an assumed normal.
    """
    resid = pop - pop.mean()
    out = {}
    for eff in effects:
        delta = math.log(eff)
        curve = {}
        for n in ns:
            draws = delta + (RNG.choice(resid, (N_REP, n), replace=True)
                             * RNG.choice([-1.0, 1.0], (N_REP, n)))
            curve[n] = float((perm_p_batch(draws) < 0.05).mean())
        out[eff] = curve
    return out


def required_n(curve, target=0.80):
    for n in sorted(curve):
        if curve[n] >= target:
            return n
    return None


def sd_interval(pop, n_boot=20000):
    """Bootstrap interval on the dispersion itself.

    Dispersion is the quantity this paper asks the field to report, so it would
    be inconsistent to report it as a point estimate. Bootstrap rather than the
    chi-square interval, to avoid assuming normality of the log-ratios and to
    avoid a scipy dependency.
    """
    n = len(pop)
    idx = RNG.integers(0, n, (n_boot, n))
    sds = pop[idx].std(axis=1, ddof=1)
    return float(np.percentile(sds, 2.5)), float(np.percentile(sds, 97.5))


def pilot_reliability(pop, pilot_sizes=(3, 5, 8, 10, 13), effect=0.90):
    """Can a small pilot tell you how many seeds you will need?

    Advising practitioners to 'estimate dispersion from a pilot' is only useful
    if the estimate is stable. This measures how much the implied sample size
    moves when the pilot is small, using the largest population available as
    ground truth.
    """
    out = {}
    delta = abs(math.log(effect))
    for k in pilot_sizes:
        if k >= len(pop):
            continue
        est = np.array([RNG.choice(pop, k, replace=False).std(ddof=1)
                        for _ in range(5000)])
        # the n implied by a normal-approximation two-sided test at 95%
        implied = (1.96 * est / delta) ** 2
        out[k] = {"sd_median": float(np.median(est)),
                  "sd_p05": float(np.percentile(est, 5)),
                  "sd_p95": float(np.percentile(est, 95)),
                  "implied_n_p05": float(np.percentile(implied, 5)),
                  "implied_n_p95": float(np.percentile(implied, 95))}
    return out


# ------------------------------------------------------------------- figure
def make_figure(pops, A, B, C, ns):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    main = list(pops)[0]

    ax = axes[0]
    a = A[main]
    xs = sorted(a)
    ax.fill_between(xs, [a[n]["p05"] for n in xs], [a[n]["p95"] for n in xs],
                    color=INK, alpha=0.18, lw=0)
    ax.plot(xs, [a[n]["p50"] for n in xs], "-", color=INK, lw=1.8)
    ax.axhline(1.0, color=RED, ls="--", lw=1)
    ax.set_xlabel("seeds run"); ax.set_ylabel("ratio a study would report")
    ax.set_title("A. What gets reported\n(90% interval)")

    ax = axes[1]
    b = B[main]
    xs = sorted(b)
    for t, col, ls in ((0.90, INK, "-"), (0.85, GOLD, "--"), (0.80, GREY, ":")):
        ax.plot(xs, [b[n][f"p_ratio_below_{t}"] * 100 for n in xs], ls,
                color=col, lw=1.8, label=f"reports $\\geq${round((1-t)*100)}% gain")
    ax.plot(xs, [b[n]["p_sig"] * 100 for n in xs], "-.", color=RED, lw=1.5,
            label="reaches $p<0.05$")
    ax.set_xlabel("seeds run"); ax.set_ylabel("% of studies")
    ax.set_title("B. False positives\n(no true effect)")
    ax.legend(loc="upper right")

    ax = axes[2]
    for eff, col, mk in ((0.90, INK, "o"), (0.80, GOLD, "s"), (0.70, GREY, "^")):
        curve = C[main][eff]
        xs = sorted(curve)
        ax.plot(xs, [curve[n] * 100 for n in xs], mk + "-", color=col, lw=1.6,
                ms=3.5, label=f"true effect {round((1-eff)*100)}%")
    ax.axhline(80, color=RED, ls="--", lw=1)
    ax.set_xlabel("seeds run"); ax.set_ylabel("% detected at $p<0.05$")
    ax.set_title("C. Power\n(dispersion as observed)")
    ax.legend(loc="lower right")

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig4_power.{ext}"))
    plt.close(fig)


def main():
    pops = load_populations(min_seeds=10)
    allpops = load_populations(min_seeds=3)
    ns = [3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 60, 100, 150, 250, 400]

    print("DISPERSION, every setting (bootstrap 95% interval on sd)")
    disp = {}
    for k, v in sorted(allpops.items(), key=lambda kv: -kv[1].std(ddof=1)):
        s = v.std(ddof=1)
        lo, hi = sd_interval(v)
        disp[k] = {"n": len(v), "ratio": float(np.exp(v.mean())),
                   "sd": float(s), "sd_ci": [lo, hi]}
        print(f"  {k:<30} n={len(v):>2}  sd {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")

    main_pop = max(pops, key=lambda k: len(pops[k]))
    print(f"\nPILOT RELIABILITY (ground truth: {main_pop}, "
          f"sd={pops[main_pop].std(ddof=1):.3f})")
    pilot = pilot_reliability(pops[main_pop])
    for k, d in pilot.items():
        print(f"  pilot n={k:>2}: sd {d['sd_median']:.3f} "
              f"[{d['sd_p05']:.3f}, {d['sd_p95']:.3f}]  -> implied n for a 10% "
              f"effect spans [{d['implied_n_p05']:.0f}, {d['implied_n_p95']:.0f}]")

    A, B, C = {}, {}, {}
    for k, v in pops.items():
        print(f"\n--- {k}")
        A[k] = sampling_variability(v, [n for n in ns if n <= len(v)])
        B[k] = false_positives(v, [n for n in ns if n <= 30])
        C[k] = power_curve(v, ns)
        print("  required n for 80% power:",
              {f"{round((1-e)*100)}%": required_n(C[k][e]) for e in C[k]})
        for nlab in (5, 10, 13, 20):
            if nlab in B[k]:
                b = B[k][nlab]
                print(f"  null, n={nlab:>2}: report 10%+ gain "
                      f"{b['p_ratio_below_0.9']*100:>4.1f}%   20%+ "
                      f"{b['p_ratio_below_0.8']*100:>4.1f}%   "
                      f"p<0.05 {b['p_sig']*100:>4.1f}%   "
                      f"both(10%+ & sig) {b['p_joint_0.9']*100:>4.1f}%")

    make_figure(pops, A, B, C, ns)
    with open(os.path.join(FIG, "power_data.json"), "w") as f:
        json.dump({"dispersion": disp, "pilot_reliability": pilot,
                   "sampling_variability": A, "false_positives": B,
                   "required_n": {k: {str(e): required_n(c) for e, c in v.items()}
                                  for k, v in C.items()},
                   "power": {k: {str(e): c for e, c in v.items()} for k, v in C.items()}},
                  f, indent=1)
    print(f"\n-> {FIG}/fig4_power.png")


if __name__ == "__main__":
    main()
