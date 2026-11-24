"""Shots needed to resolve a gradient, by entangler topology.

The probe reports a signal-to-noise ratio using the signal it measured at that
shot count. That estimate is contaminated: at low shot counts the measured
|gradient| is inflated by the very sampling noise being quantified, so the SNR
comes out optimistic and the reported hardware ceiling is too generous.

This script corrects it. The true signal is the exact (shots -> infinity)
gradient magnitude, measured separately; the noise is the measured spread at a
given shot count. Their ratio is the honest SNR, and because sampling noise
falls as 1/sqrt(shots) it inverts into the quantity a practitioner actually
needs:

    shots_to_resolve = shots * (noise / exact_signal)^2

That is the number of measurements required before the sign of a gradient
component is distinguishable from noise — the point below which an optimiser is
stepping at random regardless of how healthy the noiseless gradient looked.

    python analyze_shots.py

Reads QASA-repro/results/shot_noise_exact.csv and shot_noise_default-qubit.csv,
writes figures/fig6_shots.* and figures/shots_data.json.
"""
import csv
import json
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
RES = os.path.join(_locate("QASA-repro"), "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 160, "savefig.bbox": "tight", "font.size": 8.5,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "legend.fontsize": 7.5})
INK, GOLD, RED = "#1f3864", "#bf8f00", "#c0392b"
LABEL = {"brickwork": "brickwork (depth 2)",
         "ring_cascade": "ring cascade (depth $\\propto n$)"}


def fmt_shots(x):
    """Below one shot is not a meaningful count, so say so rather than round to 0."""
    return "<1" if x < 1 else f"{x:,.0f}"


def read(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def main():
    exact_rows = read(os.path.join(RES, "shot_noise_exact.csv"))
    shot_rows = [r for r in read(os.path.join(RES, "shot_noise_default-qubit.csv"))
                 if int(r["shots"]) > 0]

    if not exact_rows:
        raise SystemExit(
            "[shots] results/shot_noise_exact.csv not found.\n"
            "        The exact baseline was overwritten by the later sweep. Rerun:\n"
            "          python shot_noise_probe.py --shots 0 --max-qubits 10 "
            "--out results/shot_noise_exact.csv")

    exact = {(r["entangler"], int(r["qubits"])): float(r["signal_absmean"])
             for r in exact_rows}

    recs = []
    for r in shot_rows:
        key = (r["entangler"], int(r["qubits"]))
        if key not in exact:
            continue
        sig, noise, shots = exact[key], float(r["noise_sd"]), int(r["shots"])
        snr = sig / noise if noise > 0 else float("inf")
        recs.append({"entangler": key[0], "qubits": key[1], "shots": shots,
                     "exact_signal": sig, "noise_sd": noise,
                     "snr_true": snr, "snr_reported": float(r["snr"]),
                     "shots_to_resolve": shots / max(snr, 1e-12) ** 2})

    print(f"{'entangler':>13}{'qubits':>8}{'shots':>8}{'exact sig':>12}"
          f"{'noise':>11}{'SNR true':>10}{'(reported)':>12}{'shots for SNR=1':>17}")
    for r in sorted(recs, key=lambda d: (d["entangler"], d["shots"], d["qubits"])):
        print(f"{r['entangler']:>13}{r['qubits']:>8}{r['shots']:>8}"
              f"{r['exact_signal']:>12.3e}{r['noise_sd']:>11.3e}"
              f"{r['snr_true']:>10.2f}{r['snr_reported']:>12.2f}"
              f"{fmt_shots(r['shots_to_resolve']):>17}")

    # ---- the headline: cost ratio at the widest common width
    widths = sorted({r["qubits"] for r in recs})
    wmax = widths[-1]
    cost = {}
    for ent in ("brickwork", "ring_cascade"):
        sub = [r for r in recs if r["entangler"] == ent and r["qubits"] == wmax]
        if sub:
            cost[ent] = float(np.median([r["shots_to_resolve"] for r in sub]))
    print()
    if len(cost) == 2:
        ratio = cost["ring_cascade"] / cost["brickwork"]
        print(f"At {wmax} qubits, resolving one gradient component needs")
        print(f"  brickwork     ~{cost['brickwork']:,.0f} shots")
        print(f"  ring cascade  ~{cost['ring_cascade']:,.0f} shots")
        print(f"  ratio         {ratio:,.0f}x more measurement for the same circuit")
        print("  depth, from an entangler choice that is rarely reported.")

    # ---- ceilings: widest width with true SNR > 1, per shot budget
    print("\nHardware ceiling (widest width with TRUE SNR > 1):")
    ceilings = defaultdict(dict)
    for ent in sorted({r["entangler"] for r in recs}):
        for s in sorted({r["shots"] for r in recs}):
            ok = [r["qubits"] for r in recs
                  if r["entangler"] == ent and r["shots"] == s and r["snr_true"] > 1]
            ceilings[ent][s] = max(ok) if ok else None
            print(f"  {s:>6} shots, {ent:<13} "
                  f"{max(ok) if ok else 'none resolvable'}")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    ax = axes[0]
    for ent, col, mk in (("brickwork", INK, "o"), ("ring_cascade", GOLD, "s")):
        pts = sorted({(r["qubits"], r["exact_signal"])
                      for r in recs if r["entangler"] == ent})
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, mk + "-", color=col, lw=1.8, ms=4,
                    label=LABEL[ent])
    ax.set_yscale("log")
    ax.set_xlabel("qubits"); ax.set_ylabel("exact |gradient|")
    ax.set_title("Gradient magnitude, noiseless")
    ax.legend(loc="lower left")

    ax = axes[1]
    for ent, col, mk in (("brickwork", INK, "o"), ("ring_cascade", GOLD, "s")):
        for s, alpha in zip(sorted({r["shots"] for r in recs}), (0.4, 0.7, 1.0)):
            sub = sorted([r for r in recs
                          if r["entangler"] == ent and r["shots"] == s],
                         key=lambda d: d["qubits"])
            if sub:
                ax.plot([r["qubits"] for r in sub],
                        [r["shots_to_resolve"] for r in sub],
                        mk + "-", color=col, lw=1.4, ms=3.5, alpha=alpha,
                        label=f"{LABEL[ent]}, {s} shots" if alpha == 1.0 else None)
    ax.set_yscale("log")
    ax.axhline(1000, color=RED, ls="--", lw=1)
    ax.text(0.02, 0.06, "typical hardware budget", transform=ax.transAxes,
            fontsize=6.5, color=RED)
    ax.set_xlabel("qubits")
    ax.set_ylabel("shots to resolve one gradient")
    ax.set_title("Measurement cost of the same circuit depth")
    ax.legend(loc="upper left")

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig6_shots.{ext}"))
    plt.close(fig)

    with open(os.path.join(FIG, "shots_data.json"), "w") as f:
        json.dump({"records": recs, "cost_at_max_width": cost,
                   "max_width": wmax,
                   "ceilings": {k: v for k, v in ceilings.items()}}, f, indent=1)
    print(f"\n-> {FIG}/fig6_shots.png")


if __name__ == "__main__":
    main()
