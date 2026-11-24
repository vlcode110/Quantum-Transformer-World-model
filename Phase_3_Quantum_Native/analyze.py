"""The scaling analysis: does the gap widen as the quantum system grows?

This is the experiment. Not "who wins at n=6", but whether the two
architectures diverge as the system being modelled gets larger. A constant
ratio is unremarkable; a widening one is the result.

Usage
    python analyze.py
    python analyze.py --horizon 10 --out REPORT.md
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"quantum_transformer": "#1f3864", "classical_transformer": "#bf8f00"}
SHORT = {"quantum_transformer": "quantum", "classical_transformer": "classical"}


def load(run_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "result.json"))):
        with open(path) as f:
            out.append(json.load(f))
    return out


def paired_p(a, b, n_perm=20000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1., 1.], size=(n_perm, len(d))) * np.abs(d)).mean(1)
    return float((np.abs(null) >= abs(d.mean()) - 1e-15).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--observables", default=None, help="filter: z or zx")
    ap.add_argument("--d_model", type=int, default=None,
                    help="filter to one model width; defaults to the most "
                         "common one, since a capacity sweep adds others")
    ap.add_argument("--out", default=None,
                    help="default names the file after the settings analysed, "
                         "so the two observable arms cannot overwrite one "
                         "another's report")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    runs = load(os.path.join(here, args.run_dir))
    if not runs:
        raise SystemExit("[analyze] no runs found — train something first.")

    # The runs directory accumulates every experiment: the scaling sweep, the
    # observables control arm, the capacity sweep. Those differ in more than
    # system size, so mixing them would compare a quantum model under one
    # setting against a classical model under another — the exact confound
    # this phase exists to avoid. Each analysis is therefore pinned to ONE
    # setting per axis, chosen explicitly or defaulted to the most common.
    def pin(field, requested, label):
        vals = sorted({r[field] for r in runs}, key=str)
        if requested is not None:
            keep = requested
        elif len(vals) == 1:
            keep = vals[0]
        else:
            counts = {v: sum(1 for r in runs if r[field] == v) for v in vals}
            keep = max(counts, key=lambda v: counts[v])
        if len(vals) > 1:
            other = [v for v in vals if v != keep]
            print(f"[analyze] runs exist at {label} {vals}; using {keep!r}, "
                  f"ignoring {other}. Pass --{field} to choose another.")
        return keep

    obs_keep = pin("observables", args.observables, "observable sets")
    d_keep = pin("d_model", args.d_model, "model widths")
    runs = [r for r in runs
            if r["observables"] == obs_keep and r["d_model"] == d_keep]
    if not runs:
        raise SystemExit(
            f"[analyze] no runs with observables={obs_keep!r}, "
            f"d_model={d_keep}.")
    figdir = os.path.join(here, "figures")
    os.makedirs(figdir, exist_ok=True)
    k = str(args.horizon)
    stem = f"{obs_keep}_d{d_keep}_k{args.horizon}"
    out_name = args.out or f"REPORT_{stem}.md"

    R = ["# Phase 3 Stage A — Quantum-Native Dynamics", "",
         f"{len(runs)} runs, all at observables = `{obs_keep}`, "
         f"d_model = {d_keep}. Reported at horizon k = {args.horizon}.", "",
         "Both architectures have a FIXED size while the quantum system they "
         "model grows. The question is which degrades more slowly.", ""]

    # model -> n_spins -> seed -> error
    by = defaultdict(lambda: defaultdict(dict))
    meta = {}
    for r in runs:
        if r["seed"] in by[r["model"]][r["n_spins"]]:
            raise SystemExit(
                f"[analyze] two runs share model={r['model']}, "
                f"n_spins={r['n_spins']}, seed={r['seed']} at the same "
                f"settings. One would silently overwrite the other. "
                f"Offending directory: {r.get('run_dir')}")
        by[r["model"]][r["n_spins"]][r["seed"]] = r["test_rollout"][k]
        meta[(r["model"], r["n_spins"])] = r

    sizes = sorted({n for m in by for n in by[m]})

    # ---------------------------------------------------- per-size comparison
    R += ["## Error by system size", "",
          "| spins | quantum | classical | ratio q/c | seeds | quantum ahead | p |",
          "|---|---|---|---|---|---|---|"]
    ratios, ratio_sizes = [], []
    for n in sizes:
        q, c = by.get("quantum_transformer", {}).get(n, {}), \
               by.get("classical_transformer", {}).get(n, {})
        shared = sorted(set(q) & set(c))
        if not shared:
            continue
        qv = [q[s] for s in shared]
        cv = [c[s] for s in shared]
        ratio = float(np.mean(qv) / max(np.mean(cv), 1e-12))
        wins = sum(1 for a, b in zip(qv, cv) if a < b)
        pv = paired_p(qv, cv)
        ratios.append(ratio); ratio_sizes.append(n)
        R.append(f"| {n} | {np.mean(qv):.5f} | {np.mean(cv):.5f} | "
                 f"**{ratio:.3f}** | {len(shared)} | {wins}/{len(shared)} | "
                 + (f"{pv:.3f}" if not np.isnan(pv) else "—") + " |")
    R.append("")
    R.append("A ratio below 1 means the quantum model is ahead. The trend "
             "across rows is the result, not any single row.")
    R.append("")

    # ---------------------------------------------------------- the verdict
    if len(ratios) >= 3:
        slope = float(np.polyfit(ratio_sizes, ratios, 1)[0])
        first, last = ratios[0], ratios[-1]
        R += ["### Does the gap widen?", ""]
        R.append(f"- Ratio at {ratio_sizes[0]} spins: **{first:.3f}**")
        R.append(f"- Ratio at {ratio_sizes[-1]} spins: **{last:.3f}**")
        R.append(f"- Linear trend in the ratio: **{slope:+.4f} per added spin**")
        R.append("")
        if slope < -0.01:
            R.append("**The quantum model's relative advantage GROWS with "
                     "system size.** This is the outcome the phase was designed "
                     "to detect: a fixed-size quantum model tracks a growing "
                     "quantum system better than a fixed-size classical one. "
                     "Confirm it extends to larger sizes before claiming it.")
        elif slope > 0.01:
            R.append("**The quantum model degrades FASTER than the classical "
                     "one as the system grows.** A clean negative result for "
                     "the central hypothesis of this phase.")
        else:
            R.append("**The ratio is flat.** Neither architecture degrades "
                     "faster. Whatever difference exists at one size is not a "
                     "scaling effect, and the hypothesis that quantum models "
                     "suit quantum systems is not supported by this evidence.")
        R.append("")

    # ------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    for m, sizes_d in sorted(by.items()):
        xs = sorted(sizes_d)
        ys = [np.mean(list(sizes_d[n].values())) for n in xs]
        ax.plot(xs, ys, "o-", color=C.get(m), lw=1.8, ms=5, label=SHORT.get(m, m))
        if all(len(sizes_d[n]) > 1 for n in xs):
            lo = [np.percentile(list(sizes_d[n].values()), 25) for n in xs]
            hi = [np.percentile(list(sizes_d[n].values()), 75) for n in xs]
            ax.fill_between(xs, lo, hi, color=C.get(m), alpha=0.15)
    # the trivial baseline, so absolute quality is visible
    pers = [meta[("classical_transformer", n)]["baselines"]["persistence"]
            for n in sizes if ("classical_transformer", n) in meta]
    if pers and args.horizon == 1:
        ax.plot(sizes[:len(pers)], pers, ":", color="crimson", lw=1.2,
                label="persistence")
    ax.set_yscale("log"); ax.set_xlabel("spins in the chain")
    ax.set_ylabel(f"prediction error (k={args.horizon})")
    ax.set_title("Error against system size"); ax.legend(fontsize=7)

    ax = axes[1]
    if ratios:
        ax.plot(ratio_sizes, ratios, "o-", color="#4c7a34", lw=1.8, ms=5)
        ax.axhline(1.0, color="crimson", ls="--", lw=1)
        ax.set_xlabel("spins in the chain")
        ax.set_ylabel("quantum error / classical error")
        ax.set_title("Relative performance (below 1 = quantum ahead)")
    fig.savefig(os.path.join(figdir, f"scaling_{stem}.png"))
    plt.close(fig)
    R += [f"![scaling](figures/scaling_{stem}.png)", ""]

    # --------------------------------------------------- absolute quality
    R += ["## Is the task well-posed at each size?", "",
          "| spins | persistence | classical model | ×persistence | R² |",
          "|---|---|---|---|---|"]
    for n in sizes:
        m = meta.get(("classical_transformer", n))
        if not m:
            continue
        R.append(f"| {n} | {m['baselines']['persistence']:.5f} | "
                 f"{m['test_rollout']['1']:.5f} | "
                 f"{m['vs_persistence']:.1f}× | {m['r_squared']:.4f} |")
    R += ["",
          "A model scoring close to persistence has not learned the dynamics, "
          "and a comparison at that size says nothing. A model at R² above "
          "0.999 may be at the task's ceiling, where architectures cannot "
          "differ — the saturation problem found in Phase 1.", ""]

    R += ["## Notes", "",
          "- Model size is FIXED across system sizes by design; only the "
          "system grows.",
          "- The train/test split is fixed across seeds, so seeds vary only "
          "model initialisation — a deliberate change from Phase 1, which "
          "varied the partition too and lost statistical power to it.",
          "- Comparisons are paired within seed. A p above 0.10 means "
          "underpowered, not equivalent.",
          "- Observations are exact expectation values with no measurement "
          "noise. Finite-shot behaviour is a separate question.", ""]

    out = os.path.join(here, out_name)
    with open(out, "w") as f:
        f.write("\n".join(R) + "\n")
    print(f"[analyze] {len(runs)} runs -> {out}")
    if ratios:
        print(f"[analyze] ratio {ratios[0]:.3f} at {ratio_sizes[0]} spins "
              f"-> {ratios[-1]:.3f} at {ratio_sizes[-1]} spins")


if __name__ == "__main__":
    main()
