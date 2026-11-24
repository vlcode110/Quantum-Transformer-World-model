"""Aggregate Phase 1 runs: seed-level statistics, gate verdict, plots, tables.

Reuses the Phase 0 conventions deliberately — paired tests across matched
seeds, bootstrap CIs, and an explicit "underpowered" verdict rather than a
silent tie.

Usage:
    python analyze.py                       # runs/ + results/ablation.jsonl
    python analyze.py --out REPORT.md
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
C = {"quantum_transformer": "#1f3864", "classical_transformer": "#bf8f00",
     "qlstm": "#4c7a34", "pqc_mlp": "#8b3a62"}


def bootstrap_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def paired_p(a, b, n_perm=20000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1., 1.], size=(n_perm, len(d))) * np.abs(d)).mean(1)
    return float((np.abs(null) >= abs(d.mean()) - 1e-15).mean())


def load_runs(run_dir, ablation_path):
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "*", "result.json"))):
        with open(p) as f:
            out.append(json.load(f))
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("failed"):
                    out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--ablation", default="results/ablation.jsonl")
    ap.add_argument("--out", default="REPORT.md")
    ap.add_argument("--gate_k", type=int, default=10)
    ap.add_argument("--gate_ratio", type=float, default=2.0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    runs = load_runs(os.path.join(here, args.run_dir),
                     os.path.join(here, args.ablation))
    if not runs:
        raise SystemExit("[analyze] no runs found — train something first.")
    figdir = os.path.join(here, "figures"); os.makedirs(figdir, exist_ok=True)
    R = ["# Phase 1 — Offline Quantum Dynamics Model", "",
         f"{len(runs)} run(s) aggregated.", ""]

    # ------------------------------------------------ main comparison by model
    # Group by (model, capacity). Pooling runs of different capacity into one
    # "model comparison" mixes populations and can invert the verdict — models
    # at the error floor look identical while capacity-limited ones differ.
    main_runs = [r for r in runs if "ablation_axis" not in r]
    caps = sorted({r.get("d_model", "?") for r in main_runs})
    ctxs = sorted({r.get("context", "?") for r in main_runs})
    if len(caps) > 1:
        R += [f"> **Note:** runs span {len(caps)} capacities ({caps}); they are "
              f"reported separately and must not be pooled.", ""]
    if len(ctxs) > 1:
        R += [f"> **Note:** runs span {len(ctxs)} attention context lengths "
              f"({ctxs}); these are a separate axis from capacity and are also "
              f"reported separately. See the context section below.", ""]

    # Group by (model, capacity, CONTEXT).
    #
    # `context` was absent from this key in an earlier version. Because the
    # inner mapping is keyed by seed, every context length at a given width
    # and seed wrote into the same slot and the last one read from disk won.
    # Directory names sort ctx16 < ctx2 < ctx4 < ctx8, so ctx8 silently won
    # every comparison while 24 of 32 runs at each width were discarded. The
    # tell was a gate line reporting "n=8 seeds" for a width holding 32 runs.
    #
    # This is the third instance of one failure mode in this project: a
    # quantity that varied was not recorded as having varied. It is also the
    # cheapest to prevent, which is why the assertion below is unconditional.
    # Fields that make two runs comparable. `task` was absent from an earlier
    # version of this key, with the result that CartPole and Pendulum runs
    # were pooled: "pendulum" sorts after "cartpole", so a Pendulum run
    # silently replaced its CartPole counterpart at every shared
    # (capacity, context, seed). Two environments were being averaged into
    # one gate verdict. `dataset_version` is included for the same reason --
    # this script's own notes say trajectory versions must not be pooled.
    KEY_FIELDS = ("task", "latent_mode", "latent_dim", "n_qubits", "q_layers",
                  "encoding", "readout", "entangler", "d_model", "n_layers",
                  "context", "dataset_version")

    def rkey(r):
        return tuple(r.get(f, "?") for f in KEY_FIELDS)

    tasks = sorted({r.get("task", "?") for r in main_runs})
    if len(tasks) > 1:
        R += [f"> **Note:** runs span {len(tasks)} tasks ({tasks}). These are "
              f"different environments and are never pooled.", ""]

    by_model = defaultdict(lambda: defaultdict(dict))  # (model,key)->k->seed
    seen = {}
    for r in main_runs:
        key = (r["model"], r.get("d_model", "?"), r.get("context", "?"),
               rkey(r))
        for k, v in r["test_rollout"].items():
            slot = (key, int(k), r["seed"])
            if slot in seen and seen[slot] != v:
                raise SystemExit(
                    "[analyze] REFUSING TO AGGREGATE: two runs collide on\n"
                    f"  model/capacity/context/horizon/seed = {slot}\n"
                    f"  values {seen[slot]} and {v}\n"
                    "  Some setting that differs between them is missing from "
                    "the grouping key. Add it rather than letting one win.")
            seen[slot] = v
            by_model[key][int(k)][r["seed"]] = v

    if by_model:
        R += ["## 1. Model comparison", "",
              "| task | capacity | context | model | params | quantum params "
              f"| 1-step [95% CI] | {args.gate_k}-step [95% CI] | seeds |",
              "|---|---|---|---|---|---|---|---|---|"]
        for (m, cap, ctx, kk), ks in sorted(
                by_model.items(),
                key=lambda kv: (str(kv[0][3][0]), kv[0][1], kv[0][2], kv[0][0])):
            ex = next(r for r in main_runs
                      if r["model"] == m and rkey(r) == kk)
            one = list(ks.get(1, {}).values())
            gk = list(ks.get(args.gate_k, {}).values())
            m1, l1, h1 = bootstrap_ci(one)
            mk, lk, hk = bootstrap_ci(gk)
            f = lambda a, b, c: (f"{a:.5f}" if np.isnan(b)
                                 else f"{a:.5f} [{b:.5f}, {c:.5f}]")
            R.append(f"| {ex.get('task','?')} | d={cap} | K={ctx} | `{m}` | "
                     f"{ex['params']} | {ex['quantum_params']} | "
                     f"{f(m1,l1,h1)} | {f(mk,lk,hk)} | {len(one)} |")

        # gate verdict, paired across shared seeds, SEPARATELY PER CAPACITY
        R += ["", "### Phase 1 gate (per task, capacity and context)", ""]
        qkeys = {(cap, ctx, kk) for (m, cap, ctx, kk) in by_model
                 if m == "quantum_transformer"}
        ckeys = {(cap, ctx, kk) for (m, cap, ctx, kk) in by_model
                 if m == "classical_transformer"}
        for (cap, ctx, kk) in sorted(qkeys & ckeys,
                                     key=lambda t: (str(t[2][0]), t[0], t[1])):
            q = by_model[("quantum_transformer", cap, ctx, kk)]
            c = by_model[("classical_transformer", cap, ctx, kk)]
            R.append(f"**{kk[0]}, d_model = {cap}, context K = {ctx}**")
            for k, need in ((1, "<= classical"),
                            (args.gate_k, f"<= {args.gate_ratio}x classical")):
                shared = sorted(set(q.get(k, {})) & set(c.get(k, {})))
                if not shared:
                    continue
                qv = [q[k][s] for s in shared]; cv = [c[k][s] for s in shared]
                ratio = np.mean(qv) / max(np.mean(cv), 1e-12)
                p = paired_p(qv, cv)
                wins = sum(1 for a, b in zip(qv, cv) if a < b)
                ok = ratio <= (1.0 if k == 1 else args.gate_ratio)
                verdict = "PASS" if ok else "FAIL"
                if not np.isnan(p) and p > 0.10:
                    verdict += " (p={:.2f} — underpowered, add seeds)".format(p)
                elif not np.isnan(p):
                    verdict += " (p={:.3f})".format(p)
                R.append(f"- **{k}-step**: quantum/classical = {ratio:.3f}x "
                         f"(need {need}) → **{verdict}**  "
                         f"_(n={len(shared)} seeds, quantum ahead in "
                         f"{wins}/{len(shared)})_")
            R.append("")

        # rollout-error growth curve (one line per model x capacity)
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for (m, cap, ctx, kk), ks in sorted(by_model.items(),
                                            key=lambda kv: str(kv[0])):
            hs = sorted(ks)
            mean = [np.mean(list(ks[k].values())) for k in hs]
            ax.plot(hs, mean, "o-", label=f"{m} {kk[0]} d={cap} K={ctx}",
                    color=C.get(m), lw=1.5, ms=4, alpha=0.85)
            if all(len(ks[k]) > 1 for k in hs):
                lo = [np.percentile(list(ks[k].values()), 25) for k in hs]
                hi = [np.percentile(list(ks[k].values()), 75) for k in hs]
                ax.fill_between(hs, lo, hi, color=C.get(m), alpha=0.10)
        ax.set_yscale("log"); ax.set_xlabel("open-loop horizon k")
        ax.set_ylabel("rollout MSE"); ax.set_title("Error growth in imagination")
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(figdir, "rollout_error.png")); plt.close(fig)
        R += ["![rollout](figures/rollout_error.png)", "",
              "Error growth with horizon is the number that matters for a world "
              "model: a good 1-step model that diverges by k=10 is useless for "
              "imagination-based RL in Phase 2.", ""]

    # ------------------------------------------- attention context ablation
    # The architecture's claim is that attending over recent recurrent states
    # helps in proportion to how much history the dynamics require. That is a
    # statement about the TREND in K, so it is only readable at fixed width:
    # an earlier sweep varied K and width together and admitted no trend at
    # all. Each row below is a paired quantum-vs-classical comparison at one
    # context length, over the seeds the two arms share.
    # One block per (task, capacity) that has more than one context measured
    # in both arms. Varying K is only interpretable with everything else held.
    def _ctxbase(kk):
        i = KEY_FIELDS.index("context")
        return kk[:i] + ("*",) + kk[i + 1:]
    groups = defaultdict(set)
    for (m, cap, ctx, kk) in by_model:
        if m in ("quantum_transformer", "classical_transformer"):
            groups[(cap, _ctxbase(kk))].add((m, ctx, kk))
    ctx_blocks = []
    for (cap, base), members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        both = sorted({ctx for (m, ctx, kk) in members if m == "quantum_transformer"}
                      & {ctx for (m, ctx, kk) in members
                         if m == "classical_transformer"})
        if len(both) > 1:
            ctx_blocks.append((cap, base, both,
                               {ctx: kk for (m, ctx, kk) in members}))
    for cap, base, block_ctxs, kkof in ctx_blocks:
        R += [f"## Attention context length ({base[0]}, d_model = {cap})", "",
              "| context K | 1-step Q | 1-step C | ratio | p | Q ahead | "
              f"{args.gate_k}-step Q | {args.gate_k}-step C | ratio | p | "
              "Q ahead | seeds |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        trend = []
        for ctx in block_ctxs:
            kk = kkof[ctx]
            q = by_model.get(("quantum_transformer", cap, ctx, kk))
            c = by_model.get(("classical_transformer", cap, ctx, kk))
            if not (q and c):
                continue
            row, nseed = [f"| {ctx} "], 0
            for k in (1, args.gate_k):
                shared = sorted(set(q.get(k, {})) & set(c.get(k, {})))
                if not shared:
                    row.append("| — | — | — | — | — ")
                    continue
                qv = np.array([q[k][s] for s in shared])
                cv = np.array([c[k][s] for s in shared])
                ratio = qv.mean() / max(cv.mean(), 1e-12)
                pv = paired_p(qv, cv)
                wins = int((qv < cv).sum())
                nseed = len(shared)
                row.append(f"| {qv.mean():.5f} | {cv.mean():.5f} | "
                           f"{ratio:.3f}x | {pv:.3f} | {wins}/{len(shared)} ")
                if k == args.gate_k:
                    trend.append((ctx, ratio, qv.mean(), cv.mean()))
            row.append(f"| {nseed} |")
            R.append("".join(row))
        R.append("")
        if len(trend) > 1:
            lo, hi = trend[0], trend[-1]
            R += [f"Absolute {args.gate_k}-step error for the quantum arm moves "
                  f"from {lo[2]:.5f} at K={lo[0]} to {hi[2]:.5f} at K={hi[0]} "
                  f"({'improving' if hi[2] < lo[2] else 'worsening'} by "
                  f"{abs(1 - hi[2]/max(lo[2],1e-12)) * 100:.0f}%); the classical "
                  f"arm moves from {lo[3]:.5f} to {hi[3]:.5f}. A trend present in "
                  "both arms is a property of the task, not of the circuit; only "
                  "a divergence between them supports the architectural claim.",
                  ""]
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        for m in ("quantum_transformer", "classical_transformer"):
            xs, ys = [], []
            for ctx in block_ctxs:
                d = by_model.get((m, cap, ctx, kkof[ctx]))
                if d and args.gate_k in d:
                    xs.append(ctx); ys.append(np.mean(list(d[args.gate_k].values())))
            if xs:
                ax.plot(xs, ys, "o-", label=m, color=C.get(m), lw=1.6, ms=5)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(block_ctxs); ax.set_xticklabels(block_ctxs)
        ax.set_xlabel("attention context K"); ax.set_ylabel(f"{args.gate_k}-step MSE")
        ax.set_title(f"Open-loop error vs context ({base[0]}, d={cap})")
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(figdir, f"context_{base[0]}_d{cap}.png"))
        plt.close(fig)
        R += [f"![context](figures/context_{base[0]}_d{cap}.png)", ""]

    # --------------------------------------------------- capacity curves
    # THE central figure: error against parameter count, one curve per
    # architecture. Comparing architectures at a single capacity is
    # uninterpretable when the benchmark saturates (Section: capacity
    # saturation); the curve shows whether an architecture is merely smaller
    # or genuinely limited — a model that plateaus ABOVE another's floor
    # cannot be rescued by adding parameters.
    # One curve per (model, context). Pooling contexts here produced rows
    # labelled "35 seeds" that were a mixture of four different context
    # lengths at one parameter count -- a number with no referent.
    cap_pts = defaultdict(lambda: defaultdict(list))   # (model,ctx)->params->errs
    for r in runs:
        if "params" in r and "test_rollout" in r:
            cap_pts[(r["model"], r.get("task", "?"),
                     r.get("context", "?"))][r["params"]].append(
                r["test_rollout"]["1"])
    if any(len(v) > 1 for v in cap_pts.values()):
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        R += ["## 2. Capacity curves", "",
              "| model | task | context | params | 1-step (mean) | seeds |",
              "|---|---|---|---|---|---|"]
        for (m, tk, ctx), pts in sorted(cap_pts.items(), key=lambda kv: str(kv[0])):
            xs = sorted(pts)
            ys = [float(np.mean(pts[x])) for x in xs]
            ax.plot(xs, ys, "o-", label=f"{m} {tk} K={ctx}", color=C.get(m),
                    lw=1.6, ms=5, alpha=0.85)
            if all(len(pts[x]) > 1 for x in xs):
                lo = [np.percentile(pts[x], 25) for x in xs]
                hi = [np.percentile(pts[x], 75) for x in xs]
                ax.fill_between(xs, lo, hi, color=C.get(m), alpha=0.15)
            for x, y in zip(xs, ys):
                R.append(f"| `{m}` | {tk} | K={ctx} | {x} | {y:.5f} | "
                         f"{len(pts[x])} |")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("trainable parameters"); ax.set_ylabel("1-step MSE")
        ax.set_title("Error vs capacity by architecture")
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(figdir, "capacity_curves.png")); plt.close(fig)
        R += ["", "![capacity](figures/capacity_curves.png)", "",
              "A curve that flattens ABOVE another architecture's floor is "
              "representationally limited, not merely undersized — extra "
              "parameters cannot close the gap. A curve shifted left at equal "
              "error is more parameter-efficient.", ""]

    # ------------------------------------------------------------- ablations
    abl = [r for r in runs if "ablation_axis" in r]
    if abl:
        R += ["## 3. Ablations", ""]
        axes_ = sorted({r["ablation_axis"] for r in abl})
        fig, axarr = plt.subplots(1, len(axes_), figsize=(3.2 * len(axes_), 3.2))
        axarr = np.atleast_1d(axarr)
        for ax, axis in zip(axarr, axes_):
            sub = [r for r in abl if r["ablation_axis"] == axis]
            groups = defaultdict(lambda: defaultdict(list))
            for r in sub:
                groups[r["model"]][r["ablation_value"]].append(r["test_rollout"]["1"])
            R += [f"### {axis}", "",
                  "| value | model | 1-step mean | n |", "|---|---|---|---|"]
            for m, vals in sorted(groups.items()):
                xs = sorted(vals)
                ys = [np.mean(vals[v]) for v in xs]
                ax.plot(range(len(xs)), ys, "o-", label=m, color=C.get(m), ms=4)
                ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=7,
                                                                 rotation=30)
                for v in xs:
                    R.append(f"| `{v}` | {m} | {np.mean(vals[v]):.5f} | "
                             f"{len(vals[v])} |")
            ax.set_title(axis, fontsize=9); ax.set_ylabel("1-step MSE")
            if ax is axarr[0]:
                ax.legend(fontsize=7)
            R.append("")
        fig.savefig(os.path.join(figdir, "ablations.png")); plt.close(fig)
        R += ["![ablations](figures/ablations.png)", ""]

    # -------------------------------------------------------- gradient health
    grad_files = sorted(glob.glob(os.path.join(here, args.run_dir, "*",
                                               "grad_log.json")))
    series = []
    for gp in grad_files:
        with open(gp) as f:
            g = json.load(f)
        if g and "q_var" in g[0]:
            series.append((os.path.basename(os.path.dirname(gp)), g))
    if series:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for name, g in series[:8]:
            ax.plot([e["step"] for e in g], [max(e["q_var"], 1e-30) for e in g],
                    lw=1, label=name[:28])
        ax.set_yscale("log"); ax.set_xlabel("training step")
        ax.set_ylabel("Var[grad] on circuit params")
        ax.set_title("Quantum gradient health during training")
        ax.legend(fontsize=6)
        fig.savefig(os.path.join(figdir, "grad_health.png")); plt.close(fig)
        R += ["## 3. Gradient health", "",
              "Circuit-parameter gradient variance over training. A curve that "
              "decays steadily toward zero means the circuit has stopped "
              "learning while the classical wrapper keeps improving — the "
              "failure mode that makes a 'quantum' result actually classical. "
              "Compare the starting magnitude against the Phase 0 baseline at "
              "the same width.", "", "![grad](figures/grad_health.png)", ""]

    R += ["## Notes", "",
          "- Rollout errors are OPEN-LOOP: the model consumes its own "
          "predictions, which is how a world model is used in Phase 2.",
          "- Comparisons are paired across matched seeds; a `p > 0.10` verdict "
          "means underpowered, not equivalent.",
          "- Every run records its `dataset_version`; results from different "
          "trajectory versions are not comparable and should not be pooled.", ""]

    out = os.path.join(here, args.out)
    with open(out, "w") as f:
        f.write("\n".join(R) + "\n")
    print(f"[analyze] {len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
