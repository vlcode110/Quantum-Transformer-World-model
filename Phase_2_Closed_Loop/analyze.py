"""Aggregate Phase 2 agent runs: learning curves, sample efficiency, gate verdict.

Reinforcement learning returns are bounded, skewed and prone to temporary
collapse, so curves are summarised by median and interquartile range rather
than mean and standard deviation. Sample efficiency — environment steps to
first cross the gate — is reported alongside final return, because two agents
can reach the same score at very different cost.

Usage:
    python analyze.py
    python analyze.py --out REPORT_2026-09-01.md
"""
import argparse
import csv
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
COLOR = {"quantum": "#1f3864", "classical": "#bf8f00"}


def load_runs(run_dir):
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(run_dir, "*", "config.json"))):
        d = os.path.dirname(cfg_path)
        with open(cfg_path) as f:
            cfg = json.load(f)
        curve = []
        log = os.path.join(d, "log.csv")
        if os.path.exists(log):
            with open(log) as f:
                for row in csv.DictReader(f):
                    try:
                        curve.append((int(row["step"]), float(row["eval_return"])))
                    except (ValueError, KeyError):
                        continue
        result = None
        rp = os.path.join(d, "result.json")
        if os.path.exists(rp):
            with open(rp) as f:
                result = json.load(f)
        runs.append({"dir": d, "cfg": cfg, "curve": curve, "result": result})
    return runs


def group_key(cfg):
    """Runs are comparable only within identical settings apart from the
    quantum swap and the seed."""
    return (cfg["task"], bool(cfg["partial_obs"]), cfg["batch_size"],
            cfg["seq_len"], cfg["horizon"], cfg["train_every"])


def steps_to_gate(curve, gate):
    for s, r in curve:
        if r >= gate:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--out", default="REPORT.md")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    runs = load_runs(os.path.join(here, args.run_dir))
    if not runs:
        raise SystemExit("[analyze] no runs found — train something first.")
    figdir = os.path.join(here, "figures")
    os.makedirs(figdir, exist_ok=True)

    R = ["# Phase 2 — Quantum World Model Driving an Agent", "",
         f"{len(runs)} run(s).", "",
         "The quantum agent replaces DreamerV3's RSSM prior — the network "
         "predicting the next latent from the recurrent state — with a "
         "parameterized circuit. Everything else is unchanged.", ""]

    groups = defaultdict(list)
    for r in runs:
        groups[group_key(r["cfg"])].append(r)

    for gi, (key, grp) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
        task, po, bs, sl, hz, te = key
        title = f"{task}, {'partial' if po else 'full'} observation"
        R += [f"## {title}", "",
              f"Budget: batch {bs}x{sl}, horizon {hz}, gradient step every "
              f"{te} env steps. Comparisons are valid only within this block.", ""]

        gate = grp[0]["result"]["gate"] if grp[0]["result"] else None
        by_arm = defaultdict(list)
        for r in grp:
            by_arm["quantum" if r["cfg"]["quantum"] else "classical"].append(r)

        R += ["| arm | seeds | final return (median [IQR]) | steps to gate (median) | "
              "quantum params |", "|---|---|---|---|---|"]
        for arm, rs in sorted(by_arm.items()):
            finals = [r["result"]["final_eval_return"] for r in rs if r["result"]]
            hits = [steps_to_gate(r["curve"], gate) for r in rs] if gate else []
            hits = [h for h in hits if h]
            qp = next((r["result"]["quantum_params"] for r in rs if r["result"]), 0)
            if finals:
                med = np.median(finals)
                q1, q3 = np.percentile(finals, 25), np.percentile(finals, 75)
                fstr = f"{med:.1f} [{q1:.1f}, {q3:.1f}]" if len(finals) > 1 else f"{med:.1f}"
            else:
                fstr = "—"
            hstr = f"{int(np.median(hits)):,}" if hits else "not reached"
            R.append(f"| {arm} | {len(rs)} | {fstr} | {hstr} | {qp} |")
        R.append("")

        # sample-efficiency ratio, the number the Phase 2 gate is written against
        qh = [steps_to_gate(r["curve"], gate) for r in by_arm.get("quantum", [])]
        ch = [steps_to_gate(r["curve"], gate) for r in by_arm.get("classical", [])]
        qh, ch = [h for h in qh if h], [h for h in ch if h]
        if qh and ch:
            ratio = np.median(qh) / np.median(ch)
            verdict = "PASS" if ratio <= 3.0 else "FAIL"
            R += [f"**Sample efficiency**: quantum needed {ratio:.2f}x the "
                  f"environment steps of the control to first reach the gate "
                  f"(plan allows up to 3x) → **{verdict}**.", ""]
        elif ch and not qh:
            R += ["**Sample efficiency**: the control reached the gate and the "
                  "quantum agent did not within the step budget.", ""]

        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        for arm, rs in sorted(by_arm.items()):
            curves = [r["curve"] for r in rs if r["curve"]]
            if not curves:
                continue
            grid = sorted({s for c in curves for s, _ in c})
            mat = np.array([np.interp(grid, [s for s, _ in c], [v for _, v in c],
                                      left=np.nan, right=np.nan) for c in curves])
            med = np.nanmedian(mat, axis=0)
            ax.plot(grid, med, color=COLOR[arm], lw=1.8, label=f"{arm} (n={len(curves)})")
            if len(curves) > 1:
                lo = np.nanpercentile(mat, 25, axis=0)
                hi = np.nanpercentile(mat, 75, axis=0)
                ax.fill_between(grid, lo, hi, color=COLOR[arm], alpha=0.15)
        if gate is not None:
            ax.axhline(gate, color="crimson", ls="--", lw=1, label="gate")
        ax.set_xlabel("environment steps"); ax.set_ylabel("evaluation return")
        ax.set_title(title, fontsize=10); ax.legend(fontsize=7)
        fname = f"curve_{task}_{'po' if po else 'full'}.png"
        fig.savefig(os.path.join(figdir, fname)); plt.close(fig)
        R += [f"![{title}](figures/{fname})", ""]

    # gradient health of the substituted component
    series = []
    for r in runs:
        if not r["cfg"]["quantum"]:
            continue
        gp = os.path.join(r["dir"], "grad_log.json")
        if not os.path.exists(gp):
            continue
        with open(gp) as f:
            g = json.load(f)
        if g and "prior_q_var" in g[0]:
            series.append((os.path.basename(r["dir"]), g))
    if series:
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        for name, g in series[:8]:
            ax.plot([e["step"] for e in g],
                    [max(e["prior_q_var"], 1e-30) for e in g], lw=1, label=name[:30])
        ax.set_yscale("log"); ax.set_xlabel("environment steps")
        ax.set_ylabel("Var[grad] on circuit parameters")
        ax.set_title("Gradient health of the quantum prior during agent training")
        ax.legend(fontsize=6)
        fig.savefig(os.path.join(figdir, "prior_grad_health.png")); plt.close(fig)
        R += ["## Gradient health", "",
              "Circuit-parameter gradient variance across agent training. A "
              "curve decaying toward zero while returns keep improving means "
              "the circuit has stopped contributing and the classical wrapper "
              "is carrying the model — the failure mode that makes a nominally "
              "quantum result actually classical.", "",
              "![gradients](figures/prior_grad_health.png)", ""]

    R += ["## Notes", "",
          "- Curves are median with interquartile band. RL returns are skewed "
          "and collapse-prone; mean plus standard deviation implies a symmetry "
          "that is not there.",
          "- The compute budget is reduced from DreamerV3's defaults, so "
          "published DreamerV3 sample efficiencies are NOT a valid comparison. "
          "Only the control run under identical settings is.",
          "- Runs are grouped by task, observability and budget; runs from "
          "different groups must not be pooled.", ""]

    out = os.path.join(here, args.out)
    with open(out, "w") as f:
        f.write("\n".join(R) + "\n")
    print(f"[analyze] {len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
