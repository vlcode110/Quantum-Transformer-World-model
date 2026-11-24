"""Generate Paper B's figures from the completed Phase 2 runs.

    python make_figures.py

Writes PDFs into Paper_B/figures/. Reads only; no training, no
re-evaluation. Takes seconds.

Three figures:

  fig_learning.pdf   Evaluation return against environment step, both
                     arms, both observability conditions. Median with
                     interquartile band across seeds -- not mean with
                     standard error, because the per-seed distribution is
                     strongly skewed (one seed at 131 against another at
                     479) and a mean curve would imply a symmetry the
                     data does not have.

  fig_gradient.pdf   Variance of the gradient on the 28 circuit
                     parameters, against training step. This is the
                     barren-plateau diagnostic extended through training
                     rather than measured once at initialisation. A curve
                     decaying steadily toward zero would mean the circuit
                     had stopped learning while the surrounding classical
                     layers kept improving -- the failure mode that makes
                     a nominally quantum result actually classical.

  fig_kl.pdf         Prior-posterior KL for both arms. This is the term
                     the world-model loss optimises, so it reports
                     whether the two priors are being regularised
                     comparably; a systematic difference would confound
                     the return comparison.

Every figure plots individual seeds faintly behind the summary, so a
result driven by one seed is visible rather than hidden by aggregation.
"""
import csv
import glob
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "Phase_2_Closed_Loop", "runs")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42,
})
QC = "#1F3864"      # quantum
CC = "#BF8F00"      # classical
STEP_BUDGET = 80000


def retry(fn, path, attempts=12):
    """~/Documents is iCloud-backed and intermittently refuses opens."""
    last = None
    for i in range(attempts):
        try:
            return fn(path)
        except FileNotFoundError:
            return None
        except Exception as e:
            last = e
            time.sleep(0.25 + 0.15 * i)
    print(f"  [io] gave up on {os.path.basename(path)}: {type(last).__name__}")
    return None


def read_json(path):
    return retry(lambda p: json.load(open(p)), path)


def read_log(path):
    def _go(p):
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        out = {}
        for col in ("step", "eval_return", "wm_kl", "prior_q_var"):
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[col]))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            out[col] = np.array(vals)
        return out
    return retry(_go, path)


def collect():
    """Group completed transformer-K4 runs by (condition, arm)."""
    cells = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*archtransformer_K4*", ""))):
        r = read_json(os.path.join(d, "result.json"))
        if not r or r.get("env_steps") != STEP_BUDGET:
            continue
        log = read_log(os.path.join(d, "log.csv"))
        if log is None:
            continue
        key = ("partial" if r["partial_obs"] else "full",
               "quantum" if r["quantum"] else "classical")
        cells.setdefault(key, {})[r["seed"]] = log
    return cells


def band(ax, seeds, col, colour, label, logy=False):
    """Median + IQR across seeds on a common step grid, seeds faint behind."""
    if not seeds:
        return 0
    grid = None
    for log in seeds.values():
        g = log["step"][np.isfinite(log[col])]
        grid = g if grid is None else (g if len(g) < len(grid) else grid)
    if grid is None or len(grid) == 0:
        return 0
    stack = []
    for log in seeds.values():
        m = np.isfinite(log[col])
        if m.sum() < 2:
            continue
        stack.append(np.interp(grid, log["step"][m], log[col][m]))
        ax.plot(grid, stack[-1], color=colour, lw=0.5, alpha=0.22)
    if not stack:
        return 0
    S = np.vstack(stack)
    med = np.median(S, 0)
    ax.plot(grid, med, color=colour, lw=1.8,
            label=f"{label} (n={len(stack)})")
    ax.fill_between(grid, np.percentile(S, 25, 0), np.percentile(S, 75, 0),
                    color=colour, alpha=0.16, lw=0)
    if logy:
        ax.set_yscale("log")
    return len(stack)


def fig_learning(cells):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, cond, title in ((axes[0], "partial",
                             "Partial observation (velocities hidden)"),
                            (axes[1], "full", "Full observation (Markovian)")):
        n = 0
        n += band(ax, cells.get((cond, "quantum"), {}), "eval_return", QC,
                  "quantum transformer")
        n += band(ax, cells.get((cond, "classical"), {}), "eval_return", CC,
                  "classical twin")
        ax.axhline(475, color="0.4", ls=":", lw=0.9)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("environment steps")
        if n:
            ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("evaluation return")
    axes[0].text(2000, 492, "solved threshold", fontsize=6.5, color="0.4")
    fig.savefig(os.path.join(FIGS, "fig_learning.pdf"))
    plt.close(fig)
    print("  fig_learning.pdf")


def fig_gradient(cells):
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    n = 0
    for cond, ls, lbl in (("partial", "-", "partial obs"),
                          ("full", "--", "full obs")):
        seeds = cells.get((cond, "quantum"), {})
        got = 0
        for log in seeds.values():
            m = np.isfinite(log["prior_q_var"]) & (log["prior_q_var"] > 0)
            if m.sum() < 2:
                continue
            ax.plot(log["step"][m], log["prior_q_var"][m], ls, color=QC,
                    lw=0.8, alpha=0.5 if got else 0.9,
                    label=f"{lbl} (n=?)" if not got else None)
            got += 1
        n += got
        if got:
            ax.lines[-1].set_label(f"{lbl} (n={got})")
    ax.set_yscale("log")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(r"Var$[\nabla]$ on the 28 circuit parameters")
    if n:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no gradient variance logged",
                ha="center", transform=ax.transAxes, fontsize=8, color="0.4")
    fig.savefig(os.path.join(FIGS, "fig_gradient.pdf"))
    plt.close(fig)
    print("  fig_gradient.pdf" + ("" if n else "  (EMPTY - see note below)"))
    return n


def fig_kl(cells):
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    n = 0
    n += band(ax, cells.get(("partial", "quantum"), {}), "wm_kl", QC,
              "quantum transformer", logy=True)
    n += band(ax, cells.get(("partial", "classical"), {}), "wm_kl", CC,
              "classical twin", logy=True)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("prior-posterior KL")
    ax.set_title("Partial observation", fontsize=9)
    if n:
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIGS, "fig_kl.pdf"))
    plt.close(fig)
    print("  fig_kl.pdf" + ("" if n else "  (EMPTY)"))
    return n


def main():
    print(f"reading runs from {os.path.normpath(RUNS)}")
    cells = collect()
    if not cells:
        raise SystemExit("no completed transformer-K4 runs found")
    print("cells found:")
    for k in sorted(cells):
        print(f"  {k[0]:<8} {k[1]:<10} seeds {sorted(cells[k])}")
    print("writing figures:")
    fig_learning(cells)
    ngrad = fig_gradient(cells)
    fig_kl(cells)
    print(f"\n-> {FIGS}")
    if not ngrad:
        print("\nNOTE: no circuit gradient variance was logged. train_agent.py\n"
              "records it every cfg.train_every * cfg.grad_var_every steps and\n"
              "only when prior_grad_stats() returns a value, so if the column\n"
              "is empty the instrumentation did not fire during these runs.\n"
              "The figure is a placeholder and the paper should not cite it\n"
              "until this is resolved.")


if __name__ == "__main__":
    main()
