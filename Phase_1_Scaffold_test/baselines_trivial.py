"""Trivial baselines that give every error number a scale.

An MSE of 0.0022 is meaningless in isolation. These reference points make it
interpretable, and they are the first thing an examiner will ask for:

  persistence   predict z_{t+1} = z_t  ("nothing changed")
  linear        predict z_{t+1} = z_t + (z_t - z_{t-1})  (constant velocity)
  mean          predict the training-set mean  (ignores the input entirely)
  variance      the variance of the target itself — the denominator for R^2

Report your models as a ratio against persistence, and as R^2 against the
variance. "23x better than persistence" travels; "MSE 0.0022" does not.

Usage:
    python baselines_trivial.py
    python baselines_trivial.py --seed 3
"""
import os

import numpy as np

from config import Config
from dataset import build


def evaluate_trivial(split, horizons):
    """Errors of the trivial predictors on the same windows the models saw."""
    z = np.stack([split[i][0].numpy() for i in range(len(split))])   # (N,K,d)
    z_next = np.stack([split[i][2].numpy() for i in range(len(split))])
    z_fut = np.stack([split[i][3].numpy() for i in range(len(split))])  # (N,H,d)

    last = z[:, -1]                     # z_t
    prev = z[:, -2]                     # z_{t-1}
    train_mean = z.reshape(-1, z.shape[-1]).mean(0)

    out = {}
    # --- 1-step
    out["persistence_1"] = float(((last - z_next) ** 2).mean())
    out["linear_1"] = float((((2 * last - prev) - z_next) ** 2).mean())
    out["mean_1"] = float(((train_mean - z_next) ** 2).mean())
    # Per-DIMENSION variance. A flat z_next.var() uses one global mean across
    # all dimensions, so it also counts the spread BETWEEN dimension means —
    # inflating the denominator and therefore R^2. Harmless when the dimensions
    # share a mean (CartPole), badly wrong when they do not (Pendulum).
    out["target_variance"] = float(z_next.var(axis=0).mean())

    # --- k-step: persistence holds z_t fixed for the whole rollout, which is
    # the honest trivial comparator for open-loop evaluation
    for k in horizons:
        if k == 1:
            continue
        tgt = z_fut[:, :k]
        out[f"persistence_{k}"] = float(((last[:, None, :] - tgt) ** 2).mean())
    return out


def main():
    cfg = Config.from_cli()
    splits, info = build(cfg, verbose=False)
    res = evaluate_trivial(splits["test"], cfg.rollout_horizons)

    print(f"dataset {info['trajectory_version']}  latent_dim={info['z_dim']}  "
          f"seed={cfg.seed}\n")
    print(f"{'baseline':>18s} {'MSE':>12s}")
    for k, v in res.items():
        print(f"{k:>18s} {v:>12.5f}")

    p1 = res["persistence_1"]
    var = res["target_variance"]
    print("\n--- how to read your model numbers ---")
    print(f"target variance      {var:.5f}   (MSE of predicting the mean)")
    print(f"persistence (1-step) {p1:.5f}   (MSE of 'nothing changed')")
    print("\nFor a model with 1-step MSE m:")
    print(f"  improvement over persistence = {p1:.5f} / m")
    print(f"  R^2 = 1 - m / {var:.5f}")
    # Read this task's own results rather than printing hardcoded examples,
    # which are trivially mistaken for measurements of the current task.
    import glob
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    seen = {}
    for path in glob.glob(os.path.join(here, "runs", "*", "result.json")):
        with open(path) as f:
            r = json.load(f)
        if r.get("task") != cfg.task:
            continue
        key = (r["model"], r.get("d_model", "?"))
        seen.setdefault(key, []).append(r["test_rollout"]["1"])
    if seen:
        print()
        for (m_name, cap), vals in sorted(seen.items()):
            m = float(np.mean(vals))
            print(f"  {m_name:>22s} d={cap:<4} m={m:.5f}  ->  "
                  f"{p1/m:6.1f}x persistence, R^2={1 - m/var:6.4f}  "
                  f"(n={len(vals)})")
    else:
        print(f"\n  (no {cfg.task} runs found yet — train something, then rerun)")


if __name__ == "__main__":
    main()
