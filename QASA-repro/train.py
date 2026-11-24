"""Train QASA vs classical transformer on synthetic forecasting tasks.

Usage:
    python train.py --signal mackey_glass            # both models, default seeds
    python train.py --signal sine --epochs 30
    python train.py --all                            # full comparison table
Smoke test (~1 min):
    python train.py --signal sine --epochs 2 --limit 400
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import SIGNALS, windows
from qasa import Forecaster


def run(signal, mode, seed, epochs, lookback, limit, device, verbose=True,
        entangler="ring_cascade"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    series = SIGNALS[signal]()
    if limit:
        series = series[:limit]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = windows(series, lookback)
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    Xtr, ytr, Xva, yva, Xte, yte = map(to, (Xtr, ytr, Xva, yva, Xte, yte))

    model = Forecaster(mode=mode, entangler=entangler).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    batch = 64
    best_va, best_state = float("inf"), None

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = loss_fn(model(Xva), yva).item()
        if va < best_va:
            best_va = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  [{mode}] epoch {ep+1:>3d}/{epochs} val_mse {va:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte)
        mse = loss_fn(pred, yte).item()
        mae = (pred - yte).abs().mean().item()
    return {"signal": signal, "mode": mode, "seed": seed, "test_mse": mse,
            "test_mae": mae, "params": n_params,
            "quantum_params": model.n_quantum_params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", choices=list(SIGNALS), default="mackey_glass")
    ap.add_argument("--all", action="store_true", help="run all signals")
    ap.add_argument("--modes", nargs="+", default=["quantum", "classical"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="truncate series (smoke)")
    ap.add_argument("--entangler", default="ring_cascade",
                    choices=["ring_cascade", "brickwork"],
                    help="ring_cascade reproduces the paper; brickwork runs "
                         "the entangler-sensitivity check")
    args = ap.parse_args()
    device = "cpu"  # PennyLane default.qubit backprop is CPU-bound anyway

    signals = list(SIGNALS) if args.all else [args.signal]
    results = []
    t0 = time.time()
    for sig in signals:
        for mode in args.modes:
            for seed in args.seeds:
                print(f"== {sig} | {mode} | seed {seed} ==")
                r = run(sig, mode, seed, args.epochs, args.lookback,
                        args.limit, device, entangler=args.entangler)
                r["entangler"] = args.entangler if mode == "quantum" else "n/a"
                print(f"   test_mse {r['test_mse']:.5f}  test_mae {r['test_mae']:.5f}"
                      f"  ({r['params']} params, {r['quantum_params']} quantum)")
                results.append(r)

    os.makedirs("results", exist_ok=True)
    # Separate file per entangler: appending a new column to the existing
    # comparison.csv would silently misalign rows against the old header, and
    # the sensitivity check should not contaminate the reproduction data.
    fname = ("comparison.csv" if args.entangler == "ring_cascade"
             else f"comparison_{args.entangler}.csv")
    path = os.path.join("results", fname)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        if new:
            w.writeheader()
        w.writerows(results)

    print(f"\n=== summary (mean test MSE over seeds) — {time.time()-t0:.0f}s ===")
    for sig in signals:
        line = f"{sig:>14s}: "
        for mode in args.modes:
            vals = [r["test_mse"] for r in results
                    if r["signal"] == sig and r["mode"] == mode]
            line += f"{mode} {np.mean(vals):.5f} ± {np.std(vals):.5f}   "
        print(line)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
