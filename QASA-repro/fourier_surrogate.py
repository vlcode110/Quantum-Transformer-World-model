"""Is the PQC a distinct function class, or one parameterisation of a classical one?

Every other comparison in this project asks whether the quantum model performs
better. This one asks something sharper: what the quantum model can represent at
all, and whether a classical model built to span the same thing matches it.

The theory is Schuld, Sweke and Meyer (arXiv:2008.08605): a data re-uploading
circuit computes a partial Fourier series in its inputs, where the accessible
frequencies are fixed by the encoding gates and the trainable parameters set only
the coefficients. `FourierTokenLayer` parameterises exactly that, classically —
fixed integer frequencies drawn from the circuit's spectrum, learned
coefficients, identical down- and up-projections.

Sweeping the surrogate's coefficient count gives the number this experiment
exists to produce: the classical parameter budget at which the surrogate matches
the circuit.

  surrogate matches at a comparable budget -> the circuit is a particular
      parameterisation of a classical feature map. No distinct function class,
      and no expressivity route to an advantage at this scale.
  surrogate needs far more -> EITHER the circuit's constrained parameterisation
      carries a useful inductive bias, OR the frequency sample missed what
      matters. These two are not separated by this experiment.

The asymmetry is worth stating plainly, because it decides how much each outcome
is worth. At four input dimensions and six layers the circuit's accessible
spectrum holds on the order of 10^5 frequency vectors, far too many to span, so
the surrogate SAMPLES it: `n_freq` random draws held fixed, in the manner of
random Fourier features. A sample that suffices is strong evidence — nothing
special is being exploited if a handful of arbitrary frequencies do the same
job. A sample that does not suffice is weak evidence, since a better draw might.
The seed varies the draw as well as the initialisation, so seed-to-seed spread
in the surrogate arm reports how much the draw matters.

No quantum simulation is required for the surrogate arm, so it is cheap.

    python fourier_surrogate.py --signal mackey_glass --seeds 0 1 2 3 4
    python fourier_surrogate.py --n_freq 1 2 4 8 16 32 --seeds 0 1 2
    python fourier_surrogate.py --epochs 2 --limit 400 --seeds 0      (smoke)

Writes results/fourier_surrogate.csv, appending, and prints the capacity curve.
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "fourier_surrogate.csv")
# epochs and limit are recorded because a smoke run (`--epochs 2 --limit 400`)
# appends to the same file as a full run, and without them the two are separable
# only by eyeballing the error magnitude.
FIELDS = ["signal", "mode", "n_freq", "seed", "epochs", "limit", "test_mse",
          "test_mae", "params", "core_params", "quantum_params", "wall_clock_s"]


def core_param_count(model):
    """Parameters in the token-mixing block only — the part under comparison.

    The down- and up-projections are identical across arms by construction, so
    counting them would dilute exactly the difference being measured.
    """
    tok = model.token_layer
    if hasattr(tok, "n_core_params"):          # Fourier surrogate: coefficients
        return tok.n_core_params
    if hasattr(tok, "n_quantum_params"):       # PQC: circuit parameters
        return tok.n_quantum_params
    return sum(p.numel() for p in tok.parameters() if p.requires_grad)


def run(signal, mode, seed, epochs, lookback, limit, device, n_freq=4,
        verbose=False):
    """Identical to train.run(), with the surrogate arm added, so that numbers
    from this script and from train.py sit in the same units."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    series = SIGNALS[signal]()
    if limit:
        series = series[:limit]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = windows(series, lookback)
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    Xtr, ytr, Xva, yva, Xte, yte = map(to, (Xtr, ytr, Xva, yva, Xte, yte))

    t0 = time.time()
    model = Forecaster(mode=mode, n_freq=n_freq, seed=seed).to(device)
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
            print(f"    epoch {ep+1:>3d}/{epochs} val_mse {va:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte)
        mse = loss_fn(pred, yte).item()
        mae = (pred - yte).abs().mean().item()
    return {"signal": signal, "mode": mode,
            "n_freq": n_freq if mode == "fourier" else "",
            "seed": seed, "epochs": epochs, "limit": limit,
            "test_mse": mse, "test_mae": mae,
            "params": n_params, "core_params": core_param_count(model),
            "quantum_params": model.n_quantum_params,
            "wall_clock_s": round(time.time() - t0, 1)}


def append(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def summarise(rows):
    """The capacity curve, against the quantum arm as reference."""
    byq = [r["test_mse"] for r in rows if r["mode"] == "quantum"]
    byc = [r["test_mse"] for r in rows if r["mode"] == "classical"]
    print("\n" + "=" * 68)
    if byq:
        print(f"  quantum PQC        core params {rows[0]['quantum_params']:>4}"
              f"   test MSE {np.mean(byq):.6g}  (n={len(byq)})")
    if byc:
        print(f"  classical bottleneck                  "
              f"test MSE {np.mean(byc):.6g}  (n={len(byc)})")
    print("\n  Fourier surrogate — classical, spanning the circuit's spectrum")
    print(f"  {'n_freq':>7} {'core params':>12} {'test MSE':>12} "
          f"{'vs quantum':>11}   n")
    fr = sorted({r["n_freq"] for r in rows if r["mode"] == "fourier"},
                key=lambda v: int(v))
    ref = np.mean(byq) if byq else None
    crossed = None
    for nf in fr:
        sub = [r for r in rows if r["mode"] == "fourier" and r["n_freq"] == nf]
        mse = np.mean([r["test_mse"] for r in sub])
        cp = sub[0]["core_params"]
        rel = f"{mse/ref:.3f}" if ref else "—"
        mark = ""
        if ref and mse <= ref and crossed is None:
            crossed, mark = cp, "  <- matches"
        print(f"  {nf:>7} {cp:>12} {mse:>12.6g} {rel:>11}{mark}   {len(sub)}")
    print()
    if crossed is not None and ref:
        q = rows[0]["quantum_params"]
        print(f"  A classical Fourier map with {crossed} coefficients matches the "
              f"{q}-parameter circuit.")
        if crossed <= 2 * q:
            print("  The budgets are comparable, and a RANDOM SAMPLE of the circuit's")
            print("  spectrum was enough to reach it. On this evidence the circuit is")
            print("  a particular parameterisation of a classical feature map, not a")
            print("  distinct function class: there is no expressivity route to an")
            print("  advantage at this scale.")
        else:
            print(f"  The surrogate needs {crossed/q:.1f}x the circuit's budget — but note")
            print("  this arm samples the spectrum rather than spanning it, so a")
            print("  larger budget may only be compensating for a poor frequency")
            print("  draw. Check the seed-to-seed spread before reading this as an")
            print("  inductive-bias effect, and rerun with more seeds if it is wide.")
    elif ref:
        print("  No surrogate size tested matched the circuit. This is the weak")
        print("  direction of the experiment: it is consistent with an inductive-bias")
        print("  effect, but equally with the frequency sample missing what matters.")
        print("  Extend the n_freq sweep upward before drawing any conclusion.")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="mackey_glass", choices=list(SIGNALS))
    ap.add_argument("--modes", nargs="+",
                    default=["quantum", "classical", "fourier"])
    ap.add_argument("--n_freq", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        for mode in args.modes:
            freqs = args.n_freq if mode == "fourier" else [0]
            for nf in freqs:
                label = f"{mode}" + (f"(n_freq={nf})" if mode == "fourier" else "")
                print(f"== {args.signal} | {label} | seed {seed} ==", flush=True)
                r = run(args.signal, mode, seed, args.epochs, args.lookback,
                        args.limit, args.device, n_freq=nf)
                print(f"   test_mse {r['test_mse']:.6g}  "
                      f"core params {r['core_params']}  "
                      f"({r['wall_clock_s']:.0f}s)", flush=True)
                rows.append(r)

    append(rows)
    summarise(rows)
    print(f"\nappended {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
