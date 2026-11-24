"""Barren-plateau probe: does gradient variance decay exponentially with
circuit width?  (Phase 0 gate: "quantum components train stably at 8+ qubits";
success metric 7.3: "no exponential decay in the working regime".)

Method (standard in the barren-plateau literature, e.g. arXiv:2405.00781):
for each width, sample many *independent random* parameter settings, compute
the gradient of the cost at each, and take the variance across samples. If
Var[dC/dtheta] falls off geometrically in qubit count, the landscape is
flattening and training will stall as you scale.

Two cost functions are measured, because the distinction drives the design:
  - global : <Z_0 Z_1 ... Z_n>          — known to plateau quickly
  - local  : <Z_0>                      — provably milder decay
Our QASA layer reads out local <Z_i>, so 'local' is the operative curve; the
global one is the control that shows the probe can detect a plateau.

Usage:
    python gradient_variance.py                      # 2..12 qubits, 6 layers
    python gradient_variance.py --max-qubits 14 --samples 200
    python gradient_variance.py --layers 2 6 12      # depth dependence
    python gradient_variance.py --max-qubits 6 --samples 20   # smoke (~1 min)

Single-threaded by default so it can share a machine with a training run.
"""
import argparse
import csv
import os

import numpy as np
import pennylane as qml

# Torch is needed only to differentiate the circuit here. `shot_noise_probe.py`
# imports `entangle` from this module for the wiring definitions and uses the
# parameter-shift rule instead, so it must be able to import this file in an
# environment without torch — for instance the separate one needed for IonQ's
# noise models, where the dependency pins differ.
try:
    import torch
except ImportError:                                      # pragma: no cover
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError(
            "This function needs PyTorch, which is not installed in this "
            "environment.\n`entangle` can be imported without it; "
            "`build` and the sweep cannot.")


def entangle(wires, pattern):
    """Entangling block.

    'ring_cascade' — CNOT(0,1), CNOT(1,2), ... applied SEQUENTIALLY. Note this
        is not a depth-1 block: correlations propagate the whole way round the
        register within one "layer", giving circuit depth ~n. It scrambles fast,
        which makes even a local observable behave globally and produces
        Haar-like 2^-n gradient decay independent of the nominal layer count.
    'brickwork' — even pairs then odd pairs: genuinely depth 2 at any width, so
        the light cone grows with the layer count as intended and local
        observables keep their milder scaling.
    """
    if pattern == "ring_cascade":
        for w in range(wires):
            qml.CNOT(wires=[w, (w + 1) % wires])
    elif pattern == "brickwork":
        for w in range(0, wires - 1, 2):
            qml.CNOT(wires=[w, w + 1])
        for w in range(1, wires - 1, 2):
            qml.CNOT(wires=[w, w + 1])
    else:
        raise ValueError(f"unknown entangler {pattern!r}")


def build(n_qubits, n_layers, cost, entangler="ring_cascade"):
    """QASA-style re-uploading circuit; returns a differentiable cost fn."""
    _require_torch()
    wires = n_qubits + 1
    dev = qml.device("default.qubit", wires=wires)

    if cost == "global":
        obs = qml.prod(*[qml.PauliZ(i) for i in range(wires)])
    else:
        obs = qml.PauliZ(0)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for l in range(n_layers):
            for i in range(n_qubits):          # data re-uploading (RX/RZ)
                qml.RX(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i)
            for w in range(wires):             # trainable rotations
                qml.RY(weights[l, w], wires=w)
            entangle(wires, entangler)
        return qml.expval(obs)

    return circuit, (n_layers, wires)


def sample_gradients(n_qubits, n_layers, cost, n_samples, seed=0,
                     entangler="ring_cascade"):
    """Gradient of the cost w.r.t. one fixed parameter, over random inits."""
    circuit, wshape = build(n_qubits, n_layers, cost, entangler)
    rng = np.random.default_rng(seed)
    grads = []
    for _ in range(n_samples):
        # random parameters over the full rotation range, random data angles
        w = torch.tensor(rng.uniform(0, 2 * np.pi, wshape), requires_grad=True)
        x = torch.tensor(rng.uniform(-np.pi, np.pi, n_qubits))
        out = circuit(x, w)
        out.backward()
        # track a single, consistently-placed parameter (first layer, wire 0):
        # variance of one component is the standard barren-plateau diagnostic
        grads.append(float(w.grad[0, 0]))
    return np.array(grads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-qubits", type=int, default=2)
    ap.add_argument("--max-qubits", type=int, default=12)
    ap.add_argument("--layers", type=int, nargs="+", default=[6])
    ap.add_argument("--costs", nargs="+", default=["local", "global"])
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads; keep at 1 to share the machine")
    ap.add_argument("--entanglers", nargs="+", default=["ring_cascade"],
                    choices=["ring_cascade", "brickwork"],
                    help="entangling pattern(s) to sweep — see entangle()")
    ap.add_argument("--out", default="results/gradient_variance.csv")
    args = ap.parse_args()

    _require_torch()
    torch.set_num_threads(args.threads)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    print(f"{args.samples} random inits per point, {args.threads} thread(s)\n")
    print(f"{'entangler':>13s} {'cost':>7s} {'layers':>6s} {'qubits':>6s} "
          f"{'Var[grad]':>12s} {'|grad| mean':>12s}")
    for ent in args.entanglers:
        for cost in args.costs:
            for n_layers in args.layers:
                for n_qubits in range(args.min_qubits, args.max_qubits + 1):
                    g = sample_gradients(n_qubits, n_layers, cost,
                                         args.samples, args.seed, ent)
                    var, absmean = float(g.var()), float(np.abs(g).mean())
                    print(f"{ent:>13s} {cost:>7s} {n_layers:>6d} {n_qubits:>6d} "
                          f"{var:>12.3e} {absmean:>12.3e}")
                    rows.append({"entangler": ent, "cost": cost,
                                 "layers": n_layers, "qubits": n_qubits,
                                 "samples": args.samples,
                                 "grad_var": var, "grad_absmean": absmean})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # decay factor per added qubit, fitted on the upper half of each sweep
    print("\n=== decay of Var[grad] per added qubit (upper half of sweep) ===")
    factors = {}
    for ent in args.entanglers:
        for cost in args.costs:
            for n_layers in args.layers:
                sub = [r for r in rows if r["entangler"] == ent
                       and r["cost"] == cost and r["layers"] == n_layers]
                q = np.array([r["qubits"] for r in sub], float)
                v = np.array([max(r["grad_var"], 1e-30) for r in sub])
                half = len(q) // 2
                if len(q) - half >= 2:
                    factor = float(np.exp(np.polyfit(q[half:], np.log(v[half:]), 1)[0]))
                    factors[(ent, cost, n_layers)] = factor
                    verdict = ("EXPONENTIAL DECAY — plateau risk" if factor < 0.6
                               else "mild" if factor < 0.9 else "flat / benign")
                    print(f"  {ent:>13s} {cost:>7s} L={n_layers}: "
                          f"x{factor:.2f} per qubit — {verdict}")

    # Diagnostic: local and global decaying at the SAME rate means the circuit
    # is scrambling so fast that locality of the observable buys nothing —
    # usually a sign the entangling block is deeper than intended, not a
    # fundamental property of the model.
    for ent in args.entanglers:
        for n_layers in args.layers:
            loc = factors.get((ent, "local", n_layers))
            glo = factors.get((ent, "global", n_layers))
            if loc and glo and abs(loc - glo) < 0.08:
                print(f"\n  NOTE [{ent}, L={n_layers}]: local ({loc:.2f}) and "
                      f"global ({glo:.2f}) decay at the same rate. Locality is "
                      f"buying nothing, which points at the entangler rather "
                      f"than at width. Compare --entanglers brickwork.")
                break
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
