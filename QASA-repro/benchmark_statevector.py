"""Statevector throughput benchmark (Phase 0, step 4).

Measures forward+backward wall-clock for a QASA-style re-uploading circuit
across qubit counts, layer depths, and simulators, at the batch shape the
world-model loop will actually use (batch 16 x seq 32 = 512 circuit evals
per gradient step). Prints the feasible qubit envelope for in-loop training.

Usage:
    python benchmark_statevector.py                     # defaults: 2..16 qubits
    python benchmark_statevector.py --max-qubits 20 --layers 3 6
    python benchmark_statevector.py --budget 30         # per-config timeout (s)

Run on an otherwise idle machine — background jobs skew timings.
"""
import argparse
import csv
import os
import time

import numpy as np
import pennylane as qml
import torch


def make_circuit(device_name, n_qubits, n_layers):
    wires = n_qubits + 1  # QASA layout: data qubits + 1 ancilla
    try:
        dev = qml.device(device_name, wires=wires)
    except Exception:
        return None
    diff = "backprop" if device_name == "default.qubit" else "adjoint"

    @qml.qnode(dev, interface="torch", diff_method=diff)
    def circuit(inputs, weights):
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RX(inputs[..., i], wires=i)
                qml.RZ(inputs[..., i], wires=i)
            for w in range(wires):
                qml.RY(weights[l, w], wires=w)
            for w in range(wires):
                qml.CNOT(wires=[w, (w + 1) % wires])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit, (n_layers, wires)


def time_config(device_name, n_qubits, n_layers, batch, repeats, budget):
    made = make_circuit(device_name, n_qubits, n_layers)
    if made is None:
        return None  # simulator not installed
    circuit, wshape = made
    weights = torch.randn(*wshape, requires_grad=True)
    x = torch.randn(batch, n_qubits)

    # Warmup doubles as a pre-flight probe: it catches simulators that do not
    # support parameter broadcasting (they fall back to looping over the batch,
    # which is slower by orders of magnitude) BEFORE we commit to `repeats`
    # timed runs. Without this the script looks hung on lightning.qubit.
    try:
        t0 = time.perf_counter()
        out = torch.stack(circuit(x, weights), -1).sum()
        out.backward()
        warm = time.perf_counter() - t0
    except Exception as e:
        return {"error": str(e)[:80]}
    if warm > budget:
        return {"ms_per_step": 1000 * warm, "aborted": True,
                "note": f"single run took {warm:.1f}s > {budget:g}s budget"}

    times = []
    for _ in range(repeats):
        weights.grad = None
        t0 = time.perf_counter()
        out = torch.stack(circuit(x, weights), -1).sum()
        out.backward()
        times.append(time.perf_counter() - t0)
        if sum(times) > budget:
            break
    return {"ms_per_step": 1000 * float(np.median(times))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-qubits", type=int, default=16)
    ap.add_argument("--layers", type=int, nargs="+", default=[6])
    ap.add_argument("--batch", type=int, default=512,
                    help="circuit evals per gradient step (16 x seq 32)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--budget", type=float, default=30.0,
                    help="max seconds per config before skipping larger ones")
    ap.add_argument("--devices", nargs="+",
                    default=["default.qubit", "lightning.qubit"])
    ap.add_argument("--out", default=None,
                    help="output CSV; defaults to a per-device filename so a "
                         "new sweep never overwrites earlier measurements")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    if args.out:
        path = args.out
    elif args.devices == ["default.qubit", "lightning.qubit"]:
        path = os.path.join("results", "statevector_benchmark.csv")
    else:
        tag = "_".join(d.replace(".", "-") for d in args.devices)
        path = os.path.join("results", f"statevector_benchmark_{tag}.csv")
    rows = []
    fieldnames = ["device", "qubits", "layers", "batch", "ms_per_step"]
    # Write incrementally: a sweep can take a long time and may be interrupted,
    # and losing completed measurements to a Ctrl-C is inexcusable.
    csv_file = open(path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader(); csv_file.flush()

    print(f"batch={args.batch} evals/step; times are median fwd+bwd per step\n")
    print(f"{'device':>16s} {'qubits':>6s} {'layers':>6s} {'ms/step':>10s} {'steps/s':>8s}")

    try:
        for device_name in args.devices:
            for n_layers in args.layers:
                dead = False
                for n_qubits in range(2, args.max_qubits + 1):
                    if dead:
                        break
                    # announce BEFORE timing, so a slow config is visible
                    print(f"{device_name:>16s} {n_qubits:>6d} {n_layers:>6d} "
                          f"{'...':>10s}", end="\r", flush=True)
                    r = time_config(device_name, n_qubits, n_layers,
                                    args.batch, args.repeats, args.budget)
                    if r is None:
                        print(f"{device_name:>16s}  -- not installed (pip install "
                              f"pennylane-lightning) --")
                        dead = True
                        continue
                    if "error" in r:
                        print(f"{device_name:>16s} {n_qubits:>6d} {n_layers:>6d}  "
                              f"ERROR: {r['error']}")
                        continue
                    ms = r["ms_per_step"]
                    if r.get("aborted"):
                        print(f"{device_name:>16s} {n_qubits:>6d} {n_layers:>6d} "
                              f"{ms:>10.1f}  ABORTED — {r['note']}")
                        print(f"{'':>16s}  (skipping the rest of {device_name}; "
                              f"if this is lightning.qubit it is looping over "
                              f"the batch rather than broadcasting — try "
                              f"--batch 1)")
                        dead = True
                        continue
                    print(f"{device_name:>16s} {n_qubits:>6d} {n_layers:>6d} "
                          f"{ms:>10.1f} {1000/ms:>8.2f}")
                    row = {"device": device_name, "qubits": n_qubits,
                           "layers": n_layers, "batch": args.batch,
                           "ms_per_step": ms}
                    rows.append(row)
                    writer.writerow(row); csv_file.flush()
                    if ms > args.budget * 1000 / 3:
                        print(f"{'':>16s}  (stopping {device_name} at "
                              f"{n_qubits} qubits — exceeding budget)")
                        dead = True
    except KeyboardInterrupt:
        print("\n[interrupted] measurements so far are already saved.")
    finally:
        csv_file.close()

    # Feasible envelope: world-model training does ~1 gradient step per 2 env
    # steps; call a config feasible if a step takes <= 1s (>= ~86k steps/day).
    print("\n=== feasible envelope (<= 1000 ms per gradient step) ===")
    for device_name in args.devices:
        feas = [r["qubits"] for r in rows
                if r["device"] == device_name and r["ms_per_step"] <= 1000]
        if feas:
            print(f"{device_name}: up to {max(feas)} qubits")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
