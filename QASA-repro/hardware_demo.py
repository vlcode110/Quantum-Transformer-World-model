"""The cheapest honest hardware measurement: four jobs.

The full probe measures a noise floor, which needs many repeats and is therefore
expensive on a real device. That is the wrong thing to buy. The noise floor is
already known from IonQ's own hardware-calibrated simulator; what only real
hardware can establish is that the predicted SIGNAL survives on a physical
machine.

So this measures one gradient per entangler, at the same fixed parameter point
used throughout the simulator work, and compares each to its exact value. The
claim it supports:

    At 8 qubits on trapped-ion hardware the brickwork gradient is resolved
    while the ring-cascade gradient is not, as simulation predicted.

The noise model used for the prediction must name the SAME machine the job runs
on. Forte and Aria differ in fidelity, so a floor measured under `aria-1` does
not describe a Forte device.

That needs no repeats, because the uncertainty on a single measurement is known
in advance: sampling noise is sqrt(1 - <Z>^2)/sqrt(shots) and the device
contribution was measured on the noise-model simulator. A measurement
consistent with zero against a known error bar is a result; it does not require
a sample of measurements to establish the error bar from scratch.

Cost: 2 circuit evaluations per gradient (the parameter-shift rule), two
entanglers, so FOUR jobs. Nothing is submitted without --confirm.

    python hardware_demo.py                      cost estimate only
    python hardware_demo.py --confirm            submit to hardware
    python hardware_demo.py --device ionq.simulator --noise-model forte-1 --confirm
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_exact import reference          # independent exact implementation


def circuit_counts(n_qubits, n_layers, entangler):
    """Gate counts for the cost estimator, as written (before compilation)."""
    wires = n_qubits + 1
    one_q = (2 * n_qubits + wires) * n_layers       # RX, RZ per data wire; RY all
    two_q = (wires if entangler == "ring_cascade"
             else 2 * ((wires - 1) // 2)) * n_layers
    return wires, one_q, two_q


def exact_gradient(inputs, weights, n_qubits, n_layers, entangler, idx=(0, 0)):
    """Parameter-shift gradient of <Z_0>, computed exactly in NumPy."""
    wp, wm = weights.copy(), weights.copy()
    wp[idx] += np.pi / 2
    wm[idx] -= np.pi / 2
    fp = reference(inputs, wp, n_qubits, n_layers, entangler)[0]
    fm = reference(inputs, wm, n_qubits, n_layers, entangler)[0]
    return (fp - fm) / 2.0


def device_gradient(dev, inputs, weights, n_qubits, n_layers, entangler,
                    idx=(0, 0)):
    import pennylane as qml
    from gradient_variance import entangle
    wires = n_qubits + 1

    @qml.qnode(dev)
    def circuit(x, w):
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RX(x[i], wires=i)
                qml.RZ(x[i], wires=i)
            for q in range(wires):
                qml.RY(w[l, q], wires=q)
            entangle(wires, entangler)
        return qml.expval(qml.PauliZ(0))

    wp, wm = weights.copy(), weights.copy()
    wp[idx] += np.pi / 2
    wm[idx] -= np.pi / 2
    return float((circuit(inputs, wp) - circuit(inputs, wm)) / 2.0)


def explain_submission_failure(exc, args):
    """Turn an API rejection into something actionable.

    The plugin re-raises the raw response body inside a chain of PennyLane
    frames, so the one useful sentence arrives at the end of a long traceback.
    These failures are also the cheap kind — a rejected job consumes no shots —
    so the right response is a clear instruction, not a stack trace.
    """
    text = str(exc)
    print("\n  SUBMISSION REJECTED — no shots were consumed.\n")
    try:
        msg = json.loads(text).get("message", text)
    except Exception:
        msg = text.strip().splitlines()[-1] if text.strip() else repr(exc)
    print(f"    {msg}\n")

    low = msg.lower()
    if "not have" in low and "enabled" in low:
        print("  The target is not activated for this project. In the IonQ")
        print("  console: Project Settings -> QPU Targets -> activate the")
        print(f"  target, then rerun. Check the per-shot price while you are")
        print("  there; enabling a QPU may also require billing details.\n")
        print("  To confirm everything else works first, run the identical")
        print("  measurement against the noise-model simulator — same circuit,")
        print("  same parameters, no QPU entitlement needed:\n")
        print("      python hardware_demo.py --device ionq.simulator \\")
        print(f"          --noise-model {args.backend} --confirm")
    elif "credit" in low or "balance" in low or "quota" in low:
        print("  This looks like a credit or quota limit rather than a code")
        print("  problem. Check the balance in the IonQ console.")
    elif "401" in low or "unauthor" in low or "forbidden" in low:
        print("  Authentication failed. Confirm IONQ_API_KEY is current:")
        print("      python check_env.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--entanglers", nargs="+",
                    default=["brickwork", "ring_cascade"])
    ap.add_argument("--device", default="ionq.qpu")
    ap.add_argument("--backend", default="forte-1",
                    help="physical machine; aria-1 is retired")
    ap.add_argument("--noise-model", default=None,
                    help="simulator only; omit for real hardware")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confirm", action="store_true",
                    help="actually submit; without it nothing is sent")
    ap.add_argument("--out", default="results/hardware_demo.json")
    args = ap.parse_args()

    n, L = args.n_qubits, args.n_layers
    # The SAME fixed point the simulator runs used, so the comparison is exact.
    rng_fixed = np.random.default_rng(args.seed + 10_000)
    wires = n + 1
    weights = rng_fixed.uniform(0, 2 * np.pi, (L, wires))
    inputs = rng_fixed.uniform(-np.pi, np.pi, n)

    jobs = 2 * len(args.entanglers)
    print(f"device {args.device}" +
          (f" backend {args.backend}" if args.device == "ionq.qpu" else "") +
          f", {n} data qubits ({wires} wires), {L} layers, "
          f"{args.shots} shots\n")
    print("  cost estimator inputs")
    for ent in args.entanglers:
        w, o, t = circuit_counts(n, L, ent)
        print(f"    {ent:<13} qubits {w}, one-qubit gates {o}, "
              f"two-qubit gates {t}")
    print(f"\n  {jobs} jobs x {args.shots} shots = "
          f"{jobs * args.shots:,} shots total")

    print("\n  expected result (from exact simulation + measured noise)")
    exact = {}
    for ent in args.entanglers:
        g = exact_gradient(inputs, weights, n, L, ent)
        exact[ent] = g
        print(f"    {ent:<13} exact gradient {g:+.4e}")

    if not args.confirm:
        print("\n  Nothing submitted. Re-run with --confirm to execute.")
        return

    import pennylane as qml
    kwargs = {"wires": wires, "shots": args.shots}
    if args.device.startswith("ionq"):
        key = os.environ.get("IONQ_API_KEY")
        if not key:
            sys.exit("IONQ_API_KEY is not set.")
        kwargs["api_key"] = key
        if args.device == "ionq.qpu":
            kwargs["backend"] = args.backend
        elif args.noise_model:
            kwargs["noise_model"] = args.noise_model

    print(f"\n  submitting {jobs} jobs...")
    out, t0 = {}, time.time()
    for ent in args.entanglers:
        dev = qml.device(args.device, **kwargs)
        t = time.time()
        try:
            g = device_gradient(dev, inputs, weights, n, L, ent)
        except Exception as e:
            explain_submission_failure(e, args)
            sys.exit(1)
        out[ent] = {"measured": g, "exact": exact[ent],
                    "seconds": round(time.time() - t, 1)}
        print(f"    {ent:<13} measured {g:+.4e}   exact {exact[ent]:+.4e}   "
              f"({out[ent]['seconds']:.0f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"device": args.device, "backend": args.backend,
                   "n_qubits": n, "n_layers": L, "shots": args.shots,
                   "seed": args.seed, "results": out,
                   "total_shots": jobs * args.shots}, f, indent=1)
    print(f"\n  {time.time()-t0:.0f}s total -> {args.out}")
    print("\n  Read against the noise scale measured on the aria-1 simulator:")
    print("  a value within that scale of zero is unresolved, which for the")
    print("  ring cascade is the predicted outcome and the point of the run.")


if __name__ == "__main__":
    main()
