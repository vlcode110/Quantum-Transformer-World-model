"""Where does measurement noise swallow the gradient?

Phase 0 measured gradient variance on a NOISELESS simulator, where a gradient
of 1e-4 is represented exactly and training proceeds normally. On real hardware
every expectation value is estimated from a finite number of measurements, so
the smallest resolvable gradient is set by sampling noise (~1/sqrt(shots)). A
gradient far below that floor is indistinguishable from noise and the optimiser
is stepping at random.

This script measures both quantities at each configuration:

  SIGNAL  typical |gradient| across random parameter settings
          — how big the real gradient is, i.e. what the landscape offers

  NOISE   spread of repeated gradient estimates at ONE FIXED parameter setting
          — how much the measurement itself wobbles

Trainability requires SIGNAL >> NOISE. The width at which their ratio falls to
about 1 is the hardware ceiling for that circuit, and it is a number rather
than an argument.

Efficiency note: gradient variance only needs the derivative with respect to a
single parameter, and the parameter-shift rule gets that in exactly two circuit
evaluations regardless of shot count. That is what makes a remote device
practical here — a few hundred calls per configuration rather than millions.

Usage
    # exact reference (no sampling noise) — fast, local
    python shot_noise_probe.py --shots 0 --max-qubits 10

    # the local shot-noise sweep — do this first, it is free
    python shot_noise_probe.py --shots 100 1000 10000 --max-qubits 12

    # realistic device noise, needs IONQ_API_KEY in the environment
    export IONQ_API_KEY=...
    python shot_noise_probe.py --device ionq.simulator --shots 1000 \\
        --noise-model aria-1 --max-qubits 16 --samples 30 --repeats 30
"""
import argparse
import csv
import os
import time

import numpy as np
import pennylane as qml

from gradient_variance import entangle   # reuse the exact wiring definitions


# --------------------------------------------------------------- the circuit
def make_device(name, wires, shots, noise_model=None, backend=None):
    """Build the execution device.

    Remote devices need credentials. `ionq.simulator` accepts a
    hardware-calibrated `noise_model`; `ionq.qpu` instead requires a `backend`
    naming the physical machine, and charges per shot for every job.
    """
    kwargs = {"wires": wires}
    if shots:
        kwargs["shots"] = shots
    if name.startswith("ionq"):
        key = os.environ.get("IONQ_API_KEY")
        if not key:
            raise SystemExit(
                "IONQ_API_KEY is not set. Run:  export IONQ_API_KEY=your_key")
        kwargs["api_key"] = key
        if name == "ionq.qpu":
            if not backend:
                raise SystemExit(
                    "ionq.qpu needs a physical machine: pass --backend aria-1")
            kwargs["backend"] = backend
            if noise_model and noise_model != "none":
                raise SystemExit(
                    "--noise-model applies to the simulator only; real hardware "
                    "supplies its own noise.")
        elif noise_model and noise_model != "none":
            kwargs["noise_model"] = noise_model
    return qml.device(name, **kwargs)


def build(device_name, n_qubits, n_layers, cost, entangler, shots, noise_model,
          backend=None):
    """Scalar-valued QASA-style circuit. Returns (qnode, weight_shape).

    The numpy interface is used rather than torch: we differentiate by hand
    with the parameter-shift rule below, so no autograd machinery is needed and
    the same code runs unchanged against a remote device.
    """
    wires = n_qubits + 1
    dev = make_device(device_name, wires, shots, noise_model, backend)
    obs = (qml.prod(*[qml.PauliZ(i) for i in range(wires)]) if cost == "global"
           else qml.PauliZ(0))

    @qml.qnode(dev)
    def circuit(inputs, weights):
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i)
            for w in range(wires):
                qml.RY(weights[l, w], wires=w)
            entangle(wires, entangler)
        return qml.expval(obs)

    return circuit, (n_layers, wires)


def grad_one(circuit, x, w, idx=(0, 0)):
    """Derivative with respect to a single parameter, by the parameter-shift
    rule: for a rotation gate, df/dtheta = [f(theta + pi/2) - f(theta - pi/2)]/2.

    Exactly two circuit evaluations, whatever the shot count. This is what
    keeps a remote device affordable."""
    wp, wm = w.copy(), w.copy()
    wp[idx] += np.pi / 2
    wm[idx] -= np.pi / 2
    return float((circuit(x, wp) - circuit(x, wm)) / 2.0)


def _with_progress(circuit, total, enabled, label=""):
    """Report progress when every evaluation is a remote job.

    On a local simulator an evaluation takes microseconds and 400 of them pass
    unnoticed. On a cloud device each one is a separate submit-queue-poll-retrieve
    round trip taking seconds, so the same sweep runs for hours with no output
    and looks indistinguishable from a hang. This prints a running count, the
    measured per-job latency and an estimate of the time remaining.
    """
    if not enabled:
        return circuit
    state = {"n": 0, "t0": time.time()}

    def wrapped(*a, **kw):
        out = circuit(*a, **kw)
        state["n"] += 1
        n = state["n"]
        elapsed = time.time() - state["t0"]
        if n == 1 or n % 5 == 0 or n == total:
            rate = elapsed / n
            eta = rate * (total - n) / 60.0
            print(f"\r      {label}{n}/{total} jobs, {rate:.1f}s each, "
                  f"~{eta:.1f} min left   ", end="", flush=True)
        if n == total:
            print()
        return out

    return wrapped


# ------------------------------------------------------------ the measurement
def measure(device_name, n_qubits, n_layers, cost, entangler, shots,
            noise_model, samples, repeats, seed=0, backend=None):
    circuit, wshape = build(device_name, n_qubits, n_layers, cost,
                            entangler, shots, noise_model, backend)
    remote = not device_name.startswith(("default.", "lightning."))
    total = (samples + (repeats if shots else 0)) * 2
    if remote:
        print(f"\n    remote device: {total} jobs for this configuration")
    circuit = _with_progress(circuit, total, remote,
                             label=f"{entangler} {n_qubits}q: ")
    rng = np.random.default_rng(seed)

    # SIGNAL: how large is the gradient across the landscape
    grads = []
    for _ in range(samples):
        w = rng.uniform(0, 2 * np.pi, wshape)
        x = rng.uniform(-np.pi, np.pi, n_qubits)
        grads.append(grad_one(circuit, x, w))
    grads = np.asarray(grads)

    # NOISE: how much does one measurement wobble when nothing changes.
    #
    # The fixed point is drawn from its OWN stream, not from `rng`. Drawing it
    # from the shared stream made it depend on how many draws the signal loop
    # above had already consumed, so changing `--samples` silently moved the
    # point at which noise was measured. Sampling noise depends on <Z> at that
    # point — it is largest near zero — so two runs differing only in `samples`
    # produced noise figures that were not comparable, which is exactly the
    # comparison this script exists to make.
    rng_fixed = np.random.default_rng(seed + 10_000)
    w_fixed = rng_fixed.uniform(0, 2 * np.pi, wshape)
    x_fixed = rng_fixed.uniform(-np.pi, np.pi, n_qubits)
    if shots:
        repeated = np.asarray([grad_one(circuit, x_fixed, w_fixed)
                               for _ in range(repeats)])
        noise_sd = float(repeated.std(ddof=1))
    else:
        noise_sd = 0.0        # exact simulation has no measurement noise

    signal = float(np.abs(grads).mean())
    return {
        "device": device_name, "noise_model": noise_model or "none",
        "shots": shots or 0, "entangler": entangler, "cost": cost,
        "layers": n_layers, "qubits": n_qubits, "samples": samples,
        "grad_var": float(grads.var()), "signal_absmean": signal,
        "noise_sd": noise_sd,
        "snr": float(signal / noise_sd) if noise_sd > 0 else float("inf"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="default.qubit")
    ap.add_argument("--backend", default=None,
                    help="physical machine for ionq.qpu, e.g. aria-1")
    ap.add_argument("--noise-model", default="none",
                    help="ionq.simulator only, e.g. aria-1, forte-1")
    ap.add_argument("--shots", type=int, nargs="+", default=[0],
                    help="0 means exact (no sampling noise)")
    ap.add_argument("--min-qubits", type=int, default=4)
    ap.add_argument("--max-qubits", type=int, default=12)
    ap.add_argument("--layers", type=int, nargs="+", default=[2])
    ap.add_argument("--entanglers", nargs="+",
                    default=["brickwork", "ring_cascade"])
    ap.add_argument("--costs", nargs="+", default=["local"])
    ap.add_argument("--samples", type=int, default=100,
                    help="random parameter settings (lower this for remote)")
    ap.add_argument("--repeats", type=int, default=100,
                    help="repeated measurements at a fixed setting")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = args.device.replace(".", "-")
    if args.noise_model != "none":
        tag += f"_{args.noise_model}"
    out = args.out or os.path.join("results", f"shot_noise_{tag}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    calls = 2 * (args.samples + args.repeats)
    n_cfg = (len(args.shots) * len(args.entanglers) * len(args.costs)
             * len(args.layers) * (args.max_qubits - args.min_qubits + 1))
    print(f"device {args.device}  noise {args.noise_model}")
    print(f"{n_cfg} configurations x ~{calls} circuit evaluations "
          f"= ~{n_cfg * calls:,} total\n")
    print(f"{'shots':>7} {'entangler':>13} {'qubits':>6} {'signal':>11} "
          f"{'noise sd':>11} {'SNR':>8}  verdict")

    rows, t0 = [], time.time()
    with open(out, "w", newline="") as f:
        writer = None
        for shots in args.shots:
            for ent in args.entanglers:
                for cost in args.costs:
                    for L in args.layers:
                        dead = False
                        for n in range(args.min_qubits, args.max_qubits + 1):
                            if dead:
                                break
                            try:
                                r = measure(args.device, n, L, cost, ent, shots,
                                            args.noise_model, args.samples,
                                            args.repeats, backend=args.backend)
                            except Exception as e:
                                print(f"{shots:>7} {ent:>13} {n:>6}  "
                                      f"ERROR: {type(e).__name__}: {str(e)[:60]}")
                                dead = True
                                continue
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=list(r))
                                writer.writeheader()
                            writer.writerow(r); f.flush()
                            rows.append(r)
                            snr = r["snr"]
                            verdict = ("exact" if shots == 0 else
                                       "trainable" if snr > 5 else
                                       "marginal" if snr > 1 else
                                       "BURIED IN NOISE")
                            print(f"{shots:>7} {ent:>13} {n:>6} "
                                  f"{r['signal_absmean']:>11.3e} "
                                  f"{r['noise_sd']:>11.3e} "
                                  f"{snr:>8.2f}  {verdict}")

    # ---- the number this whole script exists to produce
    print(f"\n=== hardware ceiling: widest circuit with SNR > 1 ===")
    for shots in args.shots:
        if shots == 0:
            continue
        for ent in args.entanglers:
            ok = [r["qubits"] for r in rows
                  if r["shots"] == shots and r["entangler"] == ent
                  and r["snr"] > 1.0]
            widest = max(ok) if ok else None
            print(f"  {shots:>6} shots, {ent:>13}: "
                  + (f"{widest} qubits" if widest else "no width resolvable"))
    print(f"\n{len(rows)} configurations in {(time.time()-t0)/60:.1f} min -> {out}")
    print("\nSNR is the typical gradient divided by the spread of repeated\n"
          "measurements of it. Below 1 the optimiser cannot tell which way to\n"
          "step, however healthy the noiseless gradient variance looked.")


if __name__ == "__main__":
    main()
