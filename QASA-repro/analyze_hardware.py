"""Reconstruct the gradients measured on IonQ Forte-1 hardware.

Four jobs were submitted, each one circuit evaluation: the parameter-shift rule
needs f(theta + pi/2) and f(theta - pi/2) per gradient, for two entanglers.
IonQ returns a probability histogram over basis states rather than an
expectation value, so this converts each histogram to <Z_0>, pairs them, and
compares the result against exact simulation.

Two things are settled here rather than assumed.

  WIRE ORDER. The histogram keys are basis-state indices, and whether wire 0 is
      the most or least significant bit changes the sign and magnitude of
      <Z_0>. Rather than trusting a convention, both are computed and the one
      matching exact simulation is reported — with the mismatch of the other
      shown, so the choice is visible rather than hidden.

  JOB MAPPING. Jobs are matched to circuits by submission timestamp, following
      the order hardware_demo.py submits them in. The metadata .txt files carry
      those timestamps.

    python analyze_hardware.py
    python analyze_hardware.py --dir results/hardware

Expects the four result histograms as JSON named by job id, alongside the
metadata files (either extension) containing `id` and `request`.
"""
import argparse
import glob
import json
import os

import numpy as np

from verify_exact import reference

HERE = os.path.dirname(os.path.abspath(__file__))

# hardware_demo.py submits in this order
ORDER = [("brickwork", +1), ("brickwork", -1),
         ("ring_cascade", +1), ("ring_cascade", -1)]


def expectation_z0(hist, n_wires, wire0_is_msb):
    """<Z_0> from a histogram of basis-state probabilities."""
    bit = (n_wires - 1) if wire0_is_msb else 0
    return float(sum(p * (1.0 if not ((int(k) >> bit) & 1) else -1.0)
                     for k, p in hist.items()))


def load_metadata(d):
    """job id -> submission timestamp, from whichever metadata files exist."""
    meta = {}
    for path in glob.glob(os.path.join(d, "*")):
        if path.endswith(".json") and "request" not in open(path).read(200):
            continue                       # a histogram, not metadata
        try:
            j = json.load(open(path))
        except Exception:
            continue
        if isinstance(j, dict) and "id" in j and "request" in j:
            meta[j["id"]] = j
    return meta


def load_histograms(d):
    """job id -> histogram, identified by the id appearing in the filename."""
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            j = json.load(open(path))
        except Exception:
            continue
        if not isinstance(j, dict) or "id" in j:
            continue                       # metadata, not a histogram
        keys = list(j)
        if not keys or not all(str(k).isdigit() for k in keys):
            continue
        stem = os.path.basename(path).split(".")[0]
        out[stem.split("-")[0] if len(stem) < 36 else stem[:36]] = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "results", "hardware"))
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-model-sd", type=float, default=2.449e-2,
                    help="noise sd measured on the forte-1 simulator at 1000 "
                         "shots, for comparison")
    args = ap.parse_args()

    n, L = args.n_qubits, args.n_layers
    wires = n + 1

    meta = load_metadata(args.dir)
    hists = load_histograms(args.dir)
    if len(hists) < 4:
        raise SystemExit(
            f"[hardware] found {len(hists)} histogram(s) in {args.dir}; "
            f"need 4.\n  Files present: "
            f"{[os.path.basename(p) for p in glob.glob(os.path.join(args.dir,'*'))]}")

    order = sorted(meta.values(), key=lambda j: j["request"])
    mapping = {}
    for job, (ent, sign) in zip(order, ORDER):
        jid = job["id"]
        match = next((h for k, h in hists.items() if k in jid or jid in k), None)
        if match is None:
            raise SystemExit(f"[hardware] no histogram found for job {jid}")
        mapping[(ent, sign)] = (jid, match)

    # the same fixed point every simulator measurement used
    rng = np.random.default_rng(args.seed + 10_000)
    weights = rng.uniform(0, 2 * np.pi, (L, wires))
    inputs = rng.uniform(-np.pi, np.pi, n)

    print(f"IonQ Forte-1, {n} data qubits ({wires} wires), {L} layers\n")

    # ------------------------------------------------- settle the wire order
    print("Wire convention")
    print(f"  {'circuit':<30}{'exact':>10}{'MSB':>10}{'LSB':>10}")
    err = {True: 0.0, False: 0.0}
    rows = []
    for (ent, sign), (jid, hist) in mapping.items():
        w = weights.copy()
        w[0, 0] += sign * np.pi / 2
        exact = float(reference(inputs, w, n, L, ent)[0])
        msb = expectation_z0(hist, wires, True)
        lsb = expectation_z0(hist, wires, False)
        err[True] += (msb - exact) ** 2
        err[False] += (lsb - exact) ** 2
        rows.append((ent, sign, jid, exact, msb, lsb))
        print(f"  {ent + (' +' if sign > 0 else ' -'):<30}"
              f"{exact:>10.4f}{msb:>10.4f}{lsb:>10.4f}")
    msb_better = err[True] <= err[False]
    print(f"\n  RMS deviation from exact:  MSB {np.sqrt(err[True]/4):.4f}   "
          f"LSB {np.sqrt(err[False]/4):.4f}")
    print(f"  -> treating wire 0 as the "
          f"{'MOST' if msb_better else 'LEAST'} significant bit\n")

    # ------------------------------------------------------- the gradients
    print("Gradients (parameter shift, hardware vs exact)")
    print(f"  {'entangler':<16}{'hardware':>12}{'exact':>12}{'difference':>13}")
    out = {}
    for ent in ("brickwork", "ring_cascade"):
        vals, exacts = {}, {}
        for sign in (+1, -1):
            jid, hist = mapping[(ent, sign)]
            vals[sign] = expectation_z0(hist, wires, msb_better)
            w = weights.copy()
            w[0, 0] += sign * np.pi / 2
            exacts[sign] = float(reference(inputs, w, n, L, ent)[0])
        g_hw = (vals[+1] - vals[-1]) / 2.0
        g_ex = (exacts[+1] - exacts[-1]) / 2.0
        out[ent] = {"hardware": g_hw, "exact": g_ex, "difference": g_hw - g_ex,
                    "evaluations": {str(s): {"hardware": vals[s],
                                             "exact": exacts[s]}
                                    for s in (+1, -1)}}
        print(f"  {ent:<16}{g_hw:>12.4f}{g_ex:>12.4f}{g_hw - g_ex:>+13.4f}")

    # ----------------------------------------------- measured hardware noise
    devs = [abs(expectation_z0(h, wires, msb_better)
                - float(reference(inputs,
                                  (lambda w, s=s: (w.__setitem__((0, 0),
                                   w[0, 0] + s * np.pi / 2) or w))(
                                      weights.copy()), n, L, e)[0]))
            for (e, s), (_j, h) in mapping.items()]
    rms = float(np.sqrt(np.mean(np.square(devs))))
    print(f"\nMeasured deviation from exact, RMS over 4 evaluations: {rms:.4f}")
    print(f"  forte-1 noise model predicted (1000 shots):           "
          f"{args.noise_model_sd:.4f}")
    print(f"  these ran at 504 shots, so sampling alone is ~"
          f"{args.noise_model_sd * np.sqrt(1000 / 504):.4f}")
    print("\n  If the measured deviation sits near the predicted value, IonQ's")
    print("  noise model is predictive of its own hardware — worth reporting,")
    print("  since the simulator results in the paper rest on that assumption.")

    dest = os.path.join(args.dir, "hardware_gradients.json")
    with open(dest, "w") as f:
        json.dump({"wire0_is_msb": bool(msb_better), "shots": 504,
                   "rms_deviation": rms, "gradients": out,
                   "jobs": {f"{e}_{'plus' if s > 0 else 'minus'}": j
                            for (e, s), (j, _h) in mapping.items()}}, f, indent=1)
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
