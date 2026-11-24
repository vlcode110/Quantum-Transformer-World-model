"""Connect to IonQ, confirm the setup works, and check it against local exact.

Run this before spending anything on remote execution. It does three things, in
increasing order of cost, and stops at the first failure:

  1. CREDENTIALS — confirms the plugin is installed and IONQ_API_KEY is set.
     The key itself is never read into a variable, printed, or logged; only its
     presence and length are reported.

  2. CONNECTIVITY — submits a two-qubit circuit to `ionq.simulator`. This is the
     cheapest possible remote job and establishes that authentication, job
     submission and result retrieval all work.

  3. AGREEMENT — runs the project's actual circuit on IonQ's ideal simulator and
     on the local exact simulator, and compares. This is the test of the claim
     that an exact simulator is an exact simulator: if IonQ's ideal backend and
     `default.qubit` agree, then the choice of simulator cannot be responsible
     for any result in this project.

     The comparison has a caveat the script reports rather than assumes: remote
     devices are shot-based, so IonQ's "ideal" simulator may still sample. If it
     does, the residual difference should scale as 1/sqrt(shots) and not as a
     systematic offset — the script prints the shot count so this is checkable.

  4. NOISE — optionally repeats against IonQ's hardware noise models, which is
     what Tier 2 of the hardware plan actually needs.

    export IONQ_API_KEY="..."        # in your shell, not in this file
    python ionq_check.py
    python ionq_check.py --noise-models aria-1 forte-1
    python ionq_check.py --skip-remote        # local halves only

Costs: steps 2-4 submit small jobs to the SIMULATOR, which is inexpensive. This
script never touches `ionq.qpu`. Running on real hardware is a separate,
deliberate decision and should follow a successful run of this one.
"""
import argparse
import os
import sys

import numpy as np

from verify_exact import reference, pennylane_circuit


def check_credentials():
    print("1. CREDENTIALS")
    try:
        import pennylane_ionq  # noqa: F401
        print("   plugin            pennylane-ionq installed")
    except ImportError:
        print("   plugin            NOT INSTALLED")
        print("\n   Install it with:  pip install pennylane-ionq")
        return False

    key = os.environ.get("IONQ_API_KEY")
    if not key:
        print("   IONQ_API_KEY      NOT SET")
        print("""
   Get a key from the IonQ console (https://cloud.ionq.com, under API keys),
   then put it in your shell profile so it persists across sessions:

       echo 'export IONQ_API_KEY="your-key-here"' >> ~/.zshrc
       source ~/.zshrc

   Do not put the key in a source file or anything that reaches version
   control. PennyLane reads this environment variable automatically; no
   code in this project ever needs to reference it.""")
        return False
    print(f"   IONQ_API_KEY      set ({len(key)} characters)")
    return True


def check_connectivity():
    print("\n2. CONNECTIVITY  (smallest possible remote job)")
    import pennylane as qml
    try:
        dev = qml.device("ionq.simulator", wires=2)

        @qml.qnode(dev)
        def bell():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        val = float(bell())
        print(f"   submitted, returned <Z_0> = {val:+.4f}  (expect ~0 for a Bell state)")
        print(f"   device shots      {getattr(dev, 'shots', 'unknown')}")
        return True
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        print("\n   Common causes: invalid or expired key, no network, or no credit"
              "\n   on the account. Check https://cloud.ionq.com before retrying.")
        return False


def compare_quiet(dev_name, inputs, weights, n, L, ent, ref, **kw):
    """Same comparison, value only — used for the shot sweep."""
    try:
        import pennylane as qml
        dev = qml.device(dev_name, wires=n + 1, **kw)
        got = pennylane_circuit(dev, inputs, weights, n, L, ent)
        return float(np.abs(got - ref).max())
    except Exception as e:
        print(f"   sweep stopped: {type(e).__name__}: {e}")
        return None


def compare(label, dev_name, inputs, weights, n, L, ent, ref, **kw):
    try:
        import pennylane as qml
        dev = qml.device(dev_name, wires=n + 1, **kw)
        got = pennylane_circuit(dev, inputs, weights, n, L, ent)
    except Exception as e:
        print(f"   {label:<34} unavailable ({type(e).__name__}: {e})")
        return None
    diff = float(np.abs(got - ref).max())
    shots = getattr(dev, "shots", None)
    print(f"   {label:<34}{diff:>12.3e}   shots={shots}")
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2,
                    help="kept small: remote jobs are billed per circuit")
    ap.add_argument("--entangler", default="brickwork",
                    choices=["brickwork", "ring_cascade"])
    ap.add_argument("--remote-shots", type=int, nargs="+",
                    default=[256, 1024, 4096],
                    help="shot counts for the agreement sweep (each is one job)")
    ap.add_argument("--noise-models", nargs="*", default=[],
                    help="e.g. aria-1 aria-2 forte-1")
    ap.add_argument("--skip-remote", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.skip_remote:
        if not check_credentials():
            sys.exit(1)
        if not check_connectivity():
            sys.exit(1)

    n, L, ent = args.n_qubits, args.n_layers, args.entangler
    rng = np.random.default_rng(args.seed)
    inputs = rng.uniform(-np.pi, np.pi, n)
    weights = rng.uniform(-np.pi, np.pi, (L, n + 1))

    print(f"\n3. AGREEMENT  ({n} data qubits, {L} layers, {ent})")
    ref = reference(inputs, weights, n, L, ent)
    print(f"   {'backend':<34}{'max |diff| vs numpy':>12}")
    compare("default.qubit (local, exact)", "default.qubit",
            inputs, weights, n, L, ent, ref)

    if not args.skip_remote:
        print("\n   IonQ's simulator is shot-based, so it cannot return the exact")
        print("   value. Sweeping the shot count separates the two explanations for")
        print("   any deviation: sampling noise falls as 1/sqrt(shots), while a real")
        print("   disagreement in the computed object would not move at all.")
        print(f"\n   {'shots':>8}{'max |diff|':>14}{'x1/sqrt(shots)':>16}")
        base = None
        for s in args.remote_shots:
            d = compare_quiet("ionq.simulator", inputs, weights, n, L, ent, ref,
                              shots=s)
            if d is None:
                break
            if base is None:
                base = (s, d)
            expect = base[1] * np.sqrt(base[0] / s)
            print(f"   {s:>8}{d:>14.3e}{expect:>16.3e}")
        print("\n   If the measured column tracks the predicted one, the deviation is")
        print("   sampling and the two simulators agree on the underlying value.")

        for nm in args.noise_models:
            print(f"\n4. NOISE MODEL: {nm}")
            compare(f"ionq.simulator ({nm})", "ionq.simulator",
                    inputs, weights, n, L, ent, ref,
                    noise_model=nm, noise_seed=args.seed)

    print("""
Reading the result
   A difference at the 1e-15 level between local exact and IonQ's ideal
   simulator means the two compute the same object, and no choice of simulator
   can change a result in this project.

   A larger difference against the ideal backend is expected if that backend is
   shot-based: check the reported shot count, and confirm the deviation falls as
   1/sqrt(shots) rather than sitting at a fixed offset. `verify_exact.py` does
   that scaling check locally and for free.

   Differences under a noise model are the quantity of interest for the hardware
   section: they are what a real device would add on top of sampling.""")


if __name__ == "__main__":
    main()
