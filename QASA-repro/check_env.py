"""What is installed here, and what will work?

This project has lost real time to version drift three separate times: NumPy 2
against an older PyTorch, autoray 0.7 against PennyLane, and a lightning build
mismatched to its PennyLane. Each surfaced as an error far from its cause. This
script reports the versions and checks the specific combinations that have
broken before, so a bad environment is visible in two seconds rather than forty
lines into a stack trace.

    python check_env.py

Exit status is 0 if nothing known-bad is present, 1 otherwise.
"""
import importlib
import sys

PKGS = ["numpy", "torch", "pennylane", "pennylane_lightning", "pennylane_ionq",
        "autoray", "scipy", "matplotlib"]


def version(name):
    """Version of an installed package.

    `__version__` is not universal — pennylane-lightning 0.45 has no such
    attribute — so fall back to the distribution metadata, which is what pip
    actually recorded. Reporting "unknown" here previously produced a false
    mismatch warning against PennyLane.
    """
    try:
        importlib.import_module(name)
    except ImportError:
        return None
    m = sys.modules[name]
    v = getattr(m, "__version__", None)
    if v:
        return v
    from importlib import metadata
    for dist in (name, name.replace("_", "-")):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return "installed, version unknown"


def parse(v):
    out = []
    for part in str(v).split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


def main():
    py = sys.version_info
    print(f"python            {py.major}.{py.minor}.{py.micro}  ({sys.prefix})")
    vers = {}
    for p in PKGS:
        v = version(p)
        vers[p] = v
        print(f"{p:<18}{v if v else '-'}")

    problems, notes = [], []

    # Python ceiling: PennyLane dropped 3.9 after 0.38, so pip silently
    # backtracks there and takes pennylane-ionq down with it.
    if (py.major, py.minor) < (3, 10):
        notes.append(
            f"Python {py.major}.{py.minor} caps PennyLane at 0.38 and "
            f"pennylane-ionq at 0.41.\n     IonQ noise models need a newer "
            f"plugin, so they are unavailable here.\n     Fix: build the "
            f"environment on Python 3.11.")

    pl, ar = vers.get("pennylane"), vers.get("autoray")
    if pl and ar and parse(pl) < (0, 39, 0) and parse(ar) >= (0, 7, 0):
        problems.append(
            f"autoray {ar} with PennyLane {pl} raises\n     "
            f"\"module 'autoray.autoray' has no attribute 'NumpyMimic'\".\n     "
            f"Fix: pip install 'autoray<0.7'  (or move to Python 3.11 and "
            f"upgrade PennyLane).")

    np_, tv = vers.get("numpy"), vers.get("torch")
    if np_ and tv and parse(np_) >= (2, 0, 0) and parse(tv) < (2, 3, 0):
        problems.append(
            f"NumPy {np_} with torch {tv} raises \"Numpy is not available\".\n"
            f"     Fix: pip install 'numpy<2'")

    pll = vers.get("pennylane_lightning")
    if pl and pll and "unknown" not in str(pll) and parse(pl)[:2] != parse(pll)[:2]:
        problems.append(
            f"pennylane {pl} and pennylane-lightning {pll} differ in minor "
            f"version.\n     This has segfaulted lightning.qubit before. Fix: "
            f"match them.")

    ionq = vers.get("pennylane_ionq")
    if ionq and parse(ionq) < (0, 42, 0):
        notes.append(
            f"pennylane-ionq {ionq} has no `noise_model` argument; "
            f"`--noise-model` will\n     fail with a TypeError. Needs a newer "
            f"plugin, which needs Python >= 3.10.")

    print()
    for n in notes:
        print(f"  NOTE: {n}")
    for p in problems:
        print(f"  BROKEN: {p}")
    if not problems and not notes:
        print("  No known-bad combinations. Noise models available.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
