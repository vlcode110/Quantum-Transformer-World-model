"""Render the quantum dynamics core's circuit for figures and inspection.

Draws the ACTUAL circuit the models execute (imported from models/quantum_core),
so a diagram can never drift out of sync with the code that produced the
results — a common source of error in papers.

Usage:
    python draw_circuit.py                      # defaults from config.py
    python draw_circuit.py --n_qubits 4 --q_layers 2      # readable for figures
    python draw_circuit.py --entanglers brickwork ring_cascade --format pdf
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
import torch

from config import Config
from models.quantum_core import build_qnode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_qubits", type=int, default=None)
    ap.add_argument("--q_layers", type=int, default=None)
    ap.add_argument("--encoding", default=None)
    ap.add_argument("--readout", default=None)
    ap.add_argument("--entanglers", nargs="+",
                    default=["brickwork", "ring_cascade"])
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    cfg = Config()
    nq = args.n_qubits or cfg.n_qubits
    nl = args.q_layers or cfg.q_layers
    enc = args.encoding or cfg.encoding
    ro = args.readout or cfg.readout

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    x = torch.linspace(-1, 1, nq)          # placeholder angles
    w = torch.zeros(nl, nq + 1)            # placeholder weights

    for ent in args.entanglers:
        circuit, wshapes, n_out = build_qnode(nq, nl, enc, ro, ent)
        title = (f"{ent} | {nq}+1 qubits | {nl} layers | {enc} | {ro} readout "
                 f"| {nl * (nq + 1)} trainable params")

        # --- text form: handy in a terminal and for a code appendix
        txt = qml.draw(circuit, max_length=200)(x, w)
        txt_path = os.path.join(outdir, f"circuit_{ent}_{nq}q_L{nl}.txt")
        with open(txt_path, "w") as f:
            f.write(title + "\n\n" + txt + "\n")
        print(f"\n=== {title} ===")
        print(txt)

        # --- figure form: for the thesis
        fig, ax = qml.draw_mpl(circuit, style="pennylane",
                               decimals=None)(x, w)
        ax.set_title(title, fontsize=9)
        fig_path = os.path.join(outdir,
                                f"circuit_{ent}_{nq}q_L{nl}.{args.format}")
        fig.savefig(fig_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"[draw] {fig_path}")
        print(f"[draw] {txt_path}")

    print("\nTip: for a thesis figure use a small circuit (--n_qubits 4 "
          "--q_layers 2). The full 6-qubit, 4-layer circuit is faithful but "
          "too wide to read at page width; state the real dimensions in the "
          "caption instead.")


if __name__ == "__main__":
    main()
