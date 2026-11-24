"""Baselines for the Phase 1 comparison.

  classical_transformer  identical to the quantum core with the PQC swapped for
                         a parameter-comparable MLP  (the gate baseline)
  qlstm                  quantum LSTM (Chen et al. 2020, arXiv:2009.01783):
                         recurrent quantum dynamics — the "is attention doing
                         the work, or just quantumness?" control
  pqc_mlp                PQC over the flattened window, no attention at all —
                         isolates the contribution of attention itself

Keeping all three matters for the Phase 1 story: if the quantum transformer
beats the classical twin but not the PQC-MLP, the win is about quantum feature
maps, not quantum attention.
"""
import pennylane as qml
import torch
import torch.nn as nn

from .quantum_core import build_qnode


class QLSTMCell(nn.Module):
    """LSTM whose four gates are variational circuits sharing one ansatz shape.

    Follows the QLSTM formulation: classical concat of [h, x] is projected to
    rotation angles, each gate has its own PQC, outputs are projected back."""

    def __init__(self, in_dim, hidden, n_qubits=8, n_layers=4,
                 encoding="angle_reupload", readout="local",
                 entangler="brickwork"):
        super().__init__()
        self.hidden = hidden
        self.down = nn.Linear(in_dim + hidden, n_qubits)
        circuit, wshapes, n_out = build_qnode(n_qubits, n_layers, encoding,
                                              readout, entangler)
        # four gates: forget, input, update, output
        self.gates = nn.ModuleList(
            [qml.qnn.TorchLayer(circuit, wshapes) for _ in range(4)])
        self.up = nn.ModuleList([nn.Linear(n_out, hidden) for _ in range(4)])
        self.n_quantum_params = 4 * n_layers * (n_qubits + 1)

    def forward(self, x, state):
        h, c = state
        ang = torch.tanh(self.down(torch.cat([x, h], -1))) * torch.pi
        g = [up(gate(ang).float()) for gate, up in zip(self.gates, self.up)]
        f, i, u, o = torch.sigmoid(g[0]), torch.sigmoid(g[1]), \
            torch.tanh(g[2]), torch.sigmoid(g[3])
        c = f * c + i * u
        h = o * torch.tanh(c)
        return h, (h, c)


class QLSTMDynamics(nn.Module):
    """Recurrent quantum baseline with the same interface as DynamicsCore."""

    def __init__(self, z_dim, action_dim, cfg):
        super().__init__()
        self.hidden = cfg.d_model
        self.cell = QLSTMCell(z_dim + action_dim, cfg.d_model, cfg.n_qubits,
                              cfg.q_layers, cfg.encoding, cfg.readout,
                              getattr(cfg, "entangler", "brickwork"))
        self.head = nn.Linear(cfg.d_model, z_dim)
        self.n_quantum_params = self.cell.n_quantum_params

    def forward(self, z, a):
        B, K, _ = z.shape
        h = torch.zeros(B, self.hidden, device=z.device)
        state = (h, torch.zeros_like(h))
        for t in range(K):
            h, state = self.cell(torch.cat([z[:, t], a[:, t]], -1), state)
        return self.head(h)


class PQCMLP(nn.Module):
    """No attention: flatten the window, one PQC, project to the next latent.

    If this matches the quantum transformer, attention is not earning its keep —
    an important negative result to be able to report."""

    def __init__(self, z_dim, action_dim, cfg):
        super().__init__()
        flat = cfg.context * (z_dim + action_dim)
        self.down = nn.Linear(flat, cfg.n_qubits)
        circuit, wshapes, n_out = build_qnode(
            cfg.n_qubits, cfg.q_layers, cfg.encoding, cfg.readout,
            getattr(cfg, "entangler", "brickwork"))
        self.q = qml.qnn.TorchLayer(circuit, wshapes)
        self.up = nn.Sequential(nn.Linear(n_out, cfg.d_model), nn.SiLU(),
                                nn.Linear(cfg.d_model, z_dim))
        self.n_quantum_params = cfg.q_layers * (cfg.n_qubits + 1)

    def forward(self, z, a):
        x = torch.cat([z, a], -1).flatten(1)
        ang = torch.tanh(self.down(x)) * torch.pi
        return self.up(self.q(ang).float())
