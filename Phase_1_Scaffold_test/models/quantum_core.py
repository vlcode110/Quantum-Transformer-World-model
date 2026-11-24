"""Quantum transformer dynamics core.

A classical transformer over (z, a) history in which ONE encoder layer routes
each token through a parameterized quantum circuit — the QASA finding is that
position matters more than count (arXiv:2504.05336). The circuit uses angle
encoding with data re-uploading and local <Z_i> readout, both configurable.

Every design axis named in the Phase 1 ablation list is a constructor argument:
encoding, readout locality, qubit count, depth, layer position, and how the
action enters the model.
"""
import math

import pennylane as qml
import torch
import torch.nn as nn


# --------------------------------------------------------------- the circuit
def entangle(wires, pattern):
    """Entangling block. MEASURED (Phase 0, 2026-07-20, 100 random inits):

        pattern        readout   Var[grad] decay per added qubit
        ring_cascade   local     x0.49-0.52   exponential  <- plateau
        ring_cascade   global    x0.45-0.49   exponential
        brickwork      local     x0.90-0.99   FLAT         <- use this
        brickwork      global    x0.41-0.49   exponential

    'ring_cascade' applies CNOT(0,1), CNOT(1,2), ... SEQUENTIALLY, so
    correlations traverse the whole register inside a single nominal layer:
    circuit depth ~n, not 1. It scrambles fast enough that a local observable
    behaves globally, which is why its local and global curves coincide.

    'brickwork' pairs even wires then odd wires — genuinely depth 2 at any
    width — so the light cone grows with layer count as intended and local
    observables keep their mild scaling (Cerezo et al. 2021).
    """
    if pattern == "brickwork":
        for w in range(0, wires - 1, 2):
            qml.CNOT(wires=[w, w + 1])
        for w in range(1, wires - 1, 2):
            qml.CNOT(wires=[w, w + 1])
    elif pattern == "ring_cascade":
        for w in range(wires):
            qml.CNOT(wires=[w, (w + 1) % wires])
    elif pattern == "none":
        # No two-qubit gates at all: the circuit becomes a product of
        # independent single-qubit rotations, so it can produce no entanglement
        # whatever. This is the control for "is the quantumness load-bearing?" —
        # Bowles et al. (2024) report that removing entanglement frequently
        # leaves performance unchanged, and an architecture paper should
        # establish that for its own construction rather than leave it to a
        # referee. The parameter count is unaffected, so the comparison is
        # otherwise exact.
        pass
    else:
        raise ValueError(f"unknown entangler {pattern!r}")


def build_qnode(n_qubits, n_layers, encoding, readout, entangler="brickwork"):
    """Returns (qnode, weight_shapes, n_out).

    encoding: 'angle'          — encode once, then trainable layers
              'angle_reupload' — re-encode every layer (richer frequency
                                 spectrum, linear depth cost)
              'amplitude'      — amplitude encoding (log qubits, deep prep;
                                 known to lose to angle under realistic noise —
                                 kept as an ablation arm, not a default)
    readout:  'local'  — <Z_i> on each data wire (mild gradient decay)
              'global' — one <Z_0...Z_n> (plateaus fast; plateau control arm)
    """
    wires = n_qubits + 1                      # data wires + 1 ancilla
    dev = qml.device("default.qubit", wires=wires)

    def _encode(inputs):
        if encoding == "amplitude":
            qml.AmplitudeEmbedding(inputs, wires=range(wires),
                                   normalize=True, pad_with=0.0)
        else:
            for i in range(n_qubits):
                qml.RX(inputs[..., i], wires=i)
                qml.RZ(inputs[..., i], wires=i)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        _encode(inputs)
        for l in range(n_layers):
            if encoding == "angle_reupload" and l > 0:
                _encode(inputs)               # data re-uploading
            for w in range(wires):
                qml.RY(weights[l, w], wires=w)
            entangle(wires, entangler)
        if readout == "global":
            return [qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(wires)]))]
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    n_out = 1 if readout == "global" else n_qubits
    return circuit, {"weights": (n_layers, wires)}, n_out


class QuantumTokenLayer(nn.Module):
    """Per-token: project to rotation angles -> PQC -> project back to d_model.

    action_conditioning='circuit' additionally feeds the action into the
    circuit's rotation angles, so the control signal enters the quantum map
    directly rather than only through the surrounding classical layers. It is
    implemented here but was never used in any reported experiment — every
    result in either paper uses 'token' or 'none'. Keep it off for the baseline
    so the comparison is against the conventional design."""

    def __init__(self, d_model, n_qubits=8, n_layers=4, encoding="angle_reupload",
                 readout="local", action_dim=0, action_conditioning="token",
                 entangler="brickwork"):
        super().__init__()
        self.action_in_circuit = (action_conditioning == "circuit")
        in_dim = d_model + (action_dim if self.action_in_circuit else 0)
        self.down = nn.Linear(in_dim, n_qubits)
        self.circuit, wshapes, n_out = build_qnode(
            n_qubits, n_layers, encoding, readout, entangler)
        self.q = qml.qnn.TorchLayer(self.circuit, wshapes)
        self.up = nn.Linear(n_out, d_model)
        self.n_quantum_params = n_layers * (n_qubits + 1)

    def forward(self, x, actions=None):        # x: (B, T, d)
        B, T, _ = x.shape
        if self.action_in_circuit and actions is not None:
            x = torch.cat([x, actions], dim=-1)
        ang = torch.tanh(self.down(x)) * math.pi
        out = self.q(ang.reshape(B * T, -1))
        return self.up(out.reshape(B, T, -1).float())


# ------------------------------------------------------------------- encoder
class EncoderLayer(nn.Module):
    """Standard block; `token_layer` optionally inserted after attention."""

    def __init__(self, d_model, n_heads, ffn=64, token_layer=None, drop=0.1):
        super().__init__()
        # MultiheadAttention requires d_model % n_heads == 0. Capacity sweeps
        # naturally hit widths that violate this (d_model=3 with 2 heads), so
        # fall back to the largest head count that divides d_model rather than
        # crashing mid-sweep and losing the queued runs.
        while n_heads > 1 and d_model % n_heads != 0:
            n_heads -= 1
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=drop,
                                          batch_first=True)
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.token_layer = token_layer
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(),
                                 nn.Linear(ffn, d_model))

    def forward(self, x, actions=None, causal=True):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool),
                          1) if causal else None
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        h = self.ln1(x + a)
        if self.token_layer is not None:
            h = self.ln2(h + self.token_layer(h, actions))
        return self.ln3(h + self.ffn(h))


def positional_encoding(T, d, device):
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    i = torch.arange(0, d, 2, device=device).float()
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class DynamicsCore(nn.Module):
    """(z_{t-K+1..t}, a_{t-K+1..t}) -> predicted z_{t+1}.

    quantum=False gives the parameter-matched classical twin: identical in every
    respect except the PQC is replaced by a small tanh MLP. That twin is the
    baseline the Phase 1 gate is measured against, so keep the rest identical.
    """

    def __init__(self, z_dim, action_dim, cfg, quantum=True):
        super().__init__()
        d = cfg.d_model
        self.action_conditioning = cfg.action_conditioning
        self.quantum = quantum

        in_dim = z_dim + (action_dim if cfg.action_conditioning == "concat" else 0)
        self.z_in = nn.Linear(in_dim, d)
        # DECISION (config.action_conditioning): 'token' inserts actions as
        # their own sequence positions; 'concat' glues them to each latent;
        # 'circuit' pushes them into the PQC angles (never run; see README).
        self.a_in = nn.Linear(action_dim, d) if cfg.action_conditioning == "token" else None
        self.ln = nn.LayerNorm(d)

        if quantum:
            token = QuantumTokenLayer(
                d, cfg.n_qubits, cfg.q_layers, cfg.encoding, cfg.readout,
                action_dim, cfg.action_conditioning,
                getattr(cfg, "entangler", "brickwork"))
            self.n_quantum_params = token.n_quantum_params
        else:
            token = _ClassicalToken(d, cfg.n_qubits)
            self.n_quantum_params = 0

        pos = cfg.quantum_layer_position % cfg.n_layers
        self.layers = nn.ModuleList([
            EncoderLayer(d, cfg.n_heads, token_layer=(token if i == pos else None))
            for i in range(cfg.n_layers)])
        self.head = nn.Linear(d, z_dim)

    def forward(self, z, a):
        """z: (B, K, z_dim), a: (B, K, action_dim) -> (B, z_dim)"""
        if self.action_conditioning == "concat":
            h = self.z_in(torch.cat([z, a], -1))
        else:
            h = self.z_in(z)
        if self.a_in is not None:                      # interleave action tokens
            h = torch.stack([h, self.a_in(a)], dim=2).flatten(1, 2)
        h = self.ln(h)
        h = h + positional_encoding(h.shape[1], h.shape[2], h.device)
        for layer in self.layers:
            h = layer(h, a)
        return self.head(h[:, -1])                     # predict from last token


class _ClassicalToken(nn.Module):
    """Parameter-comparable stand-in for the PQC (see DynamicsCore.quantum)."""

    def __init__(self, d_model, width, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, width), nn.Tanh(),
            nn.Linear(width, hidden), nn.Tanh(),
            nn.Linear(hidden, d_model))

    def forward(self, x, actions=None):
        return self.net(x)
