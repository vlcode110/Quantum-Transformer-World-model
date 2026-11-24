"""QASA: Quantum Adaptive Self-Attention (arXiv:2504.05336) in PennyLane + PyTorch.

Architecture per the paper (Sec. 3):
  input -> linear proj + LayerNorm -> sinusoidal positional encoding
        -> (N-1) classical transformer encoder layers (MHA, residual, GELU FFN)
        -> final QUANTUM encoder layer:
             a  = MultiHeadAttention(h)
             h' = LayerNorm(h + a)
             per token: h_q = tanh(W_q h') in R^n   (n = data qubits)
             PQC on n+1 qubits, L_q layers, data re-uploading with RX/RZ,
             entanglement with CNOT + trainable RY; readout <Z_i>
             project back to d, residual + LayerNorm, FFN
        -> pooling -> linear head (forecast next value)

The classical baseline is identical except the quantum layer is replaced by a
parameter-matched classical MLP (tanh bottleneck of the same width).
"""
import math

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


# ---------------------------------------------------------------- quantum layer
def make_qlayer(n_qubits, n_layers, entangler="ring_cascade"):
    """PQC on n_qubits+1 wires. Data re-uploading RX/RZ on data wires each
    layer; trainable RY on all wires + entanglement. <Z> readout on data
    wires. Trainable params: n_layers * (n_qubits + 1).

    entangler:
      'ring_cascade' — CNOTs applied in series around the register. This is the
          literal reading of "CNOT ring" used in the paper and in most of this
          literature. Note it is NOT constant depth: correlations traverse the
          whole register inside one nominal layer. KEEP AS DEFAULT so the
          reproduction stays faithful to what the paper describes.
      'brickwork' — even pairs then odd pairs; genuinely depth 2 at any width.
          Phase 0 measured ~1300x larger local gradients at 10 qubits under
          this pattern. Selecting it turns this script into an
          entangler-sensitivity check on the reproduced result.
    """
    wires = n_qubits + 1
    dev = qml.device("default.qubit", wires=wires)

    def entangle():
        if entangler == "brickwork":
            for w in range(0, wires - 1, 2):
                qml.CNOT(wires=[w, w + 1])
            for w in range(1, wires - 1, 2):
                qml.CNOT(wires=[w, w + 1])
        elif entangler == "ring_cascade":
            for w in range(wires):
                qml.CNOT(wires=[w, (w + 1) % wires])
        else:
            raise ValueError(f"unknown entangler {entangler!r}")

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for l in range(n_layers):
            for i in range(n_qubits):            # data re-uploading
                qml.RX(inputs[..., i], wires=i)
                qml.RZ(inputs[..., i], wires=i)
            for w in range(wires):               # trainable rotations
                qml.RY(weights[l, w], wires=w)
            entangle()
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, wires)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumTokenLayer(nn.Module):
    """tanh(W_q h') -> PQC -> project back to d."""

    def __init__(self, d_model, n_qubits=4, n_layers=6, entangler="ring_cascade"):
        super().__init__()
        self.down = nn.Linear(d_model, n_qubits)
        self.q = make_qlayer(n_qubits, n_layers, entangler)
        self.up = nn.Linear(n_qubits, d_model)
        self.n_quantum_params = n_layers * (n_qubits + 1)

    def forward(self, x):                        # x: (B, T, d)
        B, T, d = x.shape
        hq = torch.tanh(self.down(x)) * math.pi  # angles in [-pi, pi]
        out = self.q(hq.reshape(B * T, -1))      # (B*T, n_qubits)
        return self.up(out.reshape(B, T, -1).float())


class ClassicalTokenLayer(nn.Module):
    """Parameter-comparable classical stand-in for the quantum layer."""

    def __init__(self, d_model, n_qubits=4, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, n_qubits), nn.Tanh(),
            nn.Linear(n_qubits, hidden), nn.Tanh(),
            nn.Linear(hidden, d_model))

    def forward(self, x):
        return self.net(x)


class FourierTokenLayer(nn.Module):
    """Classical surrogate for the PQC, built from what the PQC can represent.

    Schuld, Sweke and Meyer (arXiv:2008.08605) show that a data re-uploading
    circuit computes a partial Fourier series in its inputs: the accessible
    FREQUENCIES are fixed by the encoding gates, and the trainable parameters
    only set the COEFFICIENTS. The circuit in `make_qlayer` encodes each input
    dimension through RX and RZ once per layer, so with L layers each dimension
    passes through 2L encoding gates and its accessible frequencies are integers
    bounded by that count.

    This layer parameterises that same function class directly and classically:
    draw integer frequency vectors from the circuit's spectrum, hold them FIXED
    exactly as the circuit does, and learn the coefficients. It is a drop-in
    replacement for QuantumTokenLayer with identical down- and up-projections,
    so the only thing that differs is how the middle map is computed.

    The point is not to win. It is that `n_freq` can be swept to find the
    classical parameter count at which the surrogate matches the circuit. If a
    comparable count suffices, the circuit is one parameterisation of a
    classical feature map rather than a distinct function class; if the
    surrogate needs far more, the circuit's constrained parameterisation is
    carrying a useful inductive bias, which is the only route to an advantage at
    simulable scale.
    """

    def __init__(self, d_model, n_qubits=4, n_freq=4, max_freq=None,
                 q_layers=6, seed=0):
        super().__init__()
        # 2 encoding gates (RX, RZ) per layer bound the integer spectrum
        if max_freq is None:
            max_freq = 2 * q_layers
        self.down = nn.Linear(d_model, n_qubits)
        g = torch.Generator().manual_seed(seed)
        omega = torch.randint(-max_freq, max_freq + 1, (n_freq, n_qubits),
                              generator=g).float()
        self.register_buffer("omega", omega)        # fixed, as in the circuit
        self.coef = nn.Parameter(torch.randn(n_qubits, 2 * n_freq,
                                             generator=g) * 0.1)
        self.up = nn.Linear(n_qubits, d_model)
        self.n_freq, self.max_freq = n_freq, max_freq
        self.n_core_params = self.coef.numel()

    def forward(self, x):                            # x: (B, T, d)
        z = torch.tanh(self.down(x)) * math.pi       # same angle range as PQC
        phase = z @ self.omega.T                     # (B, T, n_freq)
        feat = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        return self.up(feat @ self.coef.T)


# ---------------------------------------------------------------- transformer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn=64, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=drop,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(),
                                 nn.Linear(ffn, d_model))

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + a)
        return self.ln2(x + self.ffn(x))


class QuantumEncoderLayer(nn.Module):
    """Final encoder block: MHA -> LayerNorm -> per-token quantum layer
    (residual) -> FFN (residual)."""

    def __init__(self, d_model, n_heads, token_layer, ffn=64, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=drop,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.token_layer = token_layer
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(),
                                 nn.Linear(ffn, d_model))

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        h = self.ln1(x + a)
        h = self.ln2(h + self.token_layer(h))
        return self.ln3(h + self.ffn(h))


def positional_encoding(T, d, device):
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    i = torch.arange(0, d, 2, device=device).float()
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Forecaster(nn.Module):
    """Sequence-to-one forecaster. mode: 'quantum' (QASA) or 'classical'."""

    def __init__(self, mode="quantum", d_model=32, n_heads=2, n_classical=1,
                 n_qubits=4, q_layers=6, entangler="ring_cascade", n_freq=4,
                 seed=0):
        super().__init__()
        self.inp = nn.Linear(1, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads) for _ in range(n_classical)])
        if mode == "quantum":
            token = QuantumTokenLayer(d_model, n_qubits, q_layers, entangler)
            self.n_quantum_params = token.n_quantum_params
        elif mode == "fourier":
            token = FourierTokenLayer(d_model, n_qubits, n_freq=n_freq,
                                      q_layers=q_layers, seed=seed)
            self.n_quantum_params = 0
        elif mode == "classical":
            token = ClassicalTokenLayer(d_model, n_qubits)
            self.n_quantum_params = 0
        else:
            raise ValueError(f"unknown mode {mode!r}")
        self.token_layer = token
        self.final = QuantumEncoderLayer(d_model, n_heads, token)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                        # x: (B, T) univariate window
        h = self.ln(self.inp(x.unsqueeze(-1)))
        h = h + positional_encoding(h.shape[1], h.shape[2], h.device)
        for layer in self.layers:
            h = layer(h)
        h = self.final(h)
        return self.head(h.mean(dim=1)).squeeze(-1)  # mean-pool -> next value
