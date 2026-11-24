"""The quantum substitution: a parameterized circuit as DreamerV3's prior.

DreamerV3's RSSM maintains a deterministic recurrent state and a stochastic
latent. The PRIOR network maps the deterministic state to a distribution over
the next stochastic latent — that is, it encodes the model's belief about what
happens next, without seeing the next observation. It is the component that
does the world-modelling, and the one imagination rollouts depend on entirely.

Replacing exactly that network with a circuit is the minimal substitution that
still yields a quantum world model driving an agent.

Nothing in the Phase 0 DreamerV3 project is modified. The swap happens on an
already-constructed agent, and the world-model optimiser is rebuilt afterwards
so it tracks the new parameters.
"""
import importlib.util
import math
import os
import sys

import pennylane as qml
import torch
import torch.nn as nn


# --------------------------------------------------------------------- import
def _load_module_from_file(path, module_name):
    """Load a module by explicit path under a unique name.

    Both this project and the Phase 1 project define `config.py`, so a plain
    import after a sys.path tweak would return the wrong module. Loading the
    circuit builder by file path sidesteps the name entirely.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_core_module(phase1_root):
    """The whole Phase 1 model module, for DynamicsCore as well as build_qnode.

    Loaded by file path for the same reason: both projects define config.py,
    and the transformer used here must be the identical object Phase 1
    evaluates offline, not a copy that can drift from it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, phase1_root, "models", "quantum_core.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Phase 1 quantum core not found at {path}. "
            f"Fix `phase1_root` in config.py.")
    return _load_module_from_file(os.path.abspath(path),
                                  "_phase1_quantum_core")


def load_build_qnode(phase1_root):
    """Reuse Phase 1's circuit builder rather than duplicating it, so the two
    phases can never drift apart in circuit definition."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, phase1_root, "models", "quantum_core.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Phase 1 quantum core not found at {path}. "
            f"Fix `phase1_root` in config.py.")
    mod = _load_module_from_file(os.path.abspath(path), "_phase1_quantum_core")
    return mod.build_qnode


# ----------------------------------------------------------------- the module
class QuantumPrior(nn.Module):
    """deterministic state -> circuit -> logits over the next stochastic latent.

    Interface-compatible with the MLP it replaces: same input width, same
    output width, so RSSM.img_step and RSSM.obs_step work unchanged.
    """

    def __init__(self, deter_dim, out_dim, build_qnode, n_qubits=6, n_layers=4,
                 encoding="angle_reupload", readout="local",
                 entangler="brickwork", hidden=64):
        super().__init__()
        self.down = nn.Linear(deter_dim, n_qubits)
        circuit, wshapes, n_out = build_qnode(
            n_qubits, n_layers, encoding, readout, entangler)
        self.q = qml.qnn.TorchLayer(circuit, wshapes)
        # A small classical head expands the n_qubits readout values to the
        # full categorical logit vector. This is unavoidable: the latent has
        # stoch*classes dimensions (256 by default) and the circuit emits
        # n_qubits. The head is deliberately narrow so the circuit, not the
        # head, carries the prediction.
        self.up = nn.Sequential(nn.Linear(n_out, hidden), nn.SiLU(),
                                nn.Linear(hidden, out_dim))
        self.n_quantum_params = n_layers * (n_qubits + 1)

    def forward(self, deter):
        ang = torch.tanh(self.down(deter)) * math.pi
        return self.up(self.q(ang).float())


class TransformerPrior(nn.Module):
    """A quantum TRANSFORMER as the RSSM prior.

    The original Phase 2 prior saw a single recurrent state and had no
    attention: down-projection, circuit, expansion head. That is a quantum
    world model but not a quantum *transformer* world model, which is what
    this project set out to build.

    This prior instead reads the last K deterministic states and attends over
    them, using Phase 1's DynamicsCore unchanged — multi-head self-attention
    with the parameterised circuit as the token-mixing block of one encoder
    layer. The same module evaluated offline in Phase 1 therefore drives the
    agent here, so the two phases cannot diverge in what "the quantum
    transformer" means.

    History is maintained internally because the RSSM calls the prior as
    `prior_net(deter)` with one state. Rather than rewriting img_step and
    obs_step, the module keeps a rolling window and exposes reset hooks that
    `swap_prior` wires into the RSSM. The buffer is state, not parameters, so
    it must be reset at every sequence boundary; getting that wrong would
    leak one episode's history into the next, silently.

    quantum=False gives the identical architecture with the circuit replaced
    by the parameter-matched classical token layer, so the comparison
    isolates the circuit rather than the attention.
    """

    def __init__(self, deter_dim, out_dim, core_module, cfg, quantum=True):
        super().__init__()
        shim = _CoreConfig(cfg)
        self.core = core_module.DynamicsCore(
            deter_dim, 1, shim, quantum=quantum)
        # DynamicsCore predicts a vector of z_dim; the prior needs categorical
        # logits, so its output head is replaced rather than stacked on top.
        self.core.head = nn.Linear(shim.d_model, out_dim)
        self.n_quantum_params = self.core.n_quantum_params
        self.K = cfg.prior_context
        self.deter_dim = deter_dim
        self._buf = None                       # (B, K, deter_dim), not a param

    # ------------------------------------------------------------- history
    def reset(self):
        """Drop all history. Called at the start of a sequence or rollout."""
        self._buf = None

    def reset_rows(self, mask):
        """Clear history for the batch rows where `mask` is true.

        obs_step receives is_first and zeroes the recurrent state for those
        rows; the window must be cleared to match, or the first step of a new
        episode attends over the previous one.
        """
        if self._buf is None:
            return
        m = mask.reshape(-1).to(dtype=torch.bool, device=self._buf.device)
        if m.shape[0] == self._buf.shape[0] and m.any():
            self._buf = self._buf.clone()
            self._buf[m] = 0.0

    def forward(self, deter):
        if self._buf is None or self._buf.shape[0] != deter.shape[0]:
            # start of a sequence, or the batch shape changed between the
            # observe pass and imagination: begin a fresh window
            self._buf = torch.zeros(deter.shape[0], self.K, self.deter_dim,
                                    device=deter.device, dtype=deter.dtype)
        self._buf = torch.cat([self._buf[:, 1:], deter.unsqueeze(1)], dim=1)
        a = torch.zeros(deter.shape[0], self.K, 1,
                        device=deter.device, dtype=deter.dtype)
        return self.core(self._buf, a)


class _CoreConfig:
    """Adapter exposing Phase 2's settings under the names DynamicsCore wants."""

    def __init__(self, cfg):
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_layers = cfg.n_layers
        self.quantum_layer_position = cfg.quantum_layer_position
        self.action_conditioning = cfg.action_conditioning
        self.n_qubits = cfg.n_qubits
        self.q_layers = cfg.q_layers
        self.encoding = cfg.encoding
        self.readout = cfg.readout
        self.entangler = cfg.entangler


class ClassicalPriorControl(nn.Module):
    """Parameter-comparable classical stand-in.

    Used when `quantum=False` so the control differs from the quantum agent
    ONLY in the substituted component. Note this is not DreamerV3's original
    prior network — it mirrors the quantum module's shape (narrow bottleneck,
    then expansion) so the comparison isolates the circuit rather than the
    surrounding architecture.
    """

    def __init__(self, deter_dim, out_dim, n_qubits=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deter_dim, n_qubits), nn.Tanh(),
            nn.Linear(n_qubits, n_qubits), nn.Tanh(),
            nn.Linear(n_qubits, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim))
        self.n_quantum_params = 0

    def forward(self, deter):
        return self.net(deter)


# ------------------------------------------------------------------ the swap
def swap_prior(agent, cfg, dcfg, build_qnode=None):
    """Replace the RSSM prior in a constructed Dreamer, in place.

    Returns (agent, info). The world-model optimiser is rebuilt because the
    original was created over the parameters of the module that has just been
    replaced; without this the new parameters would never be updated and the
    old ones would be optimised into a module no longer in the graph — a
    failure that trains quietly and produces nonsense.
    """
    rssm = agent.wm.rssm
    out_dim = rssm.stoch * rssm.classes
    arch = getattr(cfg, "prior_arch", "mlp")

    if arch == "transformer":
        # Both arms share the transformer; only the token-mixing block differs,
        # so the comparison isolates the circuit rather than the attention.
        core = load_core_module(cfg.phase1_root)
        new_prior = TransformerPrior(rssm.deter, out_dim, core, cfg,
                                     quantum=cfg.quantum)
    elif cfg.quantum:
        if build_qnode is None:
            build_qnode = load_build_qnode(cfg.phase1_root)
        new_prior = QuantumPrior(
            rssm.deter, out_dim, build_qnode,
            n_qubits=cfg.n_qubits, n_layers=cfg.q_layers,
            encoding=cfg.encoding, readout=cfg.readout,
            entangler=cfg.entangler, hidden=cfg.prior_hidden)
    else:
        new_prior = ClassicalPriorControl(
            rssm.deter, out_dim, n_qubits=cfg.n_qubits, hidden=cfg.prior_hidden)

    old_params = sum(p.numel() for p in rssm.prior_net.parameters())
    agent.wm.rssm.prior_net = new_prior.to(cfg.device)
    if isinstance(new_prior, TransformerPrior):
        _install_history_hooks(agent, new_prior)
    agent.wm.opt = torch.optim.Adam(agent.wm.parameters(),
                                    lr=dcfg["model_lr"], eps=1e-8)

    info = {
        "prior": "quantum" if cfg.quantum else "classical_control",
        "prior_params": sum(p.numel() for p in new_prior.parameters()),
        "prior_params_replaced": old_params,
        "quantum_params": new_prior.n_quantum_params,
        "total_params": sum(p.numel() for p in agent.parameters()
                            if p.requires_grad),
    }
    return agent, info


def _install_history_hooks(agent, prior):
    """Keep the prior's attention window in step with the RSSM's sequences.

    The RSSM calls `prior_net(deter)` with a single state, so the window lives
    inside the prior. That makes it mutable state on a module, and mutable
    state leaks unless every boundary resets it. Three boundaries matter:

      observe()   starts a fresh training sequence
      imagine()   starts a rollout, with a different batch shape
      is_first    an episode boundary inside a sequence, where obs_step
                  already zeroes the recurrent state

    These wrappers are installed on the instance rather than edited into
    DreamerV3, so Phase 0 remains untouched and the behaviour is visible in
    one place. A leak here would not raise: it would quietly let one episode
    attend over the previous one, so each boundary is handled explicitly.
    """
    rssm, ac = agent.wm.rssm, agent.ac

    if getattr(rssm, "_qtwm_hooked", False):
        return
    rssm._qtwm_hooked = True

    orig_observe = rssm.observe
    orig_obs_step = rssm.obs_step
    orig_imagine = ac.imagine

    def observe(embeds, actions, is_firsts, state=None):
        prior.reset()
        return orig_observe(embeds, actions, is_firsts, state)

    def obs_step(deter, stoch, action, embed, is_first):
        prior.reset_rows(is_first)
        return orig_obs_step(deter, stoch, action, embed, is_first)

    def imagine(wm, start_deter, start_stoch):
        prior.reset()
        return orig_imagine(wm, start_deter, start_stoch)

    rssm.observe, rssm.obs_step, ac.imagine = observe, obs_step, imagine


# ------------------------------------------------- gradient instrumentation
def prior_grad_stats(agent):
    """Gradient health of the substituted component, continuous with the
    Phase 0 and Phase 1 measurements.

    Logged: variance and magnitude on the circuit's trainable rotations, the
    same for the surrounding classical parameters as a control, and their
    ratio. If both collapse the problem is the loss or the data; if only the
    circuit's collapse, the quantum part has stopped learning while the
    classical wrapper carries on — the failure mode that makes a nominally
    quantum result actually classical.
    """
    prior = agent.wm.rssm.prior_net
    q, c = [], []
    for name, param in prior.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.detach().flatten()
        (q if "q." in name else c).append(g)
    out = {}
    if q:
        gq = torch.cat(q)
        out["prior_q_var"] = float(gq.var())
        out["prior_q_absmean"] = float(gq.abs().mean())
    if c:
        gc = torch.cat(c)
        out["prior_c_var"] = float(gc.var())
        out["prior_c_absmean"] = float(gc.abs().mean())
    if q and c and out.get("prior_c_absmean", 0) > 0:
        out["prior_ratio"] = out["prior_q_absmean"] / out["prior_c_absmean"]
    return out
