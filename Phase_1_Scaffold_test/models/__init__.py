"""Model registry. Add a family here and it becomes available to
train_dynamics.py, ablate.py and analyze.py without touching them."""
from .quantum_core import DynamicsCore
from .baselines import QLSTMDynamics, PQCMLP


def build_model(name, z_dim, action_dim, cfg):
    if name == "quantum_transformer":
        return DynamicsCore(z_dim, action_dim, cfg, quantum=True)
    if name == "classical_transformer":
        return DynamicsCore(z_dim, action_dim, cfg, quantum=False)
    if name == "qlstm":
        return QLSTMDynamics(z_dim, action_dim, cfg)
    if name == "pqc_mlp":
        return PQCMLP(z_dim, action_dim, cfg)
    raise ValueError(f"unknown model {name!r}")


REGISTRY = ("quantum_transformer", "classical_transformer", "qlstm", "pqc_mlp")
IS_QUANTUM = {"quantum_transformer": True, "classical_transformer": False,
              "qlstm": True, "pqc_mlp": True}
