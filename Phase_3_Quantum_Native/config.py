"""Phase 3 Stage A — can a quantum model track a quantum system better?

THE EXPERIMENT
Phase 1 found no advantage for quantum attention on classical control dynamics,
but a clear one (24.7%, p = 0.022) on chaotic forecasting. The hypothesis that
came out of it: quantum attention helps when the dynamics are chaotic or
depend on a long history, and does nothing when the current state nearly
determines the next.

Stage A tests that directly, on dynamics that are themselves quantum. A chain
of interacting spins evolves under a control field; the task is to predict its
measured properties one step ahead. Same protocol as Phase 1, different data.

THE KEY MEASUREMENT IS SCALING, NOT A SINGLE COMPARISON
A fixed-size quantum model and a fixed-size classical model both try to track
systems of growing size (2 to 10 spins). The question is not "who wins at N=6"
but "whose accuracy degrades more slowly as N grows". A constant gap is
unremarkable. A widening one says the classical model needs increasingly more
capacity to track a quantum system while the quantum model does not — which is
the strongest advantage argument available, and the one with theoretical
backing.

WHY THE OBSERVABLES ARE DELIBERATELY INCOMPLETE
By default only <Z_i> is recorded. Those N numbers do NOT determine the quantum
state, so their dynamics are not closed: predicting the next <Z> genuinely
requires history, not just the present. That makes the task non-Markovian by
construction — exactly the regime the Phase 1 hypothesis points at. Setting
observables='zx' records <X_i> too, which moves the task closer to Markovian
and provides the control arm for that hypothesis.

Run `python config.py` to print settings and the reasoning.
"""
from dataclasses import dataclass, asdict, fields
import json
import os


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    phase1_root: str = "../Phase_1_Scaffold_test"   # model definitions reused
    data_dir: str = "data"
    run_dir: str = "runs"

    # ----------------------------------------------------- the physical system
    n_spins: int = 6                # THE SCALING VARIABLE — swept 2..10
    coupling: float = 1.0           # J, spin-spin interaction strength
    # DECISION: control field levels. The system is driven by a field that the
    # "agent" would choose, so these are the analogue of discrete actions.
    # Values straddle the critical point h = J, where the dynamics are richest.
    field_levels: tuple = (0.2, 0.6, 1.0, 1.4, 1.8)
    dt: float = 0.35                # time per step
    # DECISION: 'z' records <Z_i> only — N numbers that do not determine the
    # state, so the task is non-Markovian and history matters.
    # 'zx' adds <X_i>, moving closer to Markovian. This is the control arm.
    observables: str = "z"

    # ------------------------------------------------------------------- data
    n_trajectories: int = 300
    traj_length: int = 60
    # DECISION: how the control field is chosen along a trajectory.
    # 'piecewise' holds a level for several steps then switches, which produces
    # structured dynamics rather than noise. 'random' switches every step.
    control_mode: str = "piecewise"
    hold_steps: int = 5
    init_state: str = "neel"        # neel | all_up | random_product

    # ------------------------------------------------------------ the models
    context: int = 8                # history length fed to the model
    # DECISION: the model circuit width is FIXED while the system size grows.
    # That is the whole point — we ask which architecture degrades more slowly,
    # not which one wins when allowed to grow with the problem.
    n_qubits: int = 6
    q_layers: int = 4
    encoding: str = "angle_reupload"
    readout: str = "local"
    entangler: str = "brickwork"
    d_model: int = 16               # capacity-limited, per the Phase 1 lesson
    n_heads: int = 2
    n_layers: int = 2
    quantum_layer_position: int = -1
    action_conditioning: str = "token"

    # --------------------------------------------------------------- training
    models: tuple = ("quantum_transformer", "classical_transformer")
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    grad_clip: float = 1.0
    early_stop_patience: int = 8
    seed: int = 0
    device: str = "cpu"

    # ------------------------------------------------------------- evaluation
    rollout_horizons: tuple = (1, 5, 10, 20)
    # DECISION (Phase 1 lesson): which horizon discriminates depends on how
    # smooth the dynamics are, and cannot be chosen in advance. Establish it
    # from the trivial baselines before interpreting any comparison.
    split_fracs: tuple = (0.7, 0.15, 0.15)
    # DECISION (Phase 1 lesson): fix the train/test split across seeds so that
    # seeds vary only model initialisation. Phase 1 varied the partition too,
    # which inflated seed-to-seed variance and cost statistical power.
    fixed_split: bool = True

    log_grad_var: bool = True
    grad_var_every: int = 50

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=list)

    @classmethod
    def from_cli(cls, argv=None):
        import argparse
        ap = argparse.ArgumentParser()
        for f in fields(cls):
            if isinstance(f.default, tuple):
                ap.add_argument(f"--{f.name}", nargs="+", default=None)
            elif isinstance(f.default, bool):
                ap.add_argument(f"--{f.name}",
                                type=lambda s: s.lower() not in ("false", "0", "no"),
                                default=None)
            else:
                ap.add_argument(f"--{f.name}", type=type(f.default), default=None)
        args, unknown = ap.parse_known_args(argv)
        cfg = cls()
        for k, v in vars(args).items():
            if v is None:
                continue
            cur = getattr(cfg, k)
            if isinstance(cur, tuple) and isinstance(v, list):
                elem = type(cur[0]) if cur else str
                v = tuple(elem(x) for x in v)
            setattr(cfg, k, v)
        if unknown:
            print(f"[config] ignoring unknown args: {unknown}")
        return cfg

    def obs_dim(self):
        return self.n_spins * (2 if self.observables == "zx" else 1)

    def data_tag(self):
        return (f"ising_n{self.n_spins}_{self.observables}_"
                f"J{self.coupling}_dt{self.dt}_{self.control_mode}"
                f"_T{self.traj_length}x{self.n_trajectories}")

    def tag(self):
        return (f"n{self.n_spins}_{self.observables}_d{self.d_model}"
                f"_q{self.n_qubits}L{self.q_layers}_{self.entangler}"
                f"_ctx{self.context}_seed{self.seed}")


NOTES = """
What is being simulated
-----------------------
A transverse-field Ising chain, the standard minimal model of an interacting
quantum system:

    H(t) = -J * sum_i Z_i Z_{i+1}  -  h(t) * sum_i X_i

The first term couples neighbouring spins; the second is a control field the
experimenter sets. At h = J the system sits at a critical point where its
dynamics are richest. The chain starts in a product state, the field is
switched between levels over time, and after each interval the expectation
values of the spins are recorded.

This is a genuine quantum system: its state lives in a space of size 2^N and
its evolution generates entanglement no classical description of the same size
can represent. Whether that makes it easier for a quantum model to predict is
the open question.

Scaling experiment
------------------
    for n in 2 3 4 5 6 7 8 9 10:
        generate data at n spins
        train both architectures, several seeds
    plot error against n for each architecture

Look for divergence, not for a winner at any single n.

Cost
----
Data generation is cheap: at 10 spins the state is 1024 numbers, and the time
evolution operators are precomputed once per field level. Seconds per
trajectory set. Training is the expensive part, same as Phase 1 — roughly one
hour per quantum run at six qubits.
"""

if __name__ == "__main__":
    cfg = Config.from_cli()
    print(json.dumps(asdict(cfg), indent=2, default=list))
    print(NOTES)
    print("observation dim:", cfg.obs_dim())
    print("data tag:", cfg.data_tag())
    print("run tag :", cfg.tag())
