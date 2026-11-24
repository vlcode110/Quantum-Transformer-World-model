"""Phase 2 configuration: quantum world model driving an agent.

Design principle: this directory is SELF-CONTAINED. It imports from the
Phase 0 DreamerV3 project and the Phase 1 quantum core, but modifies neither.
Every quantum substitution happens by swapping a module on an already-built
agent (see quantum_rssm.py), so the Phase 0 code runs exactly as before.

Run `python config.py` to print settings and the reasoning behind the
compute-related choices.
"""
from dataclasses import dataclass, asdict, fields
import json
import os


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    # Both are now vendored inside this repository, so these are relative to
    # this file and a fresh clone needs no path editing. train_agent.py falls
    # back to the old out-of-tree location if the in-repo copy is absent.
    dreamer_root: str = "../DreamerV3"
    phase1_root: str = "../Phase_1_Scaffold_test"
    run_dir: str = "runs"

    # ------------------------------------------------------------- experiment
    task: str = "cartpole"          # cartpole | pendulum
    quantum: bool = True            # False = identical agent, classical prior
    seed: int = 0
    device: str = "cpu"             # PennyLane backprop is CPU-bound
    steps: int = 0                  # 0 = use the task default below

    # --------------------------------------------------- partial observability
    # DECISION: the central hypothesis from Phase 1 is that quantum attention
    # helps when the dynamics are history-dependent rather than near-Markovian.
    # Hiding velocity forces the model to infer it from history, which makes
    # the latent dynamics genuinely non-Markovian while changing nothing else.
    #   cartpole full obs: [cart_pos, cart_vel, pole_angle, pole_vel]
    #   pendulum full obs: [cos(theta), sin(theta), theta_dot]
    partial_obs: bool = False
    # indices kept when partial_obs is True (positions only, velocities hidden)
    partial_obs_keep: tuple = ()    # empty = use the per-task default in envs.py

    # ------------------------------------------------------- quantum circuit
    # Carried over from Phase 0 measurements; see Phase 1 config for the
    # reasoning. Width is throughput-bound, not trainability-bound.
    n_qubits: int = 6
    q_layers: int = 4
    encoding: str = "angle_reupload"
    readout: str = "local"
    entangler: str = "brickwork"
    prior_hidden: int = 64          # width of the classical head after readout

    # ------------------------------------------------------- prior architecture
    # 'transformer' makes the prior a quantum TRANSFORMER over a window of
    # recent recurrent states: Phase 1's DynamicsCore, with multi-head
    # attention and the PQC as its token-mixing block, reading the last
    # `prior_context` deterministic states instead of only the current one.
    # This is what makes the agent a quantum transformer world model.
    #
    # 'mlp' is the original Phase 2 prior — down-projection, circuit,
    # expansion head, no attention and no history. Kept as the ablation that
    # isolates what the attention contributes.
    prior_arch: str = "transformer"       # transformer | mlp
    prior_context: int = 8                # K: recurrent states attended over
    d_model: int = 64                     # transformer width
    n_heads: int = 4
    n_layers: int = 2                     # encoder blocks
    quantum_layer_position: int = 1       # which block carries the PQC
    action_conditioning: str = "none"     # actions are already folded into
                                          # deter by the GRU, so the prior does
                                          # not re-condition on them

    # --------------------------------------------------------- compute budget
    # WHY THESE DIFFER FROM DREAMERV3 DEFAULTS.
    # The quantum prior is evaluated once per RSSM step. Two places dominate:
    #   observe()   seq_len sequential calls at batch_size
    #   imagine()   horizon sequential calls at batch_size*seq_len states
    # At DreamerV3's defaults (batch 16, seq 32, horizon 15) imagination alone
    # is 15 sequential calls over 512 states, ~2.3 s per gradient step at six
    # qubits — roughly 25 h for a CartPole run. The reductions below bring a
    # run to ~3 h. They cost sample efficiency, which is why the classical
    # control MUST be run with the identical settings rather than compared
    # against published DreamerV3 numbers.
    batch_size: int = 8             # DreamerV3 default 16
    seq_len: int = 16               # DreamerV3 default 32
    horizon: int = 10               # DreamerV3 default 15
    train_every: int = 4            # DreamerV3 default 2 (grad step per N env steps)
    warmup_steps: int = 1000

    # ------------------------------------------------------------ evaluation
    eval_every: int = 2500
    eval_episodes: int = 5

    # ------------------------------------------- gradient instrumentation
    log_grad_var: bool = True
    grad_var_every: int = 25

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
                ap.add_argument(f"--{f.name}", type=lambda s: s.lower() not in
                                ("false", "0", "no"), default=None)
            else:
                ap.add_argument(f"--{f.name}", type=type(f.default), default=None)
        args, unknown = ap.parse_known_args(argv)
        cfg = cls()
        for k, v in vars(args).items():
            if v is None:
                continue
            cur = getattr(cfg, k)
            if isinstance(cur, tuple) and isinstance(v, list):
                v = tuple(int(x) for x in v)
            setattr(cfg, k, v)
        if unknown:
            print(f"[config] ignoring unknown args: {unknown}")
        return cfg

    def tag(self):
        """Directory name for a run. EVERY setting appears, unconditionally.

        An earlier version appended only NON-DEFAULT values, so that existing
        directories would keep the names they already had. That destroyed two
        80,000-step runs: a head-width ablation at the default width produced
        exactly the same tag as the main experiment at a different budget, and
        overwrote it. Both runs completed normally and left a plausible
        result.json behind, so nothing looked wrong until the numbers were
        checked against their timestamps.

        Hence: no conditional suffixes, no omitted fields, and `steps` is
        included. A directory name being ugly, long or redundant costs
        nothing. A name collision costs a day of compute and can silently
        corrupt a published number.

        If a setting is ever added to this config, add it here as well.
        """
        return "_".join([
            self.task,
            "quantum" if self.quantum else "classical",
            f"q{self.n_qubits}L{self.q_layers}" if self.quantum else "ctrl",
            "po" if self.partial_obs else "full",
            f"b{self.batch_size}x{self.seq_len}h{self.horizon}",
            f"s{resolve_steps(self)}",
            f"arch{self.prior_arch}",
            f"K{self.prior_context}",
            f"dm{self.d_model}",
            f"nh{self.n_heads}",
            f"nl{self.n_layers}",
            f"qp{self.quantum_layer_position}",
            f"ph{self.prior_hidden}",
            f"ent{self.entangler}",
            f"ro{self.readout}",
            f"enc{self.encoding}",
            f"seed{self.seed}",
        ])

    def legacy_tag(self):
        """The previous naming scheme, for locating runs made before the fix.

        Kept so existing directories can still be found by analysis scripts;
        nothing writes with it.
        """
        parts = [self.task,
                 "quantum" if self.quantum else "classical",
                 f"q{self.n_qubits}L{self.q_layers}" if self.quantum else "ctrl",
                 "po" if self.partial_obs else "full",
                 f"b{self.batch_size}x{self.seq_len}h{self.horizon}"]
        defaults = Config()
        for name, prefix in (("prior_hidden", "h"), ("entangler", "ent"),
                             ("readout", "ro"), ("encoding", "enc")):
            if getattr(self, name) != getattr(defaults, name):
                parts.append(f"{prefix}{getattr(self, name)}")
        parts.append(f"seed{self.seed}")
        return "_".join(parts)


# env steps per task. CartPole was solved by classical DreamerV3 at 45k with
# richer settings; the reduced budget here may need more, so the ceiling is
# generous and eval logging shows where it actually crosses.
TASK_STEPS = {"cartpole": 80_000, "pendulum": 40_000}
TASK_GATE = {"cartpole": 475.0, "pendulum": -200.0}


def resolve_steps(cfg):
    return cfg.steps or TASK_STEPS[cfg.task]


NOTES = """
Phase 2 scope (deliberately minimal)
------------------------------------
Only the RSSM's PRIOR network is replaced by a quantum circuit. The prior is
the component that predicts the next stochastic latent from the deterministic
recurrent state -- i.e. it is the model's belief about what happens next,
which is precisely the "world model" function. Everything else in DreamerV3
(encoder, GRU, posterior, reward/continue heads, actor, critic, all training
machinery) is untouched and already validated in Phase 0.

Why this rather than porting the Phase 1 transformer core:
  * one circuit evaluation per RSSM step instead of one per sequence token
  * no reimplementation of Dreamer's recurrent interface (that is TransDreamer)
  * a clean like-for-like control: the same agent with a classical prior
  * it still constitutes a quantum world model driving an agent, which is the
    novelty claim the thesis rests on

Expected outcome, given Phase 1: parity rather than advantage. The purpose is
to demonstrate the system works end to end, not to win a benchmark.
"""

if __name__ == "__main__":
    cfg = Config.from_cli()
    print(json.dumps(asdict(cfg), indent=2, default=list))
    print(NOTES)
    print("run tag:", cfg.tag())
    print("env steps:", resolve_steps(cfg))
