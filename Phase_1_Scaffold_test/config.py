"""Central configuration for Phase 1 (offline quantum dynamics model).

EVERY decision that is not yet settled by data lives here and is marked
`DECISION:`. Nothing downstream hard-codes these values — change them here (or
override on the CLI with --key=value) and the whole pipeline follows.

Run `python config.py` to print the current settings and the open decisions.
"""
from dataclasses import dataclass, field, asdict, fields
import json
import os


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    # DECISION: where Phase 0 lives. Change these two if you move folders.
    # (You chose "new folder only, leave Phase 0 in place".)
    dreamer_root: str = "../DreamerV3"
    phase0_root: str = "../../../Phase0-analysis"   # not vendored; analysis only
    data_dir: str = "data/datasets"
    run_dir: str = "runs"

    # ------------------------------------------------------------- experiment
    task: str = "cartpole"          # cartpole | pendulum (pendulum: BLOCKED, see README)
    seed: int = 0
    device: str = "cpu"             # PennyLane backprop is CPU-bound

    # ------------------------------------------------------------------- data
    n_episodes: int = 200           # episodes to roll out from the trained agent
    policy_mode: str = "mixed"      # DECISION: greedy | sampled | mixed.
                                    # 'mixed' = 70% policy / 30% epsilon-random,
                                    # giving state coverage a dynamics model needs.
                                    # Pure greedy gives a narrow on-policy manifold
                                    # that flatters every model equally.
    epsilon: float = 0.3            # only used when policy_mode == 'mixed'
    context: int = 8                # DECISION: (z,a) history length fed to the core.
                                    # Longer = more tokens = more circuit evals.
    split_by_episode: bool = True   # keep True: windowing across a split boundary
                                    # leaks future frames into train. Flag exists
                                    # only so the leakage ablation is one line.
    split_fracs: tuple = (0.7, 0.15, 0.15)   # train / val / test, by EPISODE

    # ---------------------------------------------------------- latent choice
    # DECISION (you chose "both, compressed as default"):
    #   'compressed' -> autoencoder over Dreamer's [deter|stoch] features
    #   'raw'        -> predict the 16x16 categorical stochastic latent directly
    latent_mode: str = "compressed"
    # DECISION: 6 = n_qubits, so angle encoding maps one feature per qubit with
    # nothing wasted. Keep it at or below the SMALLEST n_qubits you intend to
    # run, since the dataset is built once and shared across widths — raising it
    # to 8 would silently make the 6-qubit ablation arm lossy.
    latent_dim: int = 6
    # MEASURED on the 20-episode smoke dataset: 30 epochs left the AE
    # unconverged (deter explained variance 81.8%); 150 reaches 88.5% and is
    # into diminishing returns. Recheck on the full 200-episode dataset — with
    # ~10x the data the optimum will differ.
    ae_epochs: int = 150
    ae_hidden: int = 128
    ae_lr: float = 1e-3

    # ----------------------------------------------------------------- models
    # DECISION: which families to run. Each is registered in models/__init__.py.
    models: tuple = ("quantum_transformer", "classical_transformer",
                     "qlstm", "pqc_mlp")

    # --- quantum core -------------------------------------------------------
    # SET FROM PHASE 0 MEASUREMENTS (2026-07-20). Both ceilings are known:
    #
    #   throughput (default.qubit, batch 512, L=6, 17,480 grad steps/run):
    #        6q ->  0.81 h/run,  19.5 h for a 24-run sweep
    #        8q ->  2.89 h/run,  69.4 h for a 24-run sweep   <- 3h/run limit
    #       10q -> 15.11 h/run  (single runs only)
    #   trainability (brickwork + local readout): Var[grad] is FLAT in width
    #       (x0.90-0.99 per qubit) out to 10 qubits at every depth tested.
    #       Trainability is therefore NOT the binding constraint — throughput
    #       is. (The earlier x0.50 collapse was an artefact of the sequential
    #       CNOT ring; see `entangler` below.)
    #
    # DECISION: 6 qubits is the working default so ablation sweeps finish
    # overnight rather than over three days. Headline runs can use 8 via
    # `--n_qubits 8`; nothing else needs changing. The ceiling is simulator
    # speed, not physics — a faster backend (lightning + adjoint, or GPU) moves
    # it, and that is worth revisiting before treating 8 as fundamental.
    n_qubits: int = 6
    # DECISION: depth 4. Width is now free but DEPTH still costs gradient
    # magnitude: with brickwork+local, Var[grad] at 10q falls from 2.2e-1 (L=2)
    # to 5.8e-2 (L=4) to 1.3e-2 (L=6) — the decay is in depth, not width.
    # L=2 is the safest and cheapest; 4 buys expressivity at ~4x smaller
    # gradients. Treat depth as the trainability axis from here on.
    q_layers: int = 4
    encoding: str = "angle_reupload"  # DECISION: angle | angle_reupload | amplitude
                                      # Ablation axis; angle_reupload is the
                                      # literature default for this regime.
    # DECISION: 'local' <Z_i> vs 'global' <Z_0..Z_n>. MEASURED: with brickwork,
    # local gradients are FLAT in width (x0.90-0.99/qubit) while global still
    # decays exponentially (x0.41-0.49). Local is not a preference here, it is
    # the difference between a trainable and an untrainable model.
    readout: str = "local"
    # DECISION: entangling topology. MEASURED (Phase 0): 'brickwork' keeps local
    # gradients flat to at least 10 qubits; 'ring_cascade' — the sequential CNOT
    # ring used in the QASA reproduction and most PQC papers — applies CNOTs in
    # series, giving depth ~n per nominal layer, scrambling the state, and
    # collapsing local gradients at x0.5/qubit (3 orders of magnitude smaller at
    # 10q). Kept as an ablation arm because it is the literature default.
    entangler: str = "brickwork"
    quantum_layer_position: int = -1  # DECISION: which encoder layer is quantum.
                                      # -1 = last (QASA's finding: position
                                      # matters more than count). 0 = first.
    n_layers: int = 2               # total encoder layers (classical + quantum)
    d_model: int = 32
    n_heads: int = 2
    action_conditioning: str = "token"  # DECISION: how actions enter the core.
                                        # 'token'  = action appended as its own
                                        #            sequence token
                                        # 'concat' = concatenated to each latent
                                        # 'circuit'= extra rotation angles inside
                                        #            the PQC itself
                                        # NOTE: 'circuit' is implemented but was
                                        # never run in any reported experiment;
                                        # see README. 'token' is the default so
                                        # the comparison is against the
                                        # conventional design.

    # --- training -----------------------------------------------------------
    # MEASURED (classical, 200-episode dataset, 51,781 train windows): val
    # 1-step goes 0.00282 -> 0.00121 by epoch 10, then only -> 0.00096 by
    # epoch 40. Epochs 20-40 buy ~10% for 2x the compute, and the quantum
    # models cost ~24x per step, so that trade is not worth making across a
    # multi-seed sweep. Use 40 for headline runs if a final number matters.
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    grad_clip: float = 1.0
    early_stop_patience: int = 10

    # --- evaluation ---------------------------------------------------------
    rollout_horizons: tuple = (1, 5, 10, 20)   # k-step open-loop errors
    # Phase 1 gate (research plan): match parameter-matched classical on 1-step,
    # and stay within 2x on 10-step.
    gate_k: int = 10
    gate_ratio: float = 2.0

    # --- gradient-variance instrumentation ----------------------------------
    log_grad_var: bool = True
    grad_var_every: int = 50        # steps between measurements

    # --- sweep hygiene ------------------------------------------------------
    # Skip any run whose result.json already exists, so a long sweep can be
    # restarted after an interruption without repeating finished work. Off by
    # default: a silent skip is the wrong behaviour for a single deliberate
    # run, where you want to know you are about to overwrite something.
    skip_existing: bool = False
    # Escape hatch for guard_untagged() in train_dynamics.py. Only use it
    # together with a distinct --run_dir.
    allow_untagged: bool = False

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=list)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: (tuple(v) if isinstance(v, list) else v)
                      for k, v in d.items()})

    @classmethod
    def from_cli(cls, argv=None):
        """Accept --key value or --key=value for any field. Types follow the
        dataclass defaults, so `--n_qubits 10` and `--models a b` both work."""
        import argparse
        ap = argparse.ArgumentParser()
        for f in fields(cls):
            if f.type in (tuple,) or isinstance(f.default, tuple):
                ap.add_argument(f"--{f.name}", nargs="+", default=None)
            elif isinstance(f.default, bool):
                ap.add_argument(f"--{f.name}", type=lambda s: s.lower() != "false",
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

    def tag(self):
        """Short identifier used for run directories.

        MUST include every hyperparameter that distinguishes a run, or
        different configurations silently overwrite one another's results.
        d_model/n_layers were missing originally, so capacity-sweep runs
        clobbered the default-capacity runs at the same seed.
        """
        return (f"{self.task}_{self.latent_mode}{self.latent_dim}"
                f"_q{self.n_qubits}L{self.q_layers}_{self.encoding}"
                f"_{self.readout}_{self.entangler}_ctx{self.context}"
                f"_d{self.d_model}x{self.n_layers}_seed{self.seed}")


OPEN_DECISIONS = """
Settled by Phase 0 measurements (2026-07-20)
--------------------------------------------
 n_qubits = 6         Throughput-bound: 19.5 h vs 69.4 h per 24-run sweep at
                      6 vs 8 qubits. Use --n_qubits 8 for headline runs.
 q_layers = 4         Width is free with brickwork; DEPTH is now the
                      trainability axis (4x less gradient variance per
                      doubling). Drop to 2 if training destabilises.
 latent_dim = 6       One feature per qubit under angle encoding.
 entangler = brickwork   Local gradients flat (x0.90-0.99/qubit) vs x0.50 for
                      the sequential CNOT ring - a ~1300x difference at 10q.
 readout = local      Flat vs exponential decay. Not a preference.

Still open (deliberately deferred)
----------------------------------
 task=pendulum        Checkpoint now exists, so this is UNBLOCKED - but the
                      continuous-action code paths remain UNTESTED. Run the
                      cartpole pipeline end-to-end first.
 action_conditioning  'circuit' feeds the action into the PQC's own rotation
                      angles. Implemented but never run in any reported
                      experiment; 'token' is the default so the baseline is the
                      conventional design.
 policy_mode          'mixed' for state coverage; 'greedy' for strictly
                      on-policy dynamics.
 lightning.qubit      Uncharacterised. If adjoint differentiation beats
                      backprop at width, the n_qubits ceiling rises.
"""

if __name__ == "__main__":
    cfg = Config.from_cli()
    print(json.dumps(asdict(cfg), indent=2, default=list))
    print(OPEN_DECISIONS)
    print("run tag:", cfg.tag())
