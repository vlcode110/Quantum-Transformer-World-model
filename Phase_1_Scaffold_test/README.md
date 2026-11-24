# Phase 1 — Offline Quantum Dynamics Model (scaffold)

Supervised next-latent prediction on trajectories from the trained Phase 0 agent. No RL loop — the modelling question is isolated deliberately, per the research plan.

> **Status: scaffold written, NOT executed.** My sandbox has been down since the DreamerV3 work, so nothing here has been run. Treat the first smoke command below as the real test, exactly as with `train.py` (which had one genuine bug that the smoke test surfaced).

## What's implemented

| File | Purpose |
|---|---|
| `config.py` | Every decision in one dataclass, each marked `DECISION:`. CLI-overridable. `python config.py` prints current settings + open decisions. |
| `collect_trajectories.py` | Rolls out the trained Dreamer agent, dumps `[deter\|stoch]` features, actions, rewards, episode ids. Content-hash versioned. |
| `dataset.py` | Latent representation (compressed AE **or** raw categorical), **episode-level** splitting, windowing, version manifest. |
| `models/quantum_core.py` | Quantum transformer dynamics core + the parameter-matched classical twin (same class, `quantum=False`). |
| `models/baselines.py` | QLSTM port, PQC-MLP (no attention). |
| `train_dynamics.py` | Training loop, checkpointing, 1-step + open-loop k-step evaluation, gradient-variance logging, gate check. |
| `ablate.py` | OFAT sweep runner over the plan's ablation axes. `--dry_run` first. |
| `analyze.py` | Seed aggregation, bootstrap CIs, paired tests, gate verdict, plots, ablation tables. |

## Blocked / needs your input

- **`task=pendulum` is blocked** until that run finishes and writes its checkpoint. Continuous-action paths are written but **untested** — `collect_trajectories.py` exits with a clear message rather than half-working.
- **`n_qubits` / `q_layers` are placeholders (8 / 4).** They must come from Phase 0: the *lower* of the throughput ceiling (`benchmark_statevector.py`) and the trainability ceiling (`gradient_variance.py`). Nothing else needs changing when you set them — one edit in `config.py`.
- **`latent_dim = 12` should be ≤ `n_qubits`** for one-feature-per-qubit angle encoding. Revisit alongside the above.

## Run order

```bash
source ~/.venvs/qtwm/bin/activate
cd "~/Documents/PhD/qtwm/Phase 1_Scaffold_test"   # note: quote the path, it has a space

python config.py                                   # review decisions first
python collect_trajectories.py --n_episodes 20     # smoke: ~1 min
python dataset.py                                  # check window/episode counts
python train_dynamics.py --models classical_transformer --epochs 2   # fastest path first
python train_dynamics.py --models quantum_transformer --epochs 2     # then the slow one
python ablate.py --dry_run                         # cost estimate before committing
python analyze.py
```

Start with the classical model: if the plumbing is broken, you find out in seconds rather than minutes.

Real run, once Phase 0 numbers are in:

```bash
python collect_trajectories.py                     # 200 episodes
for s in 0 1 2; do python train_dynamics.py --seed $s; done
python analyze.py
```

## Design choices worth knowing

**Episode-level splitting.** Windowing means timestep `t` and `t+1` are in different windows; a timestep-level split would put one in train and one in test. Silent leakage that flatters every model. `split_by_episode=True` is the default and the flag exists mainly so the leakage ablation is one line.

**Open-loop rollout metric.** `evaluate()` feeds the model its own predictions, not ground truth. Teacher-forced error looks much better and is not what the Phase 1 gate asks for — a model that's great at 1-step and diverges by k=10 is useless for Phase 2 imagination.

**The classical twin is the same class.** `DynamicsCore(quantum=False)` swaps only the PQC for a comparable MLP; everything else is byte-identical. This is what makes the gate comparison meaningful, and why the twin should never be "improved" separately.

**Why three baselines, not one.** If the quantum transformer beats the classical twin but not PQC-MLP, the win is quantum feature maps, not quantum *attention* — a different (and weaker) claim. QLSTM covers "is recurrence enough?". Being able to report that negative result cleanly is worth the extra runs.

**Gradient logging.** `grad_stats()` records variance and magnitude for circuit parameters, the same for classical parameters as a control, and their ratio. If both collapse it's the loss or the data, not a plateau. The ratio is the dimensionless "is the quantum part still learning?" signal, comparable across sizes and directly continuous with the Phase 0 baseline.

**Dataset versioning.** Every trajectory file carries a content hash over its generating parameters; every run records that hash. Runs from different dataset versions must not be pooled — `analyze.py` says so in its notes, but it does not currently enforce it (worth adding once you have more than one version).

## Not yet done

- Distribution-valued predictions (NLL/calibration metrics from §7.1). Currently point predictions with MSE — fine for the gate, but calibration is on the metrics list and needs a stochastic head.
- The conditional Born-machine latent head (Phase 3 item, patent-flagged).
- Enforcement of dataset-version consistency in `analyze.py`.
- Tests. There is no test suite yet; the smoke sequence above is standing in for one.
