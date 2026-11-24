# Small DreamerV3 — Classic Control Reproduction (Phase 0)

A compact, faithful PyTorch implementation of DreamerV3 ([Hafner et al., 2023, arXiv:2301.04104](https://arxiv.org/abs/2301.04104)) for low-dimensional control — the classical scaffold for the quantum transformer world model project. The RSSM sequence core (`dreamer/nets.py: RSSM`) is deliberately isolated so it can later be swapped for a quantum module.

## What's implemented (DreamerV3-specific ingredients)

- RSSM with categorical stochastic latents (16×16), straight-through gradients, 1% unimix
- KL balancing (dyn 0.5 / rep 0.1) with free bits (1 nat)
- Symlog observation encoding and symlog-MSE decoder
- Twohot symlog regression for reward and critic (255 bins)
- Imagination training (horizon 15), λ-returns (λ=0.95, γ=0.997)
- Percentile-EMA return normalization for the actor
- REINFORCE actor gradient for discrete actions (CartPole), dynamics backprop for continuous (Pendulum)
- Slow-critic EMA regularization
- Flat replay stream with `is_first` state resets

Deviations from the paper (deliberate, for a small vector-obs setup): no image encoder, no LayerNorm-GRU (plain GRUCell + LayerNorm MLPs), single environment instance, no replay-ratio scheduling.

## Setup

```bash
cd ~/Documents/DreamerV3
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python train.py --task cartpole      # ~60k steps
python train.py --task pendulum      # ~150k steps
python plot.py                       # learning_curves.png from runs/*/log.csv
```

Device is auto-selected (CUDA → Apple MPS → CPU). Logs, checkpoints: `runs/<task>_seed<seed>/`.

Quick smoke test (2–3 min, verifies the full pipeline only):

```bash
python train.py --task cartpole --steps 1200 --warmup 200
```

## Expected results ("published-level")

| Task | Target | Typical budget | Rough wall-clock (M-series CPU/MPS) |
|---|---|---|---|
| CartPole-v1 | ≥ 475 (max 500) | 30–60k env steps | 1–3 h |
| Pendulum-v1 | ≥ −200 (good: ≈ −150) | 100–150k env steps | 3–6 h |

DreamerV3 solves these environments reliably; classic-control results at these budgets are consistent with the paper's "solves 150+ tasks with one config" claim and public reimplementations. Run 3–5 seeds (`--seed 0..4`) before calling it reproduced — single-seed RL results are noise.

Watch during training: `wm/recon` and `wm/kl` should fall within the first few thousand steps; `eval_return` should move meaningfully after ~10k (CartPole). If KL collapses to the free-bits floor (≈1.0/nat per step, i.e. exactly the clamp) while recon stays high, increase `ent_scale` or lower `model_lr`.

## Files

```
config.py            # small-config hyperparameters per task
train.py             # collect → train → eval loop, CSV logging, checkpoints
plot.py              # learning curves
dreamer/nets.py      # symlog/twohot utils, MLP, RSSM, Actor
dreamer/agent.py     # WorldModel, ActorCritic (imagination), Dreamer
dreamer/buffer.py    # flat-stream replay with is_first flags
```

## Phase 0 gate (from the research plan)

Classical Dreamer reproduces reference returns on both tasks → proceed to reproducing one quantum component (QASA / QLSTM) in PennyLane, then benchmark statevector throughput to fix the feasible qubit envelope.
