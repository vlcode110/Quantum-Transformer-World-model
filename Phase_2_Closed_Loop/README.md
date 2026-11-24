# Phase 2 — A Quantum World Model Driving an Agent

The novelty claim of the thesis. Phase 1 showed a quantum transformer can *predict* latent dynamics; Phase 2 puts a quantum model inside a working agent that learns entirely from imagined rollouts.

> **Status: written, never executed.** Run `selftest.py` first — it checks the whole integration in about a minute and will catch anything broken before you commit to a multi-hour run.

## What is quantum here

DreamerV3's RSSM keeps a deterministic recurrent state and a stochastic latent. The **prior network** maps the deterministic state to a distribution over the next stochastic latent — the model's belief about what happens next, without seeing the next observation. It is the component that does the world-modelling, and imagination depends on it entirely.

Phase 2 replaces exactly that network with a parameterized quantum circuit. Everything else — encoder, GRU, posterior, reward and continue heads, actor, critic, all training machinery — is the Phase 0 code, unchanged and already validated.

**Why this rather than porting the Phase 1 transformer core:** one circuit evaluation per RSSM step instead of one per sequence token; no reimplementation of Dreamer's recurrent interface (that is TransDreamer, one to two weeks of work); and a clean like-for-like control in the form of the same agent with a classical prior of matched shape.

## Nothing existing was modified

This directory imports from `DreamerV3/` and `Phase_1_Scaffold_test/` and writes to neither. The quantum substitution happens on an already-constructed agent (`quantum_rssm.swap_prior`), so the Phase 0 project behaves exactly as before.

Both projects define `config.py`, so their modules are loaded by explicit file path under unique names — the same collision that bit Phase 1 on its first run.

## Run order

```bash
source ~/.venvs/qtwm/bin/activate
cd ~/Documents/PhD/qtwm/Phase_2_Closed_Loop

python config.py                    # review settings and the compute reasoning
python selftest.py                  # ~1 min — MUST pass before anything else
python selftest.py --quantum false  # same for the control path

python train_agent.py --steps 3000 --warmup_steps 500          # smoke, ~15 min
python train_agent.py --quantum false --steps 3000 --warmup_steps 500
```

Then the real runs. **Always run the control with identical settings** — the compute budget here is reduced from DreamerV3's defaults, so published DreamerV3 numbers are not a valid comparison:

```bash
for s in 0 1 2; do
  python train_agent.py --quantum true  --seed $s
  python train_agent.py --quantum false --seed $s
done
python analyze.py
```

## The compute budget, and why it differs from Phase 0

The quantum prior runs once per RSSM step. Two places dominate: `observe()` makes `seq_len` sequential calls at `batch_size`, and `imagine()` makes `horizon` sequential calls over `batch_size × seq_len` states.

At DreamerV3's defaults (batch 16, seq 32, horizon 15) imagination alone is 15 sequential calls over 512 states — roughly 2.3 s per gradient step at six qubits, about 25 hours for one CartPole run. The reduced defaults here (batch 8, seq 16, horizon 10, gradient step every 4 env steps) bring that to roughly 3 hours.

This costs sample efficiency. It is the reason the classical control must run under identical settings, and the reason `analyze.py` groups runs by budget and refuses to pool across groups.

## The partial-observability variant

The one hypothesis Phase 1 generated: quantum attention appears to help when dynamics are chaotic or history-dependent, and not when the current state nearly determines the next. Hiding velocity forces the agent to infer speed from history, making the latent dynamics genuinely non-Markovian while changing nothing else.

```bash
python train_agent.py --partial_obs true --quantum true  --seed 0
python train_agent.py --partial_obs true --quantum false --seed 0
```

CartPole then exposes only cart position and pole angle; Pendulum only the angle. Expect both agents to score worse than on the full-observation task — the point is whether the *gap between them* changes.

## Files

```
config.py         all settings, with the compute reasoning documented inline
envs.py           environment construction and the partial-observability wrapper
quantum_rssm.py   the quantum prior, the matched classical control, the swap
selftest.py       11 integration checks, ~1 minute
train_agent.py    training loop, checkpointing, evaluation, gradient logging
analyze.py        learning curves, sample efficiency, gate verdict
```

## What success looks like

The plan's gate: the hybrid agent reaches the environment's threshold (CartPole 475, Pendulum −200) within 3× the environment steps of the control. `analyze.py` computes this automatically and reports PASS or FAIL.

Given Phase 1, **expect parity rather than advantage**. The purpose of this phase is to demonstrate the system works end to end — that is the contribution — not to win a benchmark.

## Known gaps

- Never executed. The self-test exists precisely because of that.
- The classical control uses a bottleneck-shaped prior matching the quantum module, not DreamerV3's original prior. That isolates the circuit, but it means neither arm is exactly stock DreamerV3.
- One seed proves nothing; three is the minimum for the comparison to mean anything.
- Wall-clock figures from this machine have been unreliable throughout the project (identical workloads varying more than tenfold under background load). Do not quote them without re-timing on an idle machine.
