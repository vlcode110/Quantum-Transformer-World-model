# A Quantum Transformer World Model — code and run records

Implementation and complete experimental records for work on quantum
sequence modelling and model-based reinforcement learning.

**The architecture.** A transformer over a window of recent latent
states, with a parameterised quantum circuit (PQC) as the token-mixing
block of one encoder layer, serving as the dynamics prior of a
DreamerV3-style recurrent state-space model and trained end to end.
Because latent imagination rolls the model forward on the prior alone,
the policy trains entirely on trajectories generated through a quantum
circuit.

**The methodology work.** Seed-to-seed dispersion measured across seven
populations, a failed replication, false-positive rates resampled from
observed dispersion, and a case series of analysis failures that produced
plausible wrong numbers.

---

## What the results actually say

Read this before using anything here, because the headline is a null and
the repository should not be mistaken for a demonstration of quantum
advantage.

- **Offline prediction shows no advantage anywhere.** Across 18 paired
  comparisons spanning three model capacities, four attention context
  lengths, three bottleneck widths and two environments, exactly one
  reaches *p* < 0.05, against 0.9 expected under the global null.
- **Removing the entanglement changes nothing.** Deleting every
  two-qubit gate — so the circuit becomes a product of independent
  single-qubit rotations, incapable of generating any entanglement —
  moves offline error by about 1% (1.009× at one step, 1.011× at ten).
  At this scale the construction performs identically to a
  six-dimensional trigonometric feature map.
- **Parity is what theory predicts.** A data re-uploading circuit
  computes a truncated Fourier series whose frequencies are fixed by the
  encoding. A classical surrogate over that spectrum matches the
  30-parameter circuit with **8** parameters.
- **One positive result exists and is not yet established.** Under
  partial observation the quantum transformer prior leads the
  capacity-matched classical twin by 144 return points over eight seeds
  (*p* = 0.039, ahead in 7/8). It is one of three nominally significant
  cells among 30 comparisons recorded here, against 1.5 expected.
  Replication at fresh seeds is in progress; until it lands, the result
  should not be cited as established.
- **Dispersion spans 29×** (sd of per-seed log-ratio 0.014 to 0.404)
  across populations measured with the same two architectures. The
  sample needed to detect a 10% effect accordingly runs from 6 seeds to
  150.

The records document the failures as well as the findings: a case series
of seven analysis errors that produced plausible wrong numbers rather
than crashes, four of which changed a reported result. Two were found by
running the audit on the pipeline that recommends it.

---

## Layout

```
Phase_1_Scaffold_test/   offline dynamics models. The quantum transformer
                         core, its capacity-matched classical twin, qlstm
                         and pqc_mlp baselines. Context and capacity
                         sweeps. 196 run records.
Phase_2_Closed_Loop/     the closed-loop agent. Swaps the RSSM prior on an
                         already-constructed DreamerV3 agent, so the
                         baseline implementation is unmodified. 66 run
                         records.
Phase_3_Quantum_Native/  exactly-simulated transverse-field Ising chains,
                         2-6 spins, as quantum-native dynamics. 89 run
                         records.
DreamerV3/               faithful small-scale DreamerV3. Phase 2 swaps a
                         module into an agent built from this; the
                         implementation itself is unmodified by the
                         quantum work.
QASA-repro/              quantum attention reproduction, the forecasting
                         tasks (sine, noisy sine, Mackey-Glass), the
                         gradient-variance probe, the classical Fourier
                         surrogate, and the IonQ hardware harness.
*.sh                     sweep drivers: resumable, locked against
                         double-launch, skip completed work.
collect_all.py           consolidates every run record into AUDIT_RAW.json.

```

The repository is self-contained: `DreamerV3/` and `QASA-repro/` were
originally kept outside this tree and are now vendored in, so a fresh
clone needs no path editing. Scripts resolve them relative to their own
location and fall back to the historical out-of-tree paths, so existing
local checkouts keep working.

**Checkpoints are not committed** — see `.gitignore` for why. Every
reported number derives from the committed `result.json`, `config.json`,
`log.csv` and `grad_log.json` files. Trajectory caches are also excluded:
`dataset.py` regenerates them deterministically, and each run records the
cache hash as `dataset_version` so results built on different trajectory
versions can be distinguished rather than silently pooled.

---

## Reproducing a result

```bash
python -m venv ~/.venvs/qtwm && source ~/.venvs/qtwm/bin/activate
pip install -r requirements.txt
python QASA-repro/check_env.py     # version traps, named with their fixes
```

Offline dynamics, one configuration:

```bash
cd Phase_1_Scaffold_test
python train_dynamics.py --context 4 --d_model 32 --seed 0 \
  --models quantum_transformer classical_transformer
python analyze.py --out REPORT.md
```

Closed-loop agent (about 8 h per run at K=4):

```bash
cd Phase_2_Closed_Loop
python test_transformer_prior.py          # 9 integration checks; run first
python train_agent.py --task cartpole --quantum true --seed 0 \
  --steps 80000 --partial_obs true --prior_arch transformer --prior_context 4
python reevaluate.py --episodes 50
```

Verify the reported numbers against the records:

```bash
python collect_all.py        # gathers every run into AUDIT_RAW.json
```

---

## Known issues

Left visible rather than quietly fixed, because both are the subject of
open work:

- **The forecasting data file contains three appended experiments plus a
  failed run.** Anything that reads it whole will pool populations that
  should be kept separate.
- **One loader pools five model widths.** Results at different capacities
  are aggregated as though they were replicates of one setting.

Both are analysis-side, not run-side: the underlying run records are
sound, and the configuration is recorded in full for every run.

---

## Notes for anyone building on this

**Aggregation keys.** Every analysis script keys on the complete run
configuration and **halts on collision** rather than silently selecting a
winner. This is not stylistic. Five of the seven failures in the case
series were a quantity that varied without being recorded as having
varied, and two of them destroyed completed runs or fabricated an effect
size while leaving the *p*-value unchanged. If you add a setting, add it
to the key.

**Run naming.** Directory names encode every configuration value
unconditionally. An earlier scheme appended only non-default values and
silently overwrote two completed 80,000-step runs.

**Entangler choice is not cosmetic.** With a local cost function,
brickwork keeps gradient variance flat in width while a sequential CNOT
ring decays by roughly 0.5 per added qubit — about 1,260× apart at ten
qubits. In shots needed to resolve a gradient to fixed relative
precision, that is ~3 against ~9,360. The ring cascade is the more
common choice in the literature.

**Seed control does not fully reproduce.** Two runs at identical seed and
configuration differ by up to 2× in absolute error, and in one
closed-loop case by 3.6×. The paired protocol absorbs most of it (log sd
0.071 against 0.477 across seeds), which is why comparisons here are
paired within seed. Leading hypothesis is thread count altering
floating-point reduction order; not established.

**`action_conditioning='circuit'`** is implemented in
`Phase_1_Scaffold_test/models/quantum_core.py` but was never run in any
reported experiment. All results use `'token'` or `'none'`.

---

## Hardware

Both circuits were executed on IonQ Forte-1: eight data qubits, two
layers, 504 shots, parameter-shift gradients. Measured gradients agree
with exact simulation. Reconstruction of ⟨Z₀⟩ from the returned
histograms reproduces the provider plugin's own expectation values
exactly as rational multiples of 1/504, which settles the bit-ordering
convention empirically.

Four jobs cost $672.80. Training on hardware is infeasible — the
parameter-shift rule at the parameter and step counts of one run would
need on the order of 10¹² shots.

`IONQ_API_KEY` is read from the environment. It is never written to a
file in this tree and must not be.

---

## Status

Not a finished artefact. The replication at fresh seeds is in progress,
and the two known issues above are open. Treat the single positive result
as provisional until the replication lands.

## Licence

MIT, see `LICENSE`. The DreamerV3 implementation is an independent
reimplementation for this project, not a fork.
