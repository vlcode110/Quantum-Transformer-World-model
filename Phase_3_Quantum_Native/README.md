# Phase 3 Stage A — Quantum-Native Dynamics

Testing the one hypothesis Phases 1 and 2 generated: **quantum attention helps where dynamics are chaotic or history-dependent, and does nothing where the current state nearly determines the next.**

Phase 1 found no advantage on classical control dynamics (smooth, near-Markovian) but a clear one on chaotic forecasting (24.7%, p = 0.022). Stage A tests that directly on dynamics that are themselves quantum.

> **Status: executed. The answer is null.** See "Results" at the bottom before
> reading the rest — the design notes below describe what was tested, not what
> was found.

## The system

A transverse-field Ising chain — the standard minimal model of an interacting quantum system:

> H(t) = −J Σᵢ ZᵢZᵢ₊₁ − h(t) Σᵢ Xᵢ

Neighbouring spins want to align (the J term); a control field the experimenter sets tries to flip them (the h term). At h = J the system sits at a critical point where the dynamics are richest. The chain starts in an alternating state, the field switches between five levels over time, and expectation values are recorded after each interval.

The evolution is computed **exactly** — the Hamiltonian is built densely and exponentiated once per field level, then trajectories are matrix-vector products. At ten spins the state is 1,024 complex numbers, so exactness costs nothing and removes Trotter error as a confound.

**By default only ⟨Zᵢ⟩ is recorded, and that is deliberate.** Those N numbers do not determine the quantum state, so their dynamics are not closed: predicting the next ⟨Z⟩ genuinely requires history. That makes the task non-Markovian by construction — exactly the regime the hypothesis points at. Setting `--observables zx` adds ⟨Xᵢ⟩, moving closer to Markovian, and is the control arm.

## The experiment is the scaling

Both architectures have a **fixed** size while the system they model grows from 2 to 10 spins. The question is not who wins at any single size but **whose accuracy degrades more slowly**.

- A constant ratio → unremarkable.
- A widening one → the classical model needs increasingly more capacity to track a quantum system while the quantum model does not. That is the strongest advantage argument available, and the one with theoretical backing.

## Run order

```bash
source ~/.venvs/qtwm/bin/activate
cd ~/Documents/PhD/qtwm/Phase_3_Quantum_Native

python config.py                       # settings and reasoning
python spin_env.py --n_spins 4         # generate + diagnose the physics
python selftest.py                     # ~1 min, MUST pass first
python dataset.py --n_spins 4          # windows, splits, baselines

python train_dynamics.py --n_spins 4 --epochs 2     # smoke, minutes
```

Then the scaling sweep. Data generation is cheap; training dominates:

```bash
for n in 2 3 4 5 6 7 8; do
  for s in 0 1 2; do
    python train_dynamics.py --n_spins $n --seed $s
  done
done
python analyze.py
python analyze.py --horizon 10
```

Roughly one hour per quantum run at six qubits, so the full sweep is a few nights. Start with `--n_spins 2 4 6` and three seeds to see whether a trend exists before committing to the rest.

## Lessons from Phases 1 and 2 built in

**Trivial baselines are computed automatically** and reported alongside every result. Phase 1 spent weeks quoting errors with no scale reference; here `vs_persistence` and `r_squared` appear in every `result.json`.

**The split is fixed across seeds** (`fixed_split: True`). Phase 1 varied the data partition per seed as well as the model initialisation, which inflated seed-to-seed variance to roughly six times the effect size and cost real statistical power. Seeds here vary only model initialisation.

**Saturation is checked, not assumed.** `analyze.py` reports R² and the multiple over persistence at every system size. A model at R² > 0.999 is at the task's ceiling where architectures cannot differ — the trap that made Phase 1's first comparison uninterpretable.

**Both horizons are reported**, because Phase 1 found that which one discriminates depends on how smoothly the dynamics evolve, with opposite answers on its two environments. Do not pick one in advance.

**No autoencoder.** The recorded observables *are* the prediction target, which removes a whole class of doubt about whether compression was hiding the effect.

## Files

```
config.py           settings, each with its justification inline
spin_env.py         the Ising chain: Hamiltonian, evolution, measurement
dataset.py          windowing, trajectory-level splitting, trivial baselines
train_dynamics.py   training and evaluation, reusing Phase 1's models
analyze.py          the scaling analysis and verdict
selftest.py         integration checks
```

Phase 1's model definitions are imported by file path rather than copied, so "the quantum transformer" cannot drift between phases. Nothing in Phase 1 is modified.

## What each outcome means

**Ratio falls with system size** — the result this phase was built to find. Confirm it extends further before claiming it, and check the largest sizes are not saturating.

**Ratio flat** — no scaling effect. Combined with Phases 1 and 2, that would be strong evidence that quantum attention offers nothing for dynamics modelling at these scales, which is a clean and publishable conclusion.

**Ratio rises** — the quantum model degrades faster. Also clean, also publishable, and it would redirect the project toward Stage B on different grounds.

## Results

Two experiments were run. Both returned null.

**1. Scaling (2, 4, 6 spins × 3 seeds).** The quantum/classical error ratio was
flat and if anything drifted the wrong way: 0.957 → 0.969 → 0.975. No widening
gap, so the theoretical argument that a fixed-size quantum model should track a
growing quantum system better than a fixed-size classical one gets no support
here.

**2. The observables contrast (4 spins × 10 seeds per arm).** The direct test of
the Markovianity hypothesis. At three seeds per arm it looked promising —
non-Markovian `z` at 0.969 against nearer-Markovian `zx` at 1.025, a 5.6-point
separation in the predicted direction. At ten seeds per arm it was gone:

| horizon | `z` ratio | `zx` ratio | separation |
|---|---|---|---|
| k=1  | 1.001 | 1.015 | +1.4 pt |
| k=5  | 1.020 | 1.023 | +0.3 pt |
| k=10 | 1.036 | 1.032 | **−0.4 pt** |
| k=20 | 1.026 | 1.044 | +1.8 pt |

The separation changes sign between horizons, which is what noise looks like.
Nothing is significant anywhere: sign tests 0.34–1.00, paired permutation tests
0.22–0.94. Both architectures land within about 2% of each other, with the
classical one marginally ahead on average, at roughly 20× less compute.

**The three-seed result was a small-sample artefact**, and the design warned it
would be: at 3 versus 3 the smallest attainable two-sided p was 0.100, so the
apparent effect was never separable from chance. Treat this as the phase's
methodological lesson — a promising direction at n=3 is not a direction.

**What this leaves.** Across Phase 1 (classical control dynamics), Phase 2
(closed-loop agent) and Phase 3 (quantum-native dynamics), a quantum
transformer core is at parity with a matched classical one. The only surviving
positive anywhere in the project is Phase 1's chaotic-forecasting result
(24.7%, p = 0.022), and after what happened here it needs replication at the
same seed count before it can be relied on.

## Next, after Stage A

Stage B is the intended application. Real quantum processors drift — qubit
frequencies wander, gate amplitudes detune, couplings shift — so they are
recalibrated constantly, and that calibration consumes the machine's own time,
which is the scarce resource. The proposal is to treat calibration as a control
problem: observations are calibration telemetry, actions are which
characterisation experiment to run next, dynamics are how the device drifts and
how each action changes what is known. Train a world model on the telemetry
stream, plan candidate calibration sequences inside the model, and spend
physical device time only on the sequence the planner selects. The metric is
device-seconds needed to hold mean gate fidelity above a threshold over a fixed
window, against the standard scheduled routine.

Two notes. The world model here does **not** need to be quantum — the classical
DreamerV3 in this repository does the job, and the null results above are the
justification for that choice rather than an obstacle to it. And this direction
has substantial prior art: Google published reinforcement-learning calibration
of quantum error correction in Nature in 2026 (arXiv:2511.08493) with related
patent applications filed. Anyone picking this up should read those claims
first. Stage B was not built; it is recorded here as the direction Stage A was
meant to inform.
