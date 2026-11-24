# QASA Reproduction — Quantum Adaptive Self-Attention (Phase 0)

Reproduces the core result of [QASA, arXiv:2504.05336](https://arxiv.org/abs/2504.05336) in PennyLane + PyTorch: a hybrid transformer whose **final encoder layer routes each token through a parameterized quantum circuit**, evaluated against a parameter-comparable classical twin on synthetic forecasting tasks.

## What the paper claims (what we're checking)

1. Minimal quantum integration — a single PQC-enhanced layer (36 trainable quantum parameters in the paper) — beats a classical transformer on several synthetic signals.
2. **Task-conditional benefit**: QASA excels on chaotic/noisy/trend-dominated signals; classical transformers stay ahead on clean periodic waveforms.
3. Position matters more than count (more quantum layers ≠ better).

This repro targets claims 1–2 with three signals: `sine` (clean periodic → classical should win), `noisy_sine`, and `mackey_glass` (chaotic → quantum layer should help).

## Architecture (faithful to Sec. 3 of the paper)

- Linear input projection + LayerNorm + sinusoidal positional encoding
- Classical encoder layer(s): MHA, residuals, GELU FFN
- Final quantum encoder layer: MHA → LayerNorm → per token `h_q = tanh(W_q h′)` → PQC on n+1 qubits with data re-uploading (RX/RZ), trainable RY + CNOT ring entanglement, ⟨Z⟩ readout → project back, residual, FFN
- Default: n=4 data qubits + 1 ancilla, 6 layers → **30 trainable quantum parameters** (paper: 36; exact circuit layout isn't fully specified, so ours is faithful-in-kind; tune `q_layers`/`n_qubits` in `qasa.py` to match counts)
- Classical baseline: same model, PQC swapped for a small tanh MLP

## Run

```bash
cd ~/Documents/QASA-repro
source ../DreamerV3/.venv/bin/activate   # reuse the venv
pip install pennylane

# smoke test (~1 min)
python train.py --signal sine --epochs 2 --limit 400

# full comparison (3 signals x 2 models x 3 seeds; ~1-3 h CPU)
python train.py --all

# single signal
python train.py --signal mackey_glass
```

Results append to `results/comparison.csv`; a summary table prints at the end.

## What success looks like

- `mackey_glass`: QASA mean test MSE ≤ classical (the paper's headline direction)
- `sine`: classical ≤ QASA (the honest half of the taxonomy — if QASA won everything, suspect a bug or an unfair baseline)
- Both models train stably; quantum gradients flow (loss decreases from epoch 1)

If the direction matches on 3 seeds, the component is reproduced for Phase 0 purposes — the point is validating our PennyLane hybrid-training toolchain and the quantum-attention building block, not exact numbers (the paper's synthetic generators aren't fully specified).

## Why this matters for the project

This quantum encoder layer is the prototype for the **quantum dynamics core** of the world model (research plan §5.1): same pattern (classical scaffold, per-token PQC with re-uploading, ⟨Z⟩ readout), later conditioned on actions and predicting next latents instead of next signal values. The PennyLane `TorchLayer` integration verified here is exactly what the Dreamer RSSM swap will use.

## Result — Mackey-Glass, 25 seeds (2026-09-30)

The chaotic-forecasting advantage does not replicate.

| seeds | ratio q/c | quantum ahead | paired permutation p |
|---|---|---|---|
| 0–12 (original) | 0.787 | 9/13 | 0.069 |
| 13–24 (fresh) | 1.058 | 7/12 | 0.583 |
| **0–24 (all)** | **0.907** | **16/25** | **0.240** |

95% bootstrap CI on the pooled ratio is [0.782, 1.057] — it contains 1.

Seeds 13–24 were run specifically to test the original claim and are therefore
an out-of-sample replication. They do not reproduce it: the fresh seeds put the
classical model marginally ahead. Pooled, the point estimate still favours
quantum by about 9%, but the interval crosses parity and no test rejects the
null.

Two further notes. The originally reported p = 0.022 came from a less
conservative test than the paired permutation on log-ratios used here, which
gives 0.069 on the same 13 seeds — the choice of test was doing real work.
And the per-seed ratios span 0.38 to 2.24, a six-fold range, on a task where
the data is identical across seeds and only the initialisation varies. With
spread that large, thirteen seeds was never enough to resolve a 20% effect.

Outcome variance is also indistinguishable between the architectures
(F = 0.87), so the Phase 1 variance finding does not appear here either.

**Status: the last positive result in the project is retired.**
