#!/bin/bash
# =====================================================================
#  The two ablations that decide whether the circuit is load-bearing.
#
#      bash ~/Documents/PhD/qtwm/run_ablations.sh
#
#  Resumable: --skip_existing means re-running costs nothing.
#  Roughly 6 h at JOBS=4. Reads and writes Phase_1 only; Phase 2 is
#  untouched.
#
#  WHAT IS ALREADY CONTROLLED, AND WHAT IS NOT
#  -------------------------------------------
#  The obvious alternative explanation for the closed-loop result is
#  that a narrow bottleneck in the token-mixing path is a useful
#  regulariser and the quantum circuit is incidental to it. That
#  explanation is ALREADY excluded by the existing control: the
#  classical twin is _ClassicalToken(d_model, n_qubits), which projects
#  d_model -> 6 -> ... -> d_model. Both arms bottleneck at six
#  dimensions in the same sublayer of the same block. The comparison
#  therefore isolates the circuit, not the bottleneck.
#
#  (Phase 2's prior_hidden sweep is a different axis -- it varies the
#  EXPANSION HEAD of the older non-attentional prior, not the
#  bottleneck. Conflating the two is easy and wrong.)
#
#  Two questions remain genuinely open, and both are cheap offline
#  where a seed costs under an hour instead of twenty-seven.
#
#  A. IS THE QUANTUMNESS LOAD-BEARING?
#     Delete every two-qubit gate. The circuit becomes a product of
#     independent single-qubit rotations and can generate no
#     entanglement whatsoever. Parameter count is unchanged, so the
#     comparison is otherwise exact. Bowles et al. (arXiv:2403.07059)
#     report that removing entanglement frequently leaves performance
#     unchanged; if that holds here, the honest description of the
#     construction is "a six-dimensional trigonometric feature map",
#     not "a quantum model", and the paper must say so.
#
#  B. DOES THE EFFECT DEPEND ON THE BOTTLENECK BEING NARROW?
#     Sweep the shared bottleneck width over 4, 6, 8 in BOTH arms. If
#     the quantum advantage is largest at the narrowest width and
#     vanishes as the bottleneck opens, the circuit is acting as a
#     particular kind of constrained map rather than contributing
#     something unavailable classically -- which is consistent with
#     the Fourier argument and worth stating plainly. If the gap is
#     flat in width, that reading is wrong.
#
#  Both run at d_model=32 and context K=4, matching the closed-loop
#  configuration, so the results speak to the same regime as the
#  returns in Paper B.
# =====================================================================
set -u

QTWM="$HOME/Documents/PhD/qtwm"
P1="$QTWM/Phase_1_Scaffold_test"
LOGS="$QTWM/logs_ablation"
JOBS="${JOBS:-4}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
CTX=4
DM=32

mkdir -p "$LOGS"
source "$HOME/.venvs/qtwm/bin/activate"

LOCK="$LOGS/.ablation.lock"
if [ -e "$LOCK" ]; then
  OTHER=$(cat "$LOCK" 2>/dev/null)
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "REFUSING TO START: already running as PID $OTHER"
    exit 1
  fi
  echo "[lock] stale lock from PID ${OTHER:-unknown}, taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd "$P1" || exit 1
say() { printf '\n===== %s =====\n%s\n' "$1" "$(date '+%F %T')"; }

# ---------------------------------------------------------------------
say "A  entanglement removed (quantum arm only, 8 seeds)"
# ---------------------------------------------------------------------
# Only the quantum arm varies: the classical twin has no entangler, so
# its brickwork runs at this configuration are the shared baseline and
# already exist.
for s in $SEEDS; do echo "$s"; done | xargs -P "$JOBS" -n 1 bash -c '
  OMP_NUM_THREADS=1 python train_dynamics.py \
    --context '"$CTX"' --d_model '"$DM"' --seed "$0" \
    --entangler none \
    --models quantum_transformer \
    --skip_existing true \
    > '"$LOGS"'/entnone_seed"$0".log 2>&1
  echo "  entangler=none seed=$0 rc=$?"
'

# ---------------------------------------------------------------------
say "B  bottleneck width sweep, both arms (4 and 8; 6 already exists)"
# ---------------------------------------------------------------------
# n_qubits sets the shared bottleneck: it is the width the quantum
# token layer projects down to, AND the width _ClassicalToken projects
# down to. Sweeping it moves both arms together, which is what makes
# this a test of the bottleneck rather than of the circuit.
#
# latent_dim stays at 6 -- it is the DATA dimension and is baked into
# the cached dataset. Changing it would silently invalidate the
# comparison, which is why it is not swept here.
for q in 4 8; do
  for s in $SEEDS; do echo "$q $s"; done
done | xargs -P "$JOBS" -n 2 bash -c '
  OMP_NUM_THREADS=1 python train_dynamics.py \
    --context '"$CTX"' --d_model '"$DM"' --n_qubits "$0" --seed "$1" \
    --models quantum_transformer classical_transformer \
    --skip_existing true \
    > '"$LOGS"'/nq"$0"_seed"$1".log 2>&1
  echo "  n_qubits=$0 seed=$1 rc=$?"
'

# ---------------------------------------------------------------------
say "aggregating"
# ---------------------------------------------------------------------
# analyze.py keys on the full configuration including n_qubits and
# entangler, and halts rather than selecting a winner if two runs
# collide. If it refuses, a setting varied that is not in the key --
# read the message, do not work around it.
python analyze.py --out REPORT.md > "$LOGS/analysis.txt" 2>&1 \
  && echo "  REPORT.md written" \
  || { echo "  ANALYSIS FAILED:"; cat "$LOGS/analysis.txt"; }

say "DONE"
echo "report: $P1/REPORT.md"
echo "logs:   $LOGS"
