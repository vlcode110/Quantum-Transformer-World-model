#!/bin/bash
# =====================================================================
#  Every run still needed to finalise Papers A and B.
#
#  Run it with:        bash ~/Documents/PhD/qtwm/run_remaining.sh
#  Resume after a crash, a sleep or a Ctrl-C with exactly the same line.
#  Every stage skips work that is already finished, so restarting is free.
#
#  Stages run in order. Each is independently useful, so an interruption
#  after any stage still leaves the papers better off than before.
#
#    0  verification        seconds   confirms no condition is mixed
#    1  Phase 1 context     ~8 h      ctx 2,4,8 at FIXED width, 8 seeds
#    2  Phase 2 transformer ~18 h     the architecture claim, 5 seeds
#    3  Phase 1 ctx 16      ~8 h      fourth context point, optional
#
#  Stages 1 and 2 are what the papers need. Stage 3 adds a data point.
#
#  WHY STAGE 1 EXISTS. The context ablation already on disk is confounded
#  with model width: ctx2 ran at d_model 32, ctx4 at 8, ctx8 at 6. Those
#  three points cannot be compared, so the existing sweep cannot support
#  a statement about context length. Stage 1 re-runs all of it at the
#  current default width, d_model 32, which is the only way the trend
#  means anything.
# =====================================================================
set -u

QTWM="$HOME/Documents/PhD/qtwm"
P1="$QTWM/Phase_1_Scaffold_test"
P2="$QTWM/Phase_2_Closed_Loop"
VENV="$HOME/.venvs/qtwm/bin/activate"
LOGS="$QTWM/logs_remaining"
JOBS="${JOBS:-5}"

mkdir -p "$LOGS"

# ---------------------------------------------------------------------
# Single-instance lock.
#
# Two copies of this script running at once is worse than it sounds.
# Neither can see the other, the overwrite guard in train_agent.py only
# triggers on a COMPLETED run (result.json present), and every pending
# run is pending for both. So both processes train into the same
# directory simultaneously, interleaving writes to checkpoint.pt and
# log.csv, and the survivor writes a result.json over the wreckage. The
# run looks finished and its numbers are meaningless.
#
# It also doubles the parallelism, so 5 jobs becomes 10 on a machine
# sized for 5, and every run slows down.
#
# This happened. Hence the lock.
LOCK="$LOGS/.run_remaining.lock"
if [ -e "$LOCK" ]; then
  OTHER=$(cat "$LOCK" 2>/dev/null)
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "REFUSING TO START: already running as PID $OTHER."
    echo "  Watch it:  tail -f $LOGS/driver_resume.log"
    echo "  Stop it:   kill $OTHER; pkill -f train_agent.py"
    exit 1
  fi
  echo "[lock] stale lock from PID ${OTHER:-unknown}, taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# shellcheck disable=SC1090
source "$VENV"

say() { printf '\n===== %s =====\n%s\n' "$1" "$(date '+%Y-%m-%d %H:%M:%S')"; }

# ---------------------------------------------------------------------
say "STAGE 0  verification"
# ---------------------------------------------------------------------
# Two things must be true before any of the numbers below are trusted.
#
# (a) Every partial-observation Phase 2 run must be the SAME architecture.
#     There was a window in which prior_arch was live but tag() did not
#     encode it, so a transformer run could be written under a name that
#     says nothing about architecture. 18658 prior params means the MLP
#     prior; 84770 means the transformer. A mix invalidates the paired
#     comparison outright.
#
# (b) The untagged-settings guard must actually refuse. If it does not,
#     the collision that destroyed two 80,000-step runs can recur.

cd "$P2" || exit 1
echo "-- partial-observation runs: prior_params must all be 18658"
MIXED=0
for d in runs/*_po_*; do
  [ -f "$d/result.json" ] || continue
  pp=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["prior_params"])' \
        "$d/result.json" 2>/dev/null)
  st=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["env_steps"])' \
        "$d/result.json" 2>/dev/null)
  printf '   %-9s %-7s %s\n' "${pp:-?}" "${st:-?}" "$(basename "$d")"
  [ "${pp:-0}" = "18658" ] || MIXED=1
done
if [ "$MIXED" = "1" ]; then
  echo
  echo "   !! A partial-observation run is NOT the 18658-parameter MLP prior."
  echo "   !! The paired comparison in Paper B mixes architectures and is void."
  echo "   !! Move the offending directory aside before relying on Table 'po'."
  echo
fi

cd "$P1" || exit 1
echo "-- untagged-settings guard (expect a refusal here)"
if python3 train_dynamics.py --n_heads 4 2>&1 | grep -q "REFUSING TO RUN"; then
  echo "   guard works"
else
  echo "   !! GUARD DID NOT FIRE — stop and fix before sweeping"
  exit 1
fi

# ---------------------------------------------------------------------
say "STAGE 1  Phase 1 context ablation, width-matched at d_model 32"
# ---------------------------------------------------------------------
# Only the two transformers. qlstm and pqc_mlp are Phase 1 baselines
# already measured at ctx 8; they contribute nothing to a context sweep
# and they are 60% of its cost (qlstm alone is ~6,850 s per seed).
#
# ctx 2 already has seeds 0-2 at this width, so those are skipped.

cd "$P1" || exit 1
for k in 2 4 8; do
  for s in 0 1 2 3 4 5 6 7; do
    echo "$k $s"
  done
done | xargs -P "$JOBS" -n 2 bash -c '
  OMP_NUM_THREADS=1 python train_dynamics.py \
    --context "$0" --seed "$1" --d_model 32 \
    --models quantum_transformer classical_transformer \
    --skip_existing true \
    > '"$LOGS"'/p1_ctx"$0"_seed"$1".log 2>&1
  echo "  p1 done ctx=$0 seed=$1 rc=$?"
'

echo "-- aggregating Phase 1"
python3 analyze.py > "$LOGS/p1_analysis.txt" 2>&1 || \
  echo "   analyze.py failed; see $LOGS/p1_analysis.txt"

# ---------------------------------------------------------------------
say "STAGE 2  Phase 2 quantum transformer world model, partial observation"
# ---------------------------------------------------------------------
# This is the run set the architecture claim rests on. Every Phase 2 run
# currently on disk uses the MLP prior, so the paper's central object --
# a quantum TRANSFORMER world model driving an agent -- has never been
# trained closed-loop. Three seeds per arm would establish that it works
# and that returns are at parity; five gives a little margin.
#
# K=4, not the default 8: measured 2.8 steps/s against 0.4, a 7x
# difference, which turns a 55 h run into an 8 h one.
#
# Returns are NOT expected to separate. At the dispersion measured over
# seven MLP seeds (sd 237 on a +58 difference) that would need ~130
# seeds. The purpose here is existence and parity, not significance.

# Seeds 0-7, not 0-4. At four seeds the smallest attainable two-sided p
# under an exact sign-flip test is 0.125, so the comparison could not
# reach significance however large the effect. Six seeds puts the floor
# at 0.031 and eight at 0.008. The measured dispersion (sd 89 on a +179
# difference) says ~2 seeds would suffice for power, so the binding
# constraint here is the permutation floor, not the variance.
#
# Runs already finished are skipped by the overwrite guard, which exits
# non-zero and prints REFUSING TO OVERWRITE. That is the guard working.

cd "$P2" || exit 1
for s in 0 1 2 3 4 5 6 7; do
  echo "true $s"
  echo "false $s"
done | xargs -P "$JOBS" -n 2 bash -c '
  OMP_NUM_THREADS=1 python train_agent.py \
    --task cartpole --quantum "$0" --seed "$1" \
    --steps 80000 --partial_obs true \
    --prior_arch transformer --prior_context 4 \
    > '"$LOGS"'/p2_tf_po_"$0"_s"$1".log 2>&1
  echo "  p2 done quantum=$0 seed=$1 rc=$?"
'

# ---------------------------------------------------------------------
say "STAGE 2b  same architecture, FULL observation"
# ---------------------------------------------------------------------
# The crossover is the paper's argument: the quantum prior should lose on
# the Markovian task and win where history is required. Right now the two
# halves of that claim come from different architectures -- full
# observation measured on the MLP prior, partial observation on the
# transformer -- so it is not a crossover, it is two unrelated numbers.
# These runs supply the matching full-observation arm.
#
# Set SKIP_FULLOBS=1 to defer. Doing so means the paper can claim the
# partial-observation result but NOT the crossover.

if [ "${SKIP_FULLOBS:-0}" = "1" ]; then
  echo "skipped by request -- the crossover claim is unavailable without it"
else
  for s in 0 1 2 3 4 5; do
    echo "true $s"
    echo "false $s"
  done | xargs -P "$JOBS" -n 2 bash -c '
    OMP_NUM_THREADS=1 python train_agent.py \
      --task cartpole --quantum "$0" --seed "$1" \
      --steps 80000 --partial_obs false \
      --prior_arch transformer --prior_context 4 \
      > '"$LOGS"'/p2_tf_full_"$0"_s"$1".log 2>&1
    echo "  p2 full done quantum=$0 seed=$1 rc=$?"
  '
fi

echo "-- re-evaluating at 50 episodes"
python reevaluate.py --episodes 50 --out reevaluation_final.json \
  > "$LOGS/p2_reevaluate.txt" 2>&1
tail -40 "$LOGS/p2_reevaluate.txt"

echo "-- imagination quality"
python imagination_quality.py --episodes 20 \
  > "$LOGS/p2_imagination.txt" 2>&1 || \
  echo "   imagination_quality.py failed; see the log"

# ---------------------------------------------------------------------
say "STAGE 3  Phase 1 context 16, optional fourth point"
# ---------------------------------------------------------------------
# Roughly 8 h at JOBS=5. Skip it freely: three context points already
# show a trend, and the Phase 2 agent runs at K=4, so 16 is outside the
# range the paper actually uses. Set SKIP_CTX16=1 to pass over it.

if [ "${SKIP_CTX16:-0}" = "1" ]; then
  echo "skipped by request"
else
  cd "$P1" || exit 1
  for s in 0 1 2 3 4 5 6 7; do echo "16 $s"; done | \
  xargs -P "$JOBS" -n 2 bash -c '
    OMP_NUM_THREADS=1 python train_dynamics.py \
      --context "$0" --seed "$1" --d_model 32 \
      --models quantum_transformer classical_transformer \
      --skip_existing true \
      > '"$LOGS"'/p1_ctx"$0"_seed"$1".log 2>&1
    echo "  p1 done ctx=$0 seed=$1 rc=$?"
  '
  python3 analyze.py > "$LOGS/p1_analysis_with_ctx16.txt" 2>&1 || true
fi

say "ALL STAGES COMPLETE"
echo "logs:    $LOGS"
echo "Phase 1: $LOGS/p1_analysis.txt"
echo "Phase 2: $LOGS/p2_reevaluate.txt  $LOGS/p2_imagination.txt"
