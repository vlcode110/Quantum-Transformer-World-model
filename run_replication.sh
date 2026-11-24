#!/bin/bash
# =====================================================================
#  THE DECISIVE RUN.
#
#      bash ~/Documents/PhD/qtwm/run_replication.sh
#
#  Replicates the partial-observation closed-loop comparison at seeds
#  8-15, both arms. Sixteen runs, ~8 h each, roughly two nights at
#  JOBS=5. Resumable; locked against double-launch.
#
#  WHY THIS RUN AND NOT ANOTHER
#  ----------------------------
#  Paper B's only positive finding is a +144 return-point difference
#  under partial observation at eight seeds (p = 0.039, ahead 7/8).
#  Everything else in both papers is a null:
#
#    - 18 paired offline comparisons, 1 below p = 0.05, 0.9 expected
#    - 7 comparisons in Paper A, 1 below p = 0.05, 0.35 expected
#    - removing every two-qubit gate changes offline error by ~1%
#    - the configuration that produces the +144 is, offline,
#      indistinguishable from its classical twin
#    - three accuracy measurements see no difference at all
#
#  Across both papers: 30 tests, 3 below 0.05, 1.5 expected under the
#  global null. The +144 is one of those three. Two readings remain and
#  the data cannot separate them:
#
#    (a) the effect is real and is not mediated by predictive accuracy
#        -- the decoupling result, a genuine finding about model-based RL
#    (b) it is the expected false positive among 30 comparisons
#
#  Fresh seeds separate them. A real effect of this size reappears; a
#  false positive does not. Three effects in this project have already
#  evaporated under more seeds (+212 -> +58 on the MLP prior; an
#  apparent 29% offline gain that was two pooled tasks; a 26% open-loop
#  figure that did not survive re-measurement), so the prior is not
#  favourable and the test is worth its two nights.
#
#  WHAT TO DO WITH THE OUTCOME -- decide this BEFORE looking
#  --------------------------------------------------------
#  Pre-committing avoids reading whatever arrives as confirmation.
#
#    Seeds 8-15 reproduce a difference of comparable size and sign
#      -> pool all sixteen seeds, report the pooled estimate as the
#         headline with the replication stated, and keep the decoupling
#         framing. p_min at n=16 is 3e-05, so significance is reachable.
#
#    Seeds 8-15 show no difference, or the opposite sign
#      -> the original was the false positive. Report both halves and
#         the pooled null. Paper B becomes an architecture-and-method
#         paper with an honest null, which is what Sections 6-10 already
#         support. The abstract loses its one positive claim.
#
#    Seeds 8-15 are ambiguous (same sign, much smaller, interval
#    spanning zero)
#      -> report pooled with the interval and do not claim the effect.
#         State the required n from the pooled dispersion.
#
#  In all three cases the paper is publishable. Only one of them lets it
#  claim a result.
#
#  RUNNING IT IN TWO NIGHTS
#  ------------------------
#  The seed list is settable, so the sixteen runs can be split:
#
#      SEEDS="8 9 10 11"    bash run_replication.sh     # night one
#      SEEDS="12 13 14 15"  bash run_replication.sh     # night two
#
#  Split by SEED, never by arm. The comparison is paired within seed, so
#  a block of seeds is a complete analysable increment whereas a block
#  of one arm is nothing until its partner finishes -- and any change in
#  machine conditions between the two nights would then fall entirely on
#  one side of the comparison.
#
#  ONE STATISTICAL CONDITION, AND IT IS NOT OPTIONAL
#  -------------------------------------------------
#  Splitting the work is fine. Letting the first block's RESULT decide
#  whether to run the second is not. That is optional stopping, and it
#  inflates the false-positive rate well above the nominal 5% -- the
#  exact failure this project's companion paper is about. If seeds 8-11
#  look encouraging and that is why you continue, or look poor and that
#  is why you stop, the resulting p-value means nothing.
#
#  So: commit now to running all eight seeds whatever block one shows.
#  Check the interim logs freely for operational reasons -- is it alive,
#  did a run crash, how long is left -- but do not compute the paired
#  comparison on a partial block and do not let it influence the plan.
#  reevaluate.py runs at the end of each invocation because it is cheap
#  and the file is useful; reading its verdict after block one and
#  acting on it is the thing to avoid.
# =====================================================================
set -u

QTWM="$HOME/Documents/PhD/qtwm"
P2="$QTWM/Phase_2_Closed_Loop"
LOGS="$QTWM/logs_replication"
JOBS="${JOBS:-5}"
SEEDS="${SEEDS:-8 9 10 11 12 13 14 15}"

mkdir -p "$LOGS"
source "$HOME/.venvs/qtwm/bin/activate"

LOCK="$LOGS/.replication.lock"
if [ -e "$LOCK" ]; then
  OTHER=$(cat "$LOCK" 2>/dev/null)
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "REFUSING TO START: already running as PID $OTHER."
    echo "  Watch:  bash $QTWM/status.sh"
    echo "  Stop:   kill $OTHER; pkill -f train_agent.py"
    exit 1
  fi
  echo "[lock] stale lock from PID ${OTHER:-unknown}, taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd "$P2" || exit 1
printf '\n===== REPLICATION, seeds %s =====\n%s\n\n' \
  "$SEEDS" "$(date '+%F %T')"

# Settings identical to seeds 0-7 in every respect except the seed.
# Anything else that differs makes this a new experiment rather than a
# replication, which is the whole point.
for s in $SEEDS; do
  echo "true $s"
  echo "false $s"
done | xargs -P "$JOBS" -n 2 bash -c '
  OMP_NUM_THREADS=1 python train_agent.py \
    --task cartpole --quantum "$0" --seed "$1" \
    --steps 80000 --partial_obs true \
    --prior_arch transformer --prior_context 4 \
    > '"$LOGS"'/rep_po_"$0"_s"$1".log 2>&1
  echo "  done quantum=$0 seed=$1 rc=$?"
'

printf '\n===== re-evaluating at 50 episodes =====\n%s\n' "$(date '+%F %T')"
python -u reevaluate.py --episodes 50 --out reevaluation_replication.json \
  2>&1 | tee "$LOGS/reevaluate.txt" | tail -40

printf '\n===== DONE =====\n%s\n' "$(date '+%F %T')"
echo "logs:   $LOGS"
echo "scores: $P2/reevaluation_replication.json"
echo
echo "The paired comparison printed above now covers seeds 0-15 in one"
echo "cell. Report the two halves separately as well as pooled: a"
echo "replication that is only visible after pooling is not a"
echo "replication."
