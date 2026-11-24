#!/bin/bash
# Progress of run_remaining.sh at a glance.
#
#     bash ~/Documents/PhD/qtwm/status.sh
#
# Safe to run as often as you like; it only reads.

QTWM="$HOME/Documents/PhD/qtwm"
LOGS="$QTWM/logs_remaining"
LOCK="$LOGS/.run_remaining.lock"

printf '\n== driver ==\n'
if [ -f "$LOCK" ]; then
  PID=$(cat "$LOCK")
  if kill -0 "$PID" 2>/dev/null; then
    ps -p "$PID" -o pid=,etime=,%cpu= | \
      awk '{printf "  RUNNING  pid %s  elapsed %s\n", $1, $2}'
  else
    printf '  NOT RUNNING (stale lock from pid %s)\n' "$PID"
  fi
else
  printf '  NOT RUNNING (no lock file)\n'
fi

printf '  last driver line: %s\n' "$(tail -1 "$LOGS/driver_resume.log" 2>/dev/null)"

# ---------------------------------------------------------------- Phase 2
# Each training log prints a line per evaluation, every 2,500 of 80,000
# steps, so the last line of a log is the live position of that run. A
# log whose last line is unchanged across two checks several minutes
# apart is a stuck run, which is the thing worth catching early.
printf '\n== phase 2 training (80,000 steps each) ==\n'
shopt -s nullglob
found=0
for f in "$LOGS"/p2_tf_*.log; do
  found=1
  name=$(basename "$f" .log)
  last=$(grep -E '^step ' "$f" 2>/dev/null | tail -1)
  if [ -z "$last" ]; then
    err=$(tail -1 "$f" 2>/dev/null)
    case "$err" in
      *REFUSING*) printf '  %-22s already complete\n' "$name" ;;
      *)          printf '  %-22s starting / %s\n' "$name" "${err:0:40}" ;;
    esac
    continue
  fi
  step=$(echo "$last" | awk '{print $2}')
  ev=$(echo "$last"  | awk '{print $5}')
  rate=$(echo "$last" | grep -oE '[0-9.]+ steps/s' | awk '{print $1}')
  pct=$(( step * 100 / 80000 ))
  eta="?"
  if [ -n "$rate" ]; then
    eta=$(awk -v s="$step" -v r="$rate" \
          'BEGIN{ if (r>0) printf "%.1fh", (80000-s)/r/3600; else print "?" }')
  fi
  bar=$(awk -v p="$pct" 'BEGIN{n=int(p/5); s=""; for(i=0;i<n;i++)s=s"#";
                               for(i=n;i<20;i++)s=s"."; print s}')
  printf '  %-22s [%s] %3d%%  eval %-7s eta %s\n' "$name" "$bar" "$pct" "$ev" "$eta"
done
[ "$found" = 0 ] && printf '  no phase 2 logs yet\n'

printf '\n== completed runs on disk ==\n'
for pat in "archtransformer_K4_*po_*:partial obs, transformer K4" \
           "archtransformer_K4_*full*:full obs, transformer K4"; do
  glob=${pat%%:*}; label=${pat##*:}
  n=0
  for d in "$QTWM"/Phase_2_Closed_Loop/runs/*${glob#*_}*/; do
    [ -f "$d/result.json" ] && n=$((n+1))
  done
  printf '  %-28s %d\n' "$label" "$n"
done

printf '\n== anything stuck? ==\n'
stale=$(find "$LOGS" -name 'p2_tf_*.log' -mmin +45 -newermt '-3 days' 2>/dev/null)
if [ -n "$stale" ]; then
  printf '  no output for >45 min:\n'
  echo "$stale" | sed 's|^|    |'
  printf '  (normal only if that run finished; check against the list above)\n'
else
  printf '  all active logs written within the last 45 min\n'
fi
printf '\n'
