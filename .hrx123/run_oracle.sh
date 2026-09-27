#!/usr/bin/env bash
# run_oracle.sh <depth> <runs> <tag>
# Runs llama-bench N times at <depth>, counts HSA faults, logs to logs/<tag>.log
set -u
source "$(dirname "$0")/env.sh"
DEPTH="${1:?depth}"
RUNS="${2:?runs}"
TAG="${3:?tag}"
LOGDIR="$(dirname "$0")/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/$TAG.log"
: > "$LOG"

echo "=== oracle $TAG :: depth=$DEPTH runs=$RUNS $(date -Is) ===" | tee -a "$LOG"
echo "quiet: KFD=[$(fuser /dev/kfd 2>/dev/null || echo none)] render=[$(fuser /dev/dri/renderD128 2>/dev/null || echo none)]" | tee -a "$LOG"

faults=0
for i in $(seq 1 "$RUNS"); do
  out="$($HRX123_BIN/llama-bench -m "$HRX123_MODEL" -dev HRX0 -p 0 -n 8 -d "$DEPTH" -r 1 2>&1)"
  rc=$?
  if printf '%s' "$out" | grep -qiE "HSA_STATUS_ERROR_MEMORY_FAULT|memory access fault|failed to decode|Queue error"; then
    verdict=FAULT; faults=$((faults+1))
  elif [ $rc -ne 0 ]; then
    verdict="EXIT($rc)"
  else
    verdict=OK
  fi
  ts="$(date -Is)"
  printf 'run %d/%d  %s  rc=%d  %s\n' "$i" "$RUNS" "$verdict" "$rc" "$ts" | tee -a "$LOG"
  printf '%s\n' "$out" >> "$LOG"
done
echo "=== TOTAL $TAG: $faults/$RUNS faults ===" | tee -a "$LOG"
