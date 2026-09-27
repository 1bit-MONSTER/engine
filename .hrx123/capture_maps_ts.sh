#!/usr/bin/env bash
# capture_maps_ts.sh <tag> [depth]
# Timestamped maps polling + fault-time detection.
set -u
source "$(dirname "$0")/env.sh"
TAG="${1:?tag}"; DEPTH="${2:-2100}"
OUT="$(dirname "$0")/logs/maps-$TAG"
rm -rf "$OUT"; mkdir -p "$OUT"

$HRX123_BIN/llama-bench -m "$HRX123_MODEL" -dev HRX0 -p 0 -n 8 -d "$DEPTH" -r 1 > "$OUT/bench.log" 2>&1 &
PID=$!
echo "pid=$PID tag=$TAG"
fault_ts=""
i=0
while kill -0 "$PID" 2>/dev/null; do
  cp "/proc/$PID/maps" "$OUT/maps.$i" 2>/dev/null
  printf '%s %d\n' "$(date +%s.%N)" "$i" >> "$OUT/snapshots.ts"
  if [ -z "$fault_ts" ] && grep -qiE "failed to decode|HSA_STATUS_ERROR_MEMORY_FAULT" "$OUT/bench.log" 2>/dev/null; then
    fault_ts="$(date +%s.%N)"
    echo "$fault_ts" > "$OUT/fault.ts"
  fi
  i=$((i+1)); sleep 0.12
done
wait "$PID"; rc=$?
echo "rc=$rc snapshots=$i fault_ts=${fault_ts:-NONE}"
echo "=== snapshot just before fault ==="
awk -v f="$fault_ts" 'BEGIN{f=(f==""?1e18:f+0)} $1<=f {l=$0} END{print l}' "$OUT/snapshots.ts"
