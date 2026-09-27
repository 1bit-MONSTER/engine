#!/usr/bin/env bash
# capture_maps2.sh <tag>
# Polls /proc/PID/maps at high frequency and keeps the greatest set of mappings seen,
# to find what (if anything) is mapped just after the huge weights allocation.
set -u
source "$(dirname "$0")/env.sh"
TAG="${1:-maps2}"
OUT="$(dirname "$0")/logs/$TAG"; rm -rf "$OUT"; mkdir -p "$OUT"
$HRX123_BIN/llama-bench -m "$HRX123_MODEL" -dev HRX0 -p 0 -n 8 -d 2100 -r 1 > "$OUT/bench.log" 2>&1 &
sleep 0.4
PID="$(pgrep -n -f "$HRX123_MODEL")"
echo "pid=$PID"
i=0
while kill -0 "$PID" 2>/dev/null; do
  cp "/proc/$PID/maps" "$OUT/maps.$i" 2>/dev/null && cat "/proc/$PID/maps" >> "$OUT/union.maps"
  i=$((i+1)); sleep 0.08
done
wait; echo "snapshots=$i"
sort -u "$OUT/union.maps" > "$OUT/union.sorted"
wc -l < "$OUT/union.sorted"
