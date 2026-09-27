#!/usr/bin/env bash
# capture_maps.sh <tag> [depth]
# Runs one llama-bench, polls /proc/PID/maps into logs/maps-<tag>/, then
# prints the kernel fault lines for that PID window.
set -u
source "$(dirname "$0")/env.sh"
TAG="${1:?tag}"
DEPTH="${2:-2100}"
OUT="$(dirname "$0")/logs/maps-$TAG"
rm -rf "$OUT"; mkdir -p "$OUT"

$HRX123_BIN/llama-bench -m "$HRX123_MODEL" -dev HRX0 -p 0 -n 8 -d "$DEPTH" -r 1 > "$OUT/bench.log" 2>&1 &
PID=$!
echo "llama-bench pid=$PID depth=$DEPTH"
i=0
while kill -0 "$PID" 2>/dev/null; do
  cp "/proc/$PID/maps" "$OUT/maps.$i" 2>/dev/null
  # also capture the mapping list of the GPU heap-ish regions
  i=$((i+1))
  sleep 0.15
done
wait "$PID"; rc=$?
echo "rc=$rc  map snapshots=$i"
echo "$PID" > "$OUT/pid"
wc -l < "$OUT/bench.log" | sed 's/^/bench.log lines: /'
grep -iE "HSA_STATUS|fault|decode" "$OUT/bench.log" | head
