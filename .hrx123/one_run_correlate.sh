#!/usr/bin/env bash
# one_run_correlate.sh <tag>
set -u
source "$(dirname "$0")/env.sh"
TAG="${1:-corr}"
OUT="$(dirname "$0")/logs/$TAG"; rm -rf "$OUT"; mkdir -p "$OUT"
GGML_HRX_TRACE_ALLOC=1 GGML_HRX_TRACE_ARENA=1 "$HRX123_BIN/llama-bench" -m "$HRX123_MODEL" -dev HRX0 -p 0 -n 8 -d 2100 -r 1 > "$OUT/bench.log" 2>&1 &
sleep 0.6
PID="$(pgrep -n -f "$HRX123_MODEL")"
echo "pid=$PID"
i=0
while kill -0 "$PID" 2>/dev/null; do
  cp "/proc/$PID/maps" "$OUT/maps.$i" 2>/dev/null
  i=$((i+1)); sleep 0.1
done
wait
echo "snapshots=$i" > "$OUT/meta"
grep -cE "MEMORY_FAULT|failed to decode" "$OUT/bench.log" >> "$OUT/meta"
wait
journalctl --since "3 min ago" 2>/dev/null | awk -v p="Process llama-bench pid $PID" '$0 ~ p {f=1; next} f&&/page starting at address/{for(i=1;i<=NF;i++) if($i ~ /^0x/) print $i; f=0}' | sort -u > "$OUT/faults"
echo "faults: $(wc -l < "$OUT/faults")"
echo "$PID" > "$OUT/pid"
