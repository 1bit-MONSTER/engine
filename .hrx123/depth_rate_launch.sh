#!/usr/bin/env bash
# depth_rate_launch.sh <tag> <n_launches> <env> [server args...]
# Fresh server process per sample (so each sample gets a new address layout),
# one greedy completion at ~2113 ctx, compared against the CPU reference string.
set -u
source "$(dirname "$0")/env.sh"
D="$(dirname "$0")"; TAG="$1"; N="$2"; ENVV="$3"; shift 3
LOG="$D/logs/launch-rate-$TAG.log"; : > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }
say "=== launch rate $TAG @ $(date -Is) launches=$N env=[$ENVV] ==="
say "rev: $(git -C "$HOME/1bit-engine/third_party/llama.cpp" rev-parse --short HEAD)  align: $(grep -oE 'partial_scalar_bytes, [0-9]+' "$HOME/1bit-engine/third_party/llama.cpp/ggml/src/ggml-hrx/dispatch_registration/common/dispatch-flash-attention.cpp" | sort -u | tr '\n' ' ')"
good=0; bad=0
for i in $(seq 1 "$N"); do
  port=$((19000 + i))
  env $ENVV "$HRX123_BIN/llama-server" -m "$HRX123_MODEL" --host 127.0.0.1 --port "$port" -c 8192 -np 1 -t 8 "$@" \
      > "$D/logs/srv-$TAG-$i.log" 2>&1 &
  sp=$!
  for k in $(seq 1 240); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
  curl -s --max-time 900 "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
    --data-binary @/tmp/longprompt.json > "$D/logs/o-$TAG-$i.json" 2>&1
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  c=$(python3 -c "import json;print(json.load(open('$D/logs/o-$TAG-$i.json')).get('content'))" 2>/dev/null)
  if [ "$c" = $' Paris. \n\nThe quick brown fox jumps over the lazy dog near the riverbank' ]; then
    good=$((good+1)); say "  launch $i/$N -> OK"
  else
    bad=$((bad+1)); say "  launch $i/$N -> WRONG : $(printf '%s' "$c" | head -c 60)"
  fi
done
say "  TOTAL $TAG: $good/$N correct, $bad/$N wrong"
