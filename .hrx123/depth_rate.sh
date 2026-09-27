#!/usr/bin/env bash
# depth_rate.sh <tag> <port> <env> <n_runs> [server args...]
# Fires N greedy completions at ~2113 ctx against one server and reports how many
# differ from the CPU reference string.
set -u
source "$(dirname "$0")/env.sh"
D="$(dirname "$0")"; TAG="$1"; PORT="$2"; ENVV="$3"; N="${4:-8}"; shift 4
LOG="$D/logs/depth-rate-$TAG.log"; : > "$LOG"
REF=$' Paris. \n\nThe quick brown fox jumps over the lazy dog near the riverbank'
say() { echo "$@" | tee -a "$LOG"; }
say "=== depth rate $TAG @ $(date -Is) n=$N env=[$ENVV] ==="
say "llama.cpp: $(git -C "$HOME/1bit-engine/third_party/llama.cpp" rev-parse --short HEAD)  seed-line: $(grep -c 'lane_expert_base_i32' "$HOME/1bit-engine/third_party/llama.cpp/ggml/src/ggml-hrx/kernel-corpus/kernels/qwen_moe/qwen3_moe/router_top8_f32.loom")"
env $ENVV "$HRX123_BIN/llama-server" -m "$HRX123_MODEL" --host 127.0.0.1 --port "$PORT" -c 8192 -np 1 -t 8 "$@" \
    > "$D/logs/server-$TAG.log" 2>&1 &
SP=$!
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
same=0; diff=0
for i in $(seq 1 "$N"); do
  curl -s --max-time 900 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    --data-binary @/tmp/longprompt.json > "$D/logs/out-$TAG-$i.json" 2>&1
  c=$(python3 -c "import json;print(json.load(open('$D/logs/out-$TAG-$i.json')).get('content'))" 2>/dev/null)
  if [ "$c" = "$REF" ]; then same=$((same+1)); v=REF; else diff=$((diff+1)); v=DIVERGENT; fi
  say "  run $i/$N -> $v : $(printf '%s' "$c" | head -c 90)"
done
say "  TOTAL $TAG: $same/$N match reference, $diff/$N divergent"
kill $SP 2>/dev/null; wait $SP 2>/dev/null
