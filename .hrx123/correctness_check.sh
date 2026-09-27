#!/usr/bin/env bash
# correctness_check.sh — compare deterministic greedy output HRX0 vs CPU (same build)
set -u
source "$(dirname "$0")/env.sh"
BIN="$HRX123_BIN"
MODEL="$HRX123_MODEL"
PROMPT='Question: What is the capital of France? Answer in one word.\nAnswer:'

run_one() {
  local tag="$1"; shift
  local port="$1"; shift
  "$BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port "$port" -c 4096 -np 1 "$@" \
      > "$(dirname "$0")/logs/server-$tag.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"n_predict\":24,\"temperature\":0,\"seed\":1234,\"cache_prompt\":false}" \
    > "$(dirname "$0")/logs/out-$tag.json" 2>&1
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  python3 -c "import json,sys; d=json.load(open('$(dirname "$0")/logs/out-$tag.json')); print('$tag:', repr(d.get('content')))" 2>/dev/null \
    || echo "$tag: FAILED ($(head -c 120 "$(dirname "$0")/logs/out-$tag.json"))"
}

run_one hrx 18081 --device HRX0 -ngl 99
run_one cpu 18082 -ngl 0
