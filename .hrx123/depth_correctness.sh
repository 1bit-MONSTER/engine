#!/usr/bin/env bash
# depth_correctness.sh — greedy decode output at ~2100 ctx, multipass vs fallback vs CPU.
# Persists prompt token count, generated text, timings, and the env each server ran with.
set -u
source "$(dirname "$0")/env.sh"
BIN="$HRX123_BIN"; MODEL="$HRX123_MODEL"
D="$(dirname "$0")"; LOG="$D/logs/depth-correctness.log"; : > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }

python3 - > /tmp/longprompt.json <<'PY'
import json
para = ("The quick brown fox jumps over the lazy dog near the riverbank while the morning "
        "light spreads slowly across the valley and the birds begin to sing in the tall trees. ")
text = ("Read the following passage carefully and then answer the question at the very end.\n\n"
        + para*63 +
        "\n\nQuestion: What is the capital of France? Reply with just that one word.\nAnswer:")
print(json.dumps({"prompt": text, "n_predict": 16, "temperature": 0, "seed": 1234,
                  "cache_prompt": False, "n_probs": 0}))
PY
say "=== depth correctness @ $(date -Is) ==="
say "llama.cpp: $(git -C "$HOME/1bit-engine/third_party/llama.cpp" rev-parse --short HEAD)  alignment: $(grep -oE 'partial_scalar_bytes, [0-9]+' "$HOME/1bit-engine/third_party/llama.cpp/ggml/src/ggml-hrx/dispatch_registration/common/dispatch-flash-attention.cpp" | sort -u | tr '\n' ' ')"
say "KFD holders: [$(fuser /dev/kfd 2>/dev/null || echo none)]"

run_one() {
  local tag="$1" port="$2" envv="$3"; shift 3
  say ""
  say "--- $tag : env=[${envv:-<none>}] args=[$*] ---"
  env $envv "$BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port "$port" -c 8192 -np 1 -t 8 "$@" \
      > "$D/logs/server-$tag.log" 2>&1 &
  local pid=$!
  local ok=no
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ok=yes; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  say "  server ready: $ok"
  curl -s --max-time 1200 "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
    --data-binary @/tmp/longprompt.json > "$D/logs/out-$tag.json" 2>&1
  python3 - "$tag" "$D/logs/out-$tag.json" <<'PY' | tee -a "$LOG"
import json,sys
tag,path=sys.argv[1],sys.argv[2]
try: d=json.load(open(path))
except Exception as e:
    print(f"  {tag}: PARSE-FAIL {e} :: {open(path).read()[:200]}"); raise SystemExit
tim=d.get("timings",{})
print(f"  {tag}: tokens_evaluated={tim.get('prompt_n')} n_decoded={tim.get('predicted_n')}")
print(f"  {tag}: content={d.get('content')!r}")
PY
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}

run_one hrx-multipass 18181 "" --device HRX0 -ngl 99
run_one hrx-fallback  18182 "GGML_HRX_DISABLE_DISPATCH=flash_attention_decode_split" --device HRX0 -ngl 99
run_one cpu           18183 "" -ngl 0
say ""
say "=== done $(date -Is) ==="
