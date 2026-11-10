#!/usr/bin/env bash
# Compare the model's own token distribution on the HOST GEMM and (optionally) the NPU, same container, same prompt,
# greedy, 1 token, top-5 logprobs. A close distribution with a different argmax = precision; a wild one = a real bug.
#   ./w3b-dist.sh host         (NPU_INFER_HOST_GEMM=1, no device, no ticket)
#   ./w3b-dist.sh device <dir> (uses the kernel dir given, must run inside npu-hold)
set -uo pipefail
MODE=${1:-host}; KDIR=${2:-$HOME/ogaload/oga13/kernels-2048-6144-b8}
B=$HOME/wt/zha-main/build-npu/1bit
M=$HOME/scratch-packs/olmo1b-b8-container
PORT=18098; [ "$MODE" = device ] && PORT=18099
export ONEBIT_NPU_KERNELS=$KDIR
ENV=""
[ "$MODE" = host ] && ENV="NPU_INFER_HOST_GEMM=1"
LOG=/home/bcloud/.cache/scratch-pi-rlk/w3b-dist-$MODE-$$.log
env $ENV "$B" serve -m "$M" --device npu --port $PORT --host 127.0.0.1 --ctx-size 2048 >"$LOG" 2>&1 &
SP=$!; trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for _ in $(seq 1 150); do
  kill -0 $SP 2>/dev/null || { echo "serve died:"; tail -8 "$LOG"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 1
done
for prompt in "The capital of France is" "The first person to walk on the Moon was"; do
  echo "=== [$MODE] prompt: $prompt"
  curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys;print(json.dumps({'prompt':sys.argv[1],'max_tokens':1,'temperature':0,'logprobs':True}))" "$prompt")" \
  | python3 -c '
import json,sys
raw=sys.stdin.read()
try: d=json.loads(raw)
except Exception: print("  not json:",raw[:300]); raise SystemExit
if "choices" not in d:
    print("  ERROR RESPONSE:",json.dumps(d)[:300]); raise SystemExit
ch=d["choices"][0]
lp=ch.get("logprobs") or {}
top=lp.get("top_logprobs") or []
if top:
    for t in top[0].items(): print(f"   {t[0]!r:24s} {t[1]:+.4f}")
else:
    print("   text:",repr(ch.get("text"))); print("   logprobs:",json.dumps(lp)[:200])
'
done
