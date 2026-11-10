#!/usr/bin/env bash
# Copy-task probe: a correct pipeline reproduces a supplied string; a broken GEMM does not. Host or device.
# Usage: w3b-copyprobe.sh host|device <kerneldir>
set -uo pipefail
MODE=$1; KDIR=${2:-$HOME/ogaload/oga13/kernels-2048-6144-b8}
B=$HOME/wt/zha-main/build-npu/1bit; M=$HOME/scratch-packs/olmo1b-b8-container
PORT=18098; [ "$MODE" = device ] && PORT=18099
export ONEBIT_NPU_KERNELS=$KDIR
E=""; [ "$MODE" = host ] && E="NPU_INFER_HOST_GEMM=1"
LOG=/home/bcloud/.cache/scratch-pi-rlk/copy-$MODE.log
env $E "$B" serve -m "$M" --device npu --port $PORT --host 127.0.0.1 --ctx-size 2048 >"$LOG" 2>&1 &
SP=$!; trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for _ in $(seq 1 150); do
  kill -0 $SP 2>/dev/null || { echo "serve died:"; tail -6 "$LOG"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break; sleep 1
done
for p in "Repeat exactly: alpha beta gamma delta" "2, 4, 6, 8," "Water is made of hydrogen and"; do
  out=$(curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json,sys;print(json.dumps({'prompt':sys.argv[1],'max_tokens':20,'temperature':0}))" "$p")" \
        | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['choices'][0]['text'] if 'choices' in d else 'ERR '+json.dumps(d)[:120])")
  printf '  [%s] %-38s -> %s\n' "$MODE" "$p" "$(printf '%s' "$out" | tr '\n' ' ')"
done
