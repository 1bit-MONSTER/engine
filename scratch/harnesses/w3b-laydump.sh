#!/usr/bin/env bash
# Per-layer mean|.| / absmax trace (NPU_INFER_DUMP_LAYERS) for one greedy token, host GEMM or NPU kernels.
# Usage: w3b-laydump.sh host|device <kerneldir> <outfile>
set -uo pipefail
MODE=$1; KDIR=${2:-$HOME/ogaload/oga13/kernels-2048-6144-b8}; OUT=${3:-/home/bcloud/.cache/scratch-pi-rlk/lay-$1.txt}
B=$HOME/wt/zha-main/build-npu/1bit; M=$HOME/scratch-packs/olmo1b-b8-container
PORT=18098; [ "$MODE" = device ] && PORT=18099
export ONEBIT_NPU_KERNELS=$KDIR
E=""; [ "$MODE" = host ] && E="NPU_INFER_HOST_GEMM=1"
LOG=/home/bcloud/.cache/scratch-pi-rlk/lay-$MODE.log
env $E NPU_INFER_DUMP_LAYERS=1 "$B" serve -m "$M" --device npu --port $PORT --host 127.0.0.1 --ctx-size 2048 >"$LOG" 2>&1 &
SP=$!; trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for _ in $(seq 1 150); do
  kill -0 $SP 2>/dev/null || { echo "serve died:"; tail -8 "$LOG"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break; sleep 1
done
curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":"The first person to walk on the Moon was","max_tokens":1,"temperature":0}' >/dev/null
sleep 1
grep -E '^\s*\[G\] ' "$LOG" > "$OUT" || true
echo "[$MODE] $(wc -l < "$OUT") [G] lines -> $OUT"
head -4 "$OUT"
