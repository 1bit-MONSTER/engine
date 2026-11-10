#!/usr/bin/env bash
# Per-layer residual dump (NPU_INFER_DUMP_DIR) for ONE greedy token, host GEMM or NPU kernels.
# The engine writes layer_<l>.f32 (M x H floats, row-major) per layer; cosine-compare the two dirs to localise
# the first layer where the two GEMM paths disagree. Usage: w3b-layerdump.sh host|device <kerneldir> <outdir>
set -uo pipefail
MODE=$1; KDIR=${2:-$HOME/ogaload/oga13/kernels-2048-6144-b8}; DD=$3
B=$HOME/wt/zha-main/build-npu/1bit; M=${M:-$HOME/scratch-packs/olmo1b-b8-container}   # overridable: the family under test
PORT=18098; [ "$MODE" = device ] && PORT=18099
export ONEBIT_NPU_KERNELS=$KDIR
E=""; [ "$MODE" = host ] && E="NPU_INFER_HOST_GEMM=1"
rm -rf "$DD"; mkdir -p "$DD"
LOG=/home/bcloud/.cache/scratch-pi-rlk/ld-$MODE-$(basename "$M").log   # per model: a fixed name let the second family overwrite the first's trace
env $E NPU_INFER_DUMP_DIR="$DD" NPU_INFER_DUMP_LAYERS=1 "$B" serve -m "$M" --device npu --port $PORT --host 127.0.0.1 --ctx-size 2048 >"$LOG" 2>&1 &
SP=$!; trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  kill -0 $SP 2>/dev/null || { echo "serve died:"; tail -6 "$LOG"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break; sleep 1
done
curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":"The first person to walk on the Moon was","max_tokens":1,"temperature":0}' >/dev/null
sleep 1
echo "[$MODE] $(ls "$DD"/layer_*.f32 2>/dev/null | wc -l) layer files in $DD; $(grep -c '\[LB\]' "$LOG" 2>/dev/null || echo 0) [LB] lines"
grep -E '\[LB\] m=0 ' "$LOG" 2>/dev/null | head -3
