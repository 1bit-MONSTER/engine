#!/usr/bin/env bash
# Decisive correctness check for the 2048x6144 family: NPU_INFER_HOST_GEMM=1 runs the SAME container, geometry and
# quantisation through the forward's HOST GEMM instead of the NPU ELFs (q4nx_forward.cpp: the device is not even opened),
# so the two texts are directly comparable and no device ticket is needed.
# If the host GEMM says the same thing as the NPU, the text is the model/quantisation; if it says something else,
# the NPU kernels and the host path disagree and THAT is the finding.
set -uo pipefail
B=$HOME/wt/zha-main/build-npu/1bit
M=$HOME/scratch-packs/olmo1b-b8-container
PORT=18098     # never 18097
LOG=/home/bcloud/.cache/scratch-pi-rlk/w3b-cpu-$$.log
P='{"prompt":"The first person to walk on the Moon was","max_tokens":32,"temperature":0}'
export ONEBIT_NPU_KERNELS=$HOME/ogaload/oga13/kernels-2048-6144-b8   # the host arm never loads these; serve validates the dir exists
NPU_INFER_HOST_GEMM=1 "$B" serve -m "$M" --device npu --port $PORT --host 127.0.0.1 --ctx-size 2048 >"$LOG" 2>&1 &
SP=$!; trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for _ in $(seq 1 120); do
  kill -0 $SP 2>/dev/null || { echo "CPU SERVE DIED:"; tail -20 "$LOG"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 1
done
echo "cpu serve up; device line: $(grep -m1 -iE 'device|backend' "$LOG" | cut -c1-110)"
for r in 1 2; do
  t0=$(date +%s.%N); resp=$(curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' -d "$P"); t1=$(date +%s.%N)
  python3 - "$resp" "$(python3 -c "print(f'{$t1-$t0:.2f}')")" "$r" <<'PY'
import json,sys
raw,el,r=sys.argv[1],float(sys.argv[2]),sys.argv[3]
d=json.loads(raw); ch=d['choices'][0]; u=d.get('usage',{})
txt=ch.get('text') if ch.get('text') is not None else ch.get('message',{}).get('content')
print(f"cpu run {r} ({el}s): {txt!r}")
ct=u.get('completion_tokens') or 0
if ct: print(f"   cpu decode: {ct/el:.2f} tok/s")
PY
done
