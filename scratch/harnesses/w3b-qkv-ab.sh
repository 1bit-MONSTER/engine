#!/usr/bin/env bash
# QKV tiling A/B on the device, inside ONE npu-hold ticket: identical proven O/GU/D (llama-b8, content-identical to
# mine), only the QKV ELF changes. c8 = the generator's auto-pick (cols=8, 937,736 B); c6 = forced cols=6 (1,000,424 B).
# One greedy request per configuration; the text is the datum.
set -uo pipefail
B=$HOME/wt/zha-main/build-npu/1bit
M=$HOME/scratch-packs/olmo1b-b8-container
PORT=18099
P='{"prompt":"The first person to walk on the Moon was","max_tokens":32,"temperature":0}'
for v in c8 c6; do
    K=$HOME/ogaload/oga13/kernels-2048-6144-iso-$v
    LOG=/home/bcloud/.cache/scratch-pi-rlk/w3b-ab-$v-$$.log
    echo "=== QKV variant $v: $(stat -c%s $K/full_i8_QKV_K2048_N6144_b8.elf) B"
    ONEBIT_NPU_KERNELS=$K "$B" serve -m "$M" --device npu --port $PORT --host 127.0.0.1 --ctx-size 2048 >"$LOG" 2>&1 &
    SP=$!
    ready=0
    for _ in $(seq 1 90); do
        kill -0 "$SP" 2>/dev/null || break
        curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && { ready=1; break; }
        sleep 1
    done
    if [ "$ready" = 1 ]; then
        grep -m1 'QKV M=' "$LOG" | sed 's/^/   /'
        resp=$(curl -s --max-time 180 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' -d "$P")
        python3 - "$resp" "$v" <<'PY'
import json,sys
d=json.loads(sys.argv[1]); ch=d['choices'][0]
print(f"   [{sys.argv[2]}] {ch.get('text')!r}")
PY
    else
        echo "   serve never became ready; log:"; tail -6 "$LOG" | sed 's/^/     /'
    fi
    kill $SP 2>/dev/null; wait $SP 2>/dev/null
    sleep 3
done
echo "=== reference text for this prompt (host GEMM): ' Neil Armstrong. He and his Apollo 11 crewmates, Buzz Aldrin and Michael Collins, were the first to set foot on the lunar surface. They landed on'"
