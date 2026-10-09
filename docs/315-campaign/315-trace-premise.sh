#!/usr/bin/env bash
# engine#315 premise check: does a host-visible buffer back the logits?
#
# This is NOT a campaign arm — no regime claim is made. The buffer trace fires at allocation
# (model load), which is independent of GTT pressure or the fault rate, so it needs no neighbours.
# That matters because the peer #315 session is currently holding the regime band (observed at
# 103 GiB), which makes any in-regime composition with my own neighbours impossible.
set -uo pipefail
B=/home/bcloud/wt/315instr3-build/bin
M=/home/bcloud/models/GLM-4.7-Flash-Q4_K_M.gguf
LOG=/tmp/315trace.log
IREE=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
G=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
echo "gtt_before=$(( $(cat "$G") / 1073741824 )) GiB  (peer may be resident; foreign servers are not mine)"
echo "mode: GGML_HRX_USE_UNIFIED_MEMORY=${UNIFIED:-unset}  (this run answers whether the logits buffer becomes coherent)"
LOG=${LOG_OVERRIDE:-$LOG}

systemd-run --user --scope -q -p MemoryMax=60G --unit=315trace \
    --setenv=IREE_HAL_AMDGPU_LIBHSA_PATH="$IREE" \
    --setenv=LD_LIBRARY_PATH="$B" \
    --setenv=GGML_HRX_BUFFER_TRACE=1 \
    ${UNIFIED:+--setenv=GGML_HRX_USE_UNIFIED_MEMORY=$UNIFIED} \
    "$B/llama-server" -m "$M" -dev HRX0 -ngl 99 -fa on --host 127.0.0.1 --port 19911 -c 2048 -np 1 --no-webui \
    > "$LOG" 2>&1 &
for _t in $(seq 1 90); do
    grep -q "\[hrx-buf\]" "$LOG" 2>/dev/null && break
    sleep 2
done
echo "trace lines after load: $(grep -c '\[hrx-buf\]' "$LOG" 2>/dev/null || echo 0)"

# The logits buffer is not allocated at load — it appears once the graph runs. So decode before reading
# the trace, otherwise only the model weights are ever observed (which is exactly what the first run of
# this script did, and why it was inconclusive).
for _t in $(seq 1 150); do curl -sf localhost:19911/health >/dev/null 2>&1 && break; sleep 2; done
if curl -sf localhost:19911/health >/dev/null 2>&1; then
    echo "decoding to force the graph/logits allocations"
    for _r in 1 2 3; do
        curl -s localhost:19911/completion -H 'Content-Type: application/json' \
            -d '{"prompt":"Count: one two three four five six seven eight","n_predict":16,"temperature":0}' \
            > /dev/null 2>&1
        sleep 2
    done
else
    echo "server did not become healthy; the trace below is load-time only"
fi
echo "trace lines after decode: $(grep -c '\[hrx-buf\]' "$LOG" 2>/dev/null || echo 0)"
echo "=== visibility / coherence pairs (count) ==="
grep -o 'host_visible=[01] direct_host_binding=[01] memory_type=0x[0-9a-f]*' "$LOG" 2>/dev/null | sort | uniq -c | sort -rn | head -6
echo "=== the logits size specifically, with its flags ==="
for sz in 619520 1239040 4956160; do
    n=$(grep -c "size=$sz" "$LOG" 2>/dev/null || echo 0)
    [ "$n" != "0" ] && { echo "  size=$sz count=$n :"; grep "size=$sz" "$LOG" | head -2 | cut -c1-160 | sed 's/^/    /'; }
done
echo "=== buffer type names seen ==="
grep -o 'type=[^ ]*' "$LOG" 2>/dev/null | sort | uniq -c | sort -rn | head -6
echo "=== allocations by size (largest 6) ==="
grep -o 'memory_type=0x[0-9a-f]* host_visible=[01] direct_host_binding=[01]' "$LOG" 2>/dev/null | head -2
grep -o 'size=[0-9]*' "$LOG" 2>/dev/null | cut -d= -f2 | sort -rn | head -6 | tr '\n' ' '; echo
echo "=== the logits size specifically (154880 x f32 = 619520; also check f16 309760, and vocab-ish) ==="
for sz in 619520 309760 154880 1239040; do
    n=$(grep -c "size=$sz" "$LOG" 2>/dev/null || echo 0)
    [ "$n" != "0" ] && { echo "  size=$sz count=$n :"; grep "size=$sz" "$LOG" | head -2 | cut -c1-150 | sed 's/^/    /'; }
done
echo "=== load errors? ==="
grep -iE "error|failed|abort" "$LOG" 2>/dev/null | grep -viE "sync|fallback" | head -4 | cut -c1-140 | sed 's/^/  /'
systemctl --user stop 315trace 2>/dev/null
sleep 2
for P in $(pgrep -f "port 19911" 2>/dev/null); do kill -TERM "$P" 2>/dev/null; done
sleep 4
echo "teardown: servers_left=$(pgrep -c llama-server || echo 0) gtt_after=$(( $(cat "$G") / 1073741824 )) GiB"
echo "315TRACE_DONE"
