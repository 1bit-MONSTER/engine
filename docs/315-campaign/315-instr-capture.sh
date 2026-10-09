#!/usr/bin/env bash
# engine#315: capture an in-regime arm with the writeback fingerprint diagnostic enabled.
#
# Runs the pre-registered shape set (via 315-detector.py, so Face B is measured too) against the
# instrumented backend, with GGML_HRX_WRITEBACK_FINGERPRINT=1, and collects the [hrx-wb-fp] records
# plus both failure faces. Set FP=0 for the flag-off control that must behave identically.
#
# Usage: 315-instr-capture.sh <bin_dir> <neighbours> <neigh_ctx> <tag> [fp=1]
set -uo pipefail
BIN=${1:?bin dir}; NB=${2:-2}; NCTX=${3:-2048}; TAG=${4:-315instr}; FP=${5:-1}
DET=$(dirname "$0")/315-detector.py
M=/home/bcloud/models/GLM-4.7-Flash-Q4_K_M.gguf
PORT=""
G=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
gtt() { echo $(( $(cat $G) / 1073741824 )); }
O=$HOME/wt/rt-det/$TAG-$(basename $M .gguf)
rm -rf "$O"; mkdir -p "$O"

echo "capture start $(date -Is) bin=$BIN fp=$FP gtt_before=$(gtt)G"
"$BIN/llama-server" --version | sed 's/^/  /'
for P in $(pgrep -f llama-server 2>/dev/null); do
  echo "  foreign pid=$P $(tr '\0' ' ' < /proc/$P/cmdline | cut -c1-90)"
done

for i in $(seq 1 $NB); do
  systemd-run --user --scope -q -p MemoryMax=40G --unit=${TAG}-neigh-$i \
    "$BIN/llama-server" -m "$M" -dev HRX0 -ngl 99 -fa on -c $NCTX --host 127.0.0.1 --port $((20250+i)) \
    > "$O/neigh-$i.log" 2>&1 &
done
for i in $(seq 1 $NB); do for t in $(seq 1 300); do curl -sf localhost:$((20250+i))/health >/dev/null 2>&1 && break; sleep 1; done; done
echo "neighbours up gtt=$(gtt)G"

teardown() {
  for i in $(seq 1 $NB); do
    systemctl --user stop ${TAG}-neigh-$i 2>/dev/null
    systemctl --user kill -s TERM ${TAG}-neigh-$i 2>/dev/null
  done
  sleep 3
  for P in $(pgrep -f "llama-server" 2>/dev/null); do
    C=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null)
    case "$C" in *"port 2025"*) kill -TERM $P 2>/dev/null ;; esac
    if [ -n "$PORT" ]; then case "$C" in *"port $PORT"*) kill -TERM $P 2>/dev/null ;; esac; fi
  done
  sleep 5
  echo "teardown $(date -Is) servers_left=$(pgrep -c llama-server || echo 0) gtt_after=$(gtt)G"
}
trap teardown EXIT

PORT=$((19300 + RANDOM % 400))
SETENV=(--setenv=IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1)
[ "$FP" = "1" ] && SETENV+=(--setenv=GGML_HRX_WRITEBACK_FINGERPRINT=1)
systemd-run --user --scope -q -p MemoryMax=60G --unit=${TAG}-arm "${SETENV[@]}" \
  "$BIN/llama-server" -m "$M" -dev HRX0 -ngl 99 -fa on --host 127.0.0.1 --port $PORT -c 8192 -np 1 --no-webui \
  > "$O/server.log" 2>&1 &
UP=0; for t in $(seq 1 300); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { UP=1; break; }; sleep 1; done
G0=$(gtt); echo "gtt_start ${G0}GiB fp=$FP" > "$O/run.log"
echo "arm up=$UP gtt_start=${G0}G port=$PORT"

if [ "$UP" = "1" ]; then
  python3 "$DET" "$PORT" "$O" 50 > "$O/detector.log" 2>&1
  tail -1 "$O/detector.log" | sed 's/^/  /'
fi

echo "--- diagnostic records ---"
grep -c "\[hrx-wb-fp\]" "$O/server.log" 2>/dev/null | xargs echo "  hrx-wb-fp lines:"
grep -m3 "\[hrx-wb-fp\]" "$O/server.log" 2>/dev/null | cut -c1-240 | sed 's/^/  /'
echo "--- faces ---"
grep -c "HRX returned NaN logits" "$O/server.log" 2>/dev/null | xargs echo "  faceA nan:"
grep -iE "HSA_STATUS_ERROR|Queue error|wait for HRX graph replay commands failed" "$O/server.log" 2>/dev/null \
  | grep -vcE "hrx_allocator_allocate_buffer|allocate HRX host staging buffer" | xargs echo "  faceA hsa:"
grep -cE "hrx_allocator_allocate_buffer.*failed" "$O/server.log" 2>/dev/null | xargs echo "  startup alloc failures:"
if [ -f "$O/detector.json" ]; then
  python3 - "$O/detector.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  faceB utf8={d['utf8_failures']} parse={d['parse_failures']} http={d['http_failures']} "
      f"transport={d['transport_failures']} req={d['requests']}/66 truncated={d['truncated']}")
print(f"  first_failure={d['first_failure']!r}")
PY
fi
echo "CAPTURE_DONE tag=$TAG out=$O"
