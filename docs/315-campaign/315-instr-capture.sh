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
# The HRX device only appears when this is set: without it llama-server prints
# "Available devices: (none)" and any -dev HRX0 dies with "invalid device: HRX0". Neighbours and the
# arm both need it, so it is exported here rather than passed per-process.
export IREE_HAL_AMDGPU_LIBHSA_PATH=${IREE_HAL_AMDGPU_LIBHSA_PATH:-/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1}
export LD_LIBRARY_PATH="$BIN"
source "$(dirname "$0")/315-regime.sh"
G=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
gtt() { echo $(( $(cat $G) / 1073741824 )); }
O=$HOME/wt/rt-det/$TAG-$(basename $M .gguf)
rm -rf "$O"; mkdir -p "$O"

# Refuse to hold the box lock for a run that cannot work.
regime_preflight || exit 3

echo "capture start $(date -Is) bin=$BIN fp=$FP gtt_before=$(gtt)G"
"$BIN/llama-server" --version | sed 's/^/  /'
for P in $(pgrep -f llama-server 2>/dev/null); do
  echo "  foreign pid=$P $(tr '\0' ' ' < /proc/$P/cmdline | cut -c1-90)"
done

# Serialize with other sessions, but do not block the campaign forever: after LOCK_WAIT seconds the
# arm proceeds concurrently and records that it did, so the condition is visible in the result.
if [ -z "${NOLOCK:-}" ]; then
  exec 9>"$HOME/.cache/lax-decode/box.lock"
  if flock -w "${LOCK_WAIT:-900}" 9; then
    echo "capture: box lock acquired at $(date -Is)"
  else
    echo "capture: box lock busy after ${LOCK_WAIT:-900}s — proceeding CONCURRENTLY (UNLOCKED)"
  fi
fi
regime_neighbours 1 6 "$NCTX" || { echo "capture: could not establish the regime composition — arm is invalid"; trap - EXIT; regime_teardown; exit 4; }
trap regime_teardown EXIT

PORT=$((19300 + RANDOM % 400))
REGIME_ARM_PORT=$PORT
SETENV=(--setenv=IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1)
[ "$FP" = "1" ] && SETENV+=(--setenv=GGML_HRX_WRITEBACK_FINGERPRINT=1)
# UB / EXTRA_ENV let a run reproduce the recorded *high-rate* configurations (the issue measured
# 11/12 faults with -ub 512 vs 0/12 with -ub 128), which makes mechanism discrimination cheap:
# at ~5%/arm a differential test would need dozens of arms to see a handful of faults.
ARM_UB=(); [ -n "${UB:-}" ] && ARM_UB=(-ub "$UB")
if [ -n "${EXTRA_ENV:-}" ]; then
  IFS=',' read -r -a _extra <<< "$EXTRA_ENV"
  for kv in "${_extra[@]}"; do SETENV+=(--setenv="$kv"); done
fi
echo "arm config: ub=${UB:-default} extra_env=${EXTRA_ENV:-none}"
systemd-run --user --scope -q -p MemoryMax=60G --unit=${TAG}-arm "${SETENV[@]}" \
  "$BIN/llama-server" -m "$M" -dev HRX0 -ngl 99 -fa on --host 127.0.0.1 --port $PORT -c 8192 -np 1 --no-webui \
  "${ARM_UB[@]}" \
  > "$O/server.log" 2>&1 &
UP=0; for t in $(seq 1 300); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { UP=1; break; }; sleep 1; done
G0=$(gtt); echo "gtt_start ${G0}GiB fp=$FP" > "$O/run.log"
echo "arm up=$UP gtt_start=${G0}G port=$PORT"

# Watchdog, as the recorded harness (rt-det.sh) runs it: sample GTT/temperature and abort the arm if
# it leaves the pre-registered regime (>=24 GiB growth from the arm start, or >=93 C). Without this the
# arm would be classified by its gtt_start alone while actually running hotter.
K=$(for h in /sys/class/hwmon/*; do [ "$(cat "$h/name" 2>/dev/null)" = k10temp ] && echo "$h"; done)
G0B=$(cat "$G"); : > "$O/gtt.log"
( while pgrep -f "port $PORT" >/dev/null 2>&1 || systemctl --user is-active ${TAG}-arm >/dev/null 2>&1; do
    g=$(cat "$G"); echo "$g" >> "$O/gtt.log"
    if [ -n "$K" ]; then t=$(( $(cat "$K/temp1_input" 2>/dev/null || echo 0) / 1000 )); else t=0; fi
    if [ $(( (g - G0B) / 1073741824 )) -ge 24 ] || [ "$t" -ge 93 ]; then
      echo "WATCHDOG gtt=$g t=$t" >> "$O/run.log"
      systemctl --user stop ${TAG}-arm 2>/dev/null
      for P in $(pgrep -f "port $PORT" 2>/dev/null); do kill -TERM $P 2>/dev/null; done
      break
    fi
    sleep 1
  done ) &
WD=$!

if [ "$UP" = "1" ]; then
  timeout "${ARM_TIMEOUT:-1200}" python3 "$DET" "$PORT" "$O" 50 > "$O/detector.log" 2>&1
  echo "  detector exit=$? (124 = arm exceeded ${ARM_TIMEOUT:-1200}s and was cut off)"
  tail -1 "$O/detector.log" | sed 's/^/  /'
fi
kill "$WD" 2>/dev/null
echo "gtt_max $(sort -n "$O/gtt.log" 2>/dev/null | tail -1) gtt_start $G0B" >> "$O/run.log"
echo "  watchdog: $(grep -o 'WATCHDOG.*' "$O/run.log" 2>/dev/null | head -1 || echo 'no abort') | gtt samples: $(wc -l < "$O/gtt.log" 2>/dev/null || echo 0)"

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
