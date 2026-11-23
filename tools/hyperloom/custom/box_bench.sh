#!/bin/bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# box_bench.sh OUT -- ONE engine HRX measurement (benchmark.py) ON the box, inside `box-hold gpu timeout ...` started by
# custom_radeon8060s.sh. Never started by hand outside box-hold.
#
# The box is the appliance: the engine payload at /opt/1bit-package/1bit is the build under test (`1bit` at its top,
# llama-bench / llama-perplexity under hrx/llama/bin -- the layout benchmark.py reads as HL_BUILD). Before the workload
# loads anything this script checks room twice over:
#   RAM  MemAvailable >= HL_RAM_NEED_GB + 16 (fleet floor), and a watchdog stops the job if it falls under 16 GB;
#   GTT  GTT in use + HL_GTT_NEED_GIB <= HL_GTT_RULE_GIB (21 / 95 GiB: `1bit serve` prints need 20.21 GiB for the 27B),
#        because the engine's guardrail refuses a load that does not fit only AFTER the claim is ours.
# OUT/status is "ok" or the reason; benchmark.py writes OUT/inferencex_result.json itself.
set -u
OUT=${1:?usage: box_bench.sh OUT}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT" || exit 2
status() { echo "$*" > "$OUT/status"; echo "[box_bench] $*"; }
num() { case "$2" in ''|*[!0-9]*) status "refused: $1=$2 is not a whole number"; exit 2 ;; esac; }
NEED=${HL_RAM_NEED_GB:-32}; num HL_RAM_NEED_GB "$NEED"
FLOOR=${HL_RAM_FLOOR_GB:-16}; num HL_RAM_FLOOR_GB "$FLOOR"
GTT_NEED=${HL_GTT_NEED_GIB:-21}; num HL_GTT_NEED_GIB "$GTT_NEED"
GTT_RULE=${HL_GTT_RULE_GIB:-95}; num HL_GTT_RULE_GIB "$GTT_RULE"
mem_avail_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
have=$(mem_avail_gb)
echo "ram: need ${NEED} GB (+${FLOOR} floor), MemAvailable ${have} GB at start" > "$OUT/ram.txt"
[ "$have" -ge $((NEED + FLOOR)) ] || { status "refused: RAM need $NEED + $FLOOR GB, MemAvailable $have GB"; exit 3; }
f=/sys/class/drm/card0/device/mem_info_gtt_used
g=$([ -r "$f" ] && echo $(( $(cat "$f") / 1073741824 )) || echo 0)
[ $((g + GTT_NEED)) -le "$GTT_RULE" ] || { status "refused: GTT $g GiB in use + $GTT_NEED GiB > $GTT_RULE GiB"; exit 3; }
echo "gtt: ${g} GiB in use + ${GTT_NEED} GiB needed <= ${GTT_RULE} GiB" >> "$OUT/ram.txt"

# Root, via the caller's `sudo -n`: the 1bit OS appliance has /dev/kfd root:root 0600 (Lemonade runs as root), and as
# bcloud hsa_init fails with HSA_STATUS_ERROR_OUT_OF_RESOURCES -> HRX registers no device -> `-dev HRX0` is rejected
# (measured 23 Nov, ticket hl-20261123T044207Z-2210413). The output and the gate's state go back to the box user.
OWNER=${HL_OUT_OWNER:-bcloud}
ulimit -l unlimited 2>/dev/null
give_back() { chown -R "$OWNER:$OWNER" "$OUT" 2>/dev/null; [ -n "${HL_STATE:-}" ] && chown -R "$OWNER:$OWNER" "$HL_STATE" 2>/dev/null; }
trap give_back EXIT
PKG=${HL_PACKAGE:-/opt/1bit-package/1bit}
export HL_BUILD=${HL_BUILD:-$PKG}
LIBHSA=$PKG/hrx/llama/bin/libhsa-runtime64.so.1
[ -f "$LIBHSA" ] && export IREE_HAL_AMDGPU_LIBHSA_PATH=${IREE_HAL_AMDGPU_LIBHSA_PATH:-$LIBHSA}
export HSA_USE_SVM=${HSA_USE_SVM:-0} HSA_USERPTR_FOR_PAGED_MEM=${HSA_USERPTR_FOR_PAGED_MEM:-0} HRX_ALLOW_NO_SVM=${HRX_ALLOW_NO_SVM:-1}
export RESULT_DIR=$OUT RESULT_FILENAME=inferencex_result

# Its own process group: benchmark.py's `finally` that stops `1bit serve` does not run on SIGTERM, so a stop kills the
# whole group (and serve.pid lets the caller's kill-by-ticket reach it even if this shell is already gone).
setsid python3 -u "$HERE/benchmark.py" > "$OUT/benchmark.log" 2>&1 &
job=$!
echo "$job" > "$OUT/serve.pid"
( low=999; while kill -0 "$job" 2>/dev/null; do a=$(mem_avail_gb); [ "$a" -lt "$low" ] && { low=$a; echo "$low" > "$OUT/ram.low"; }
    if [ "$a" -lt "$FLOOR" ]; then echo "RAM-GUARD: MemAvailable $a GB < $FLOOR at $(date -u +%T)" >> "$OUT/ram.txt"
        kill -TERM -"$job"; exit; fi
    sleep 2; done ) &
guard=$!
trap 'status "killed (signal)"; kill -TERM -"$job" 2>/dev/null; sleep 5; kill -KILL -"$job" 2>/dev/null; kill "$guard" 2>/dev/null; rm -f "$OUT/serve.pid"; give_back; exit 143' TERM INT
wait "$job"; rc=$?
kill "$guard" 2>/dev/null
kill -KILL -"$job" 2>/dev/null   # anything the measurement left in its group
rm -f "$OUT/serve.pid"
grep -q RAM-GUARD "$OUT/ram.txt" && { status "killed by the RAM guard (ram.txt)"; exit 1; }
[ $rc = 0 ] && status ok || status "failed: benchmark.py rc=$rc (benchmark.log)"
