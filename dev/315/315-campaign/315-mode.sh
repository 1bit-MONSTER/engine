#!/usr/bin/env bash
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
# engine#315: the execution-mode toggle. Diagnostic, never acceptance-eligible.
#
# Why this is a lever and not a repeat. The knob inventory recorded `HRX_COMMAND_BUFFER_MODE` and dismissed it
# for the *wait-barrier strategy* question ("selects the command-buffer builder, not the queue execution mode")
# - correct, and never run as a fault-rate toggle. It is more than a builder switch:
#
#   runtime/src/iree/hal/drivers/amdgpu/logical_device.c:159   default is AQL
#   runtime/src/iree/hal/drivers/amdgpu/logical_device.c:244-253  the string param takes aql | pm4 | auto and
#                                                                REJECTS anything else (no silent fallback)
#   runtime/src/iree/hal/drivers/amdgpu/logical_device.c:2018-2027  PM4 and AUTO "promise reusable dispatch
#                                                                command buffers, including dynamic binding-table
#                                                                fixups" and provision queue-control storage for it
#   runtime/src/iree/hal/drivers/amdgpu/registration/driver_module.c:61  the flag default is also "aql"
#
# So `pm4` selects a different execution path on two axes at once: packet/barrier semantics (PM4 IB rather than
# native AQL, on a device whose capability set has no AQL_BARRIER_VALUE) and REUSABLE command buffers whose
# binding tables are fixed up at submission instead of recorded once. The routed-FFN path binds two *generated
# transient* tables (expert table, partition table) whose producer dispatches sit in the same plan
# (dispatch-mul-mat-id-common.h:342-368). A binding fixup that resolves those to the wrong generation is exactly
# a per-token failure, and it exists only on the PM4/AUTO path.
#
# The instrument is deliberately absent: the tensor fingerprint adds 16 device-to-host reads per graph and the
# copy probe prints from the submit path, either of which could perturb the timing under test (same argument as
# 315-graphsync.sh). Compare the counts against the no-knob arms already on disk from the same harness.
#
# Readout, pre-committed:
#   fault rate falls on the pm4 cell        -> the default AQL path's per-execution binding/barrier behaviour is
#                                              the carrier; the next step is to diff the two paths' dispatches
#   fault rate rises or is unchanged        -> the mode is not the carrier, and the reusable-command-buffer
#                                              hypothesis (candidate 1 in evidence/315-moe-routing-routed-ffn-static.md)
#                                              is falsified too
#   the knob is not observable in the logs  -> report that, do not read the counts as evidence either way
#
# Usage: 315-mode.sh [pm4|aql|auto] [pairs] [tag]
set -uo pipefail
MODE=${1:-pm4}; PAIRS=${2:-6}; TAG=${3:-315mode}
IREE_HAL_AMDGPU_LIBHSA_PATH=${IREE_HAL_AMDGPU_LIBHSA_PATH:-/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1}
export IREE_HAL_AMDGPU_LIBHSA_PATH
D=/home/bcloud/wt
CONSOLE=$D/$TAG-console.log

exec 9>"$HOME/.cache/lax-decode/box.lock"
if flock -w "${MODE_LOCK_WAIT:-10800}" 9; then
  echo "mode: box lock held $(date -Is)"
else
  echo "mode: lock unobtainable after ${MODE_LOCK_WAIT:-10800}s - aborting rather than running concurrently"
  exit 3
fi

# Preflight: is the knob observable at all? A knob that does not reach the driver would make the arm counts
# meaningless, so detect that first and say so rather than reading the counts.
echo "=== mode preflight $(date -Is) ==="
OBS="unobservable"
for M in aql "$MODE"; do
  L=$D/modepre-$M.log
  HRX_GPU_DEBUG=1 HRX_COMMAND_BUFFER_MODE=$M timeout 240 "$D/315instr3-build/bin/llama-server" \
      -m /home/bcloud/models/Qwen3-0.6B-Q4_K_M.gguf -dev HRX0 -ngl 99 -c 512 -np 1 --no-webui \
      --host 127.0.0.1 --port $((20500 + RANDOM % 200)) > "$L" 2>&1 &
  sp=$!
  sleep 90; kill $sp 2>/dev/null; wait $sp 2>/dev/null
  echo "  $M: log $L $(wc -l < "$L") lines, $(grep -ciE 'command.?buffer|pm4|aql' "$L") mode-related lines"
done
if [ -f "$D/modepre-aql.log" ] && [ -f "$D/modepre-$MODE.log" ] && \
   ! cmp -s "$D/modepre-aql.log" "$D/modepre-$MODE.log"; then
  OBS="observable (logs differ)"
fi
echo "  knob: $OBS"
echo "  if unobservable, the arm counts below are NOT evidence about the mode - recorded and reported as such"

cd "$D" || exit 1
echo "=== mode arms mode=$MODE tag=$TAG pairs=$PAIRS $(date -Is) ==="
REGIME_LO=${REGIME_LO:-80} REGIME_HI=${REGIME_HI:-92} \
CONTROL_EVERY=999 NOSTOP=1 MINCLEAN=999 NOLOCK=1 ARM_TIMEOUT=${ARM_TIMEOUT:-1800} \
EXTRA_ENV="HRX_COMMAND_BUFFER_MODE=$MODE" \
  bash "$D/315-paired.sh" "$PAIRS" "${NB:-2}" "${NCTX:-2048}" \
       "$D/315instr3-build/bin" "$D/315instr3-build/bin" "$TAG"

echo "=== mode readout $(date -Is) knob=$OBS ==="
grep -hE "^  pair" "$CONSOLE" || true
echo "in-regime clean (OK):  $(grep -c 'arm l: OK' "$CONSOLE" || true)"
echo "in-regime fault:       $(grep -c 'arm l: \(FAULT\|INVALID\)' "$CONSOLE" || true)"
echo "out-of-regime (OOR):   $(grep -c 'arm l: OOR' "$CONSOLE" || true)"
echo "compare: the no-knob in-regime record for this harness is on disk (315chain/315gsync/315row arm dirs)"
