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
#
# Hyperloom custom-workload entrypoint for the engine on HRX (Strix Halo, gfx1151).
#
# Hyperloom injects RESULT_DIR, RESULT_FILENAME and RUNNER_TYPE, and restarts this script with a
# different environment for every candidate -- the lever variables levers.json declares, or ones
# the optimizer proposes. That is the whole mechanism; this script only wires the box etiquette
# around benchmark.py:
#
#   * it builds NOTHING. The framework checkout must already be built (README.md): a build inside
#     the measurement would be charged to whichever candidate happened to run first.
#   * ONE measurement is ONE claim on the box's fair GPU FIFO, so Hyperloom never measures
#     outside it, whatever else is on the box. `-t 20` is the hold's ceiling and its watchdog
#     aborts the run at the critical die temperature (93 C); HL_GPU_HOLD_MINUTES can lower it.
#
# Exit code: 0 for a measurement that ran (a rejected candidate is a 0 with
# quality_gate.passed=false, which is how Hyperloom rejects it and moves on), non-zero only for
# an infrastructure failure.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
HERE=$PWD
: "${RESULT_DIR:=$PWD}"
export RESULT_DIR

# gpu-hold lives in the box user's ~/.local/bin, which is NOT $HOME when the caller runs under a
# sandboxed HOME (a fresh home for the agent CLIs). Resolve it from the password database too, so the
# measurement is never started outside the FIFO because of a HOME override.
GPU_HOLD=${HL_GPU_HOLD:-}
HOLD_MIN=${HL_GPU_HOLD_MINUTES:-20}
if [ -z "$GPU_HOLD" ] && command -v gpu-hold >/dev/null 2>&1; then
  GPU_HOLD=$(command -v gpu-hold)
fi
if [ -z "$GPU_HOLD" ]; then
  REAL_HOME=$(getent passwd "$(id -u)" | cut -d: -f6)
  for cand in "$HOME/.local/bin/gpu-hold" "${REAL_HOME:-/nonexistent}/.local/bin/gpu-hold"; do
    [ -x "$cand" ] && { GPU_HOLD=$cand; break; }
  done
fi
GPU_HOLD=${GPU_HOLD:-$HOME/.local/bin/gpu-hold}
if [ ! -x "$GPU_HOLD" ]; then
  echo "[1bit-hl] FATAL: $GPU_HOLD is not executable; a measurement outside the GPU FIFO is not allowed" >&2
  exit 1
fi

# The FIFO's state lives next to the REAL home, not next to a sandboxed HOME. gpu-hold defaults both
# paths to $HOME, and Hyperloom runs this entrypoint under a fresh home (the preflight writes into
# ~/.claude/config.json), so without this every measurement would take a private token and queue in a
# private directory -- inside no queue at all, which is the one thing this claim exists to guarantee.
REAL_HOME=${REAL_HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}
export GPU_TOKEN_FILE=${GPU_TOKEN_FILE:-${REAL_HOME:-$HOME}/GPU-TOKEN}
export GPU_QUEUE_DIR=${GPU_QUEUE_DIR:-${REAL_HOME:-$HOME}/GPU-QUEUE}

# The HRX backend reaches its HSA runtime through IREE's loader, which has to be told where that
# library is: without this the backend registers no device at all ("Available devices: (none)").
LIBHSA=${HL_LIBHSA:-/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1}
export IREE_HAL_AMDGPU_LIBHSA_PATH=${IREE_HAL_AMDGPU_LIBHSA_PATH:-$LIBHSA}

# The nightly libhsa page-faults on an HRX allocation unless SVM is off (engine #715), so every HRX
# launch on this box carries these three. Defaulted, not forced: a candidate that wants SVM back can
# still set them, and the run records whichever value was in force.
export HSA_USE_SVM=${HSA_USE_SVM:-0}
export HSA_USERPTR_FOR_PAGED_MEM=${HSA_USERPTR_FOR_PAGED_MEM:-0}
export HRX_ALLOW_NO_SVM=${HRX_ALLOW_NO_SVM:-1}

# --- Room: before the claim, and again inside it ------------------------------------------------
# gpu-hold admits a measurement on the die's temperature and the FIFO's order. It has no idea how much
# GTT the workload needs, and the engine's own guardrail (ggml_hrx, engine #680) refuses a load whose
# device buffers do not fit in what is left -- correctly, and *after* the claim is already ours. On
# 2026-11-08 that difference cost two claims in a row, both taken at 48.9 / 59.9 C while other lanes
# held 89.1 / 89.6 GiB of GTT, both dying in the first leg:
#   HRX0 guardrail: ... pool budget 14.4 GiB - a launch that needs more is refused here
#   llama_model_load: error loading model: unable to allocate HRX0 buffer
# and both artifacts recorded `llama-bench failed (rc=1, rows=[])`, score 0.0 -- which reads like a
# broken workload and is really a claim taken for work that could not fit. So the entrypoint waits for
# room *before* claiming and re-checks the moment the claim is ours, because the box moves while queued
# (the window harness learned the same lesson the same week: a pre-claim check alone is a race).
# 21 GiB, not 24: the workload states its own requirement. `1bit serve` prints it at load --
#   need 20.21 GiB (weights 16.35 + KV/state 1.40 + compute 0.50; draft 1.96)
# -- and a round 24 refused two windows that fitted. Measured 2026-11-08 08:18: GTT 74 GiB in use,
# 74 + 20.21 = 94.21 GiB, inside the 95 GiB rule and with 30 GiB of the engine's 104 GiB ceiling free;
# the gate refused at 98 GiB because the constant said 24. The other refusal that session (06:26, GTT 75)
# was correct at any value >= 20.21, so the constant was costing claims for nothing. Keep it a hair above
# the measured figure rather than a round number nobody measured.
GTT_NEED=${HL_GTT_NEED_GIB:-21}
GTT_RULE=${HL_GTT_RULE_GIB:-95}     # this box's GTT rule; the engine's own ceiling is 104 with a 16 GiB reserve
GTT_WAIT=${HL_GTT_ROOM_WAIT:-1800}
export HL_GTT_NEED_GIB="$GTT_NEED" HL_GTT_RULE_GIB="$GTT_RULE"
gtt_gib() {
  local f=/sys/class/drm/card0/device/mem_info_gtt_used
  [ -r "$f" ] || { echo 0; return; }          # unreadable sysfs must not be read as "full"
  echo $(( $(cat "$f") / 1073741824 ))
}
waited=0
while :; do
  g=$(gtt_gib)
  if [ $(( g + GTT_NEED )) -le "$GTT_RULE" ]; then
    echo "[1bit-hl] room ok: GTT ${g} GiB in use + ${GTT_NEED} GiB needed = $(( g + GTT_NEED )) GiB <= ${GTT_RULE} GiB rule"
    break
  fi
  if [ "$waited" -ge "$GTT_WAIT" ]; then
    echo "[1bit-hl] REFUSING: GTT ${g} GiB in use + ${GTT_NEED} GiB needed = $(( g + GTT_NEED )) GiB > ${GTT_RULE} GiB, and it did not clear in ${GTT_WAIT}s -- not claiming a measurement the engine would refuse (HL_GTT_NEED_GIB / HL_GTT_RULE_GIB / HL_GTT_ROOM_WAIT)" >&2
    exit 3
  fi
  echo "[1bit-hl] waiting for room: GTT ${g} GiB + ${GTT_NEED} GiB = $(( g + GTT_NEED )) GiB > ${GTT_RULE} GiB (${waited}s of ${GTT_WAIT}s)"
  sleep 20; waited=$((waited + 20))
done

# Inside the claim the same check runs again, before the workload loads anything: whatever the queue
# promised, the box is shared and the grant can arrive into a pool that has since filled up.
exec "$GPU_HOLD" -l hyperloom -t "$HOLD_MIN" -- bash -c '
  f=/sys/class/drm/card0/device/mem_info_gtt_used
  g=$([ -r "$f" ] && echo $(( $(cat "$f") / 1073741824 )) || echo 0)
  if [ $(( g + ${HL_GTT_NEED_GIB} )) -gt "${HL_GTT_RULE_GIB}" ]; then
    echo "[1bit-hl] REFUSING after the grant: GTT ${g} GiB + ${HL_GTT_NEED_GIB} GiB = $(( g + ${HL_GTT_NEED_GIB} )) GiB > ${HL_GTT_RULE_GIB} GiB; releasing the claim instead of starting a load the engine will refuse" >&2
    exit 3
  fi
  echo "[1bit-hl] room ok inside the claim: GTT ${g} GiB in use"
  exec python3 "$1"
' _ "$HERE/benchmark.py"
