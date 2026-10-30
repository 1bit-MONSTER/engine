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

GPU_HOLD=${HL_GPU_HOLD:-$HOME/.local/bin/gpu-hold}
HOLD_MIN=${HL_GPU_HOLD_MINUTES:-20}
if [ ! -x "$GPU_HOLD" ]; then
  echo "[1bit-hl] FATAL: $GPU_HOLD is not executable; a measurement outside the GPU FIFO is not allowed" >&2
  exit 1
fi

# The HRX backend reaches its HSA runtime through IREE's loader, which has to be told where that
# library is: without this the backend registers no device at all ("Available devices: (none)").
LIBHSA=${HL_LIBHSA:-/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1}
export IREE_HAL_AMDGPU_LIBHSA_PATH=${IREE_HAL_AMDGPU_LIBHSA_PATH:-$LIBHSA}

exec "$GPU_HOLD" -l hyperloom -t "$HOLD_MIN" -- python3 "$HERE/benchmark.py"
