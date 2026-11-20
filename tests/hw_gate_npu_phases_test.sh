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
# tests/hw_gate_npu_phases_test.sh — an NPU-only phase set does not claim the GPU token (engine#3030).
#
# The gate claims ~/GPU-TOKEN for the whole run, which is right when a GPU phase is named and wrong when only NPU
# phases are: measured 2026-11-20, `--phases npuserve` waited 858 s behind GPU work and then held the GPU for an
# NPU-only test -- a wait nobody explains and a hold another lane pays for.
#
# Hermetic: `gpu-hold` is a stub on PATH that logs its own name, and the gate is stopped by its own usage exit after
# the claim would have happened. No token, no device, no build.
#
# usage: bash tests/hw_gate_npu_phases_test.sh    (exit 0 = every arm passes, 1 = a failure)
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
gate=$root/scripts/hw-gate.sh
[ -f "$gate" ] || { echo "hw_gate_npu_phases_test: no $gate" >&2; exit 2; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"
cat > "$tmp/bin/gpu-hold" <<'STUB'
#!/bin/sh
printf '%s\n' "$*" >> "$STUBLOG"
exec "$@"
STUB
chmod +x "$tmp/bin/gpu-hold"

fails=0
arm() {  # <name> <phases|-> <expected calls>
    local name=$1 phases=$2 want=$3 log="$tmp/log.$RANDOM"
    : > "$log"
    if [ "$phases" = "-" ]; then
        STUBLOG=$log PATH="$tmp/bin:$PATH" timeout 60 bash "$gate" > "$tmp/out.$RANDOM" 2>&1
    else
        STUBLOG=$log PATH="$tmp/bin:$PATH" timeout 60 bash "$gate" --phases "$phases" > "$tmp/out.$RANDOM" 2>&1
    fi
    local got
    got=$(grep -c . "$log" 2> /dev/null || true)
    got=${got:-0}
    if [ "$got" = "$want" ]; then
        echo "ok   $name: gpu-hold called $got time(s)"
    else
        echo "FAIL $name: gpu-hold called $got time(s), wanted $want"
        fails=$((fails + 1))
    fi
}

# The claim is what an NPU-only run must not make.
arm "npuserve alone is NPU-only"        npuserve        0
arm "npu,npuserve is NPU-only"          "npu,npuserve"  0
# ... and what any GPU phase must still make, or this change would quietly drop the queue the gate depends on.
arm "placement still claims the GPU"    placement       1
arm "build,ops still claim the GPU"     "build,ops"     1
arm "a mixed set still claims the GPU"  "npu,placement" 1

[ "$fails" = 0 ] && { echo "hw_gate_npu_phases_test: pass"; exit 0; }
echo "hw_gate_npu_phases_test: $fails failure(s)"
exit 1
