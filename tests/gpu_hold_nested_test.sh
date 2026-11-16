#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
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
# SPDX-License-Identifier: Apache-2.0
#
# gpu_hold_nested_test.sh - a payload that wraps itself in another gpu-hold on the SAME queue is refused, because it can
# only ever wait for the ticket it is already inside.
#
# Why (engine#2727): measured 2026-11-16 - one ticket took the token at 17:34:09 and its payload spawned a second
# gpu-hold with the same label on the same queue; that inner ticket then polled for the token its own parent held for
# 764 s, the outer bound ended the window with the GPU idle, and coder-llm (every PR's pr-agent/reviewer check) waited
# 25 min in the queue behind it. gpu-hold already hands its child GPU_HOLD_TICKET and GPU_HOLD_LABEL, but nothing said
# which QUEUE the child was inside, so "am I already under a ticket?" was unanswerable for the self-wrapping idiom.
#
# Arms, all on a private queue and a sandboxed HOME (the real ~/GPU-QUEUE and ~/GPU-TOKEN are never touched):
#   1. nested on the same queue -> rc 2, message names the outer label and engine#2727
#   2. nested on a DIFFERENT queue -> runs (npu-hold's documented shape: a GPU claim inside an NPU ticket)
#   3. the marker is absent for a command started outside any ticket
#   4. the payload sees GPU_HOLD_HELD_QUEUE and GPU_HOLD_HELD_LABEL
#   5. GPU_HOLD_ALLOW_NESTED=1 lets a same-queue nest through instead of refusing
#
# usage: gpu_hold_nested_test.sh <path-to-gpu-hold>
set -u
HOLD=${1:?usage: gpu_hold_nested_test.sh <path-to-gpu-hold>}
[ -x "$HOLD" ] || [ -f "$HOLD" ] || { echo "  SKIP gpu-hold not found at $HOLD"; exit 77; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ghnested.XXXXXX") || exit 77
trap 'rm -rf "$tmp"' EXIT
fails=0
env_common=(HOME="$tmp" ONEBIT_HOLD_DIR="$tmp/hold" GPU_HOLD_COOLDOWN=0 GPU_HOLD_RELEASE_WAIT=0
            GPU_HOLD_WAIT=20 GPU_HOLD_MINUTES=1 GPU_HOLD_THERMAL_FLAG="$tmp/none")

# 1. same queue -> refused by name
out=$(env "${env_common[@]}" timeout 60 bash "$HOLD" -l outer-arm1 -- \
        bash -c '"$1" -l inner-arm1 -t 0.05 -- true; printf "INNER_RC=%s\n" "$?"' _ "$HOLD" 2>&1)
rc=$?
inner_rc=$(printf '%s\n' "$out" | sed -n 's/^INNER_RC=//p' | head -1)
printf '  arm1 outer rc=%d inner rc=%s\n' "$rc" "${inner_rc:-<none>}"
if [ "$rc" != 0 ]; then echo "  FAIL arm1: the outer hold did not run (rc=$rc)"; fails=$((fails+1)); fi
if [ "${inner_rc:-}" != 2 ]; then echo "  FAIL arm1: same-queue nest was not refused (inner rc=${inner_rc:-<none>}, want 2)"; fails=$((fails+1)); fi
printf '%s\n' "$out" | grep -q "engine#2727" || { echo "  FAIL arm1: the refusal does not cite engine#2727"; fails=$((fails+1)); }
printf '%s\n' "$out" | grep -q "outer-arm1" || { echo "  FAIL arm1: the refusal does not name the outer label"; fails=$((fails+1)); }

# 2. a DIFFERENT queue still runs: npu-hold's GPU claim inside an NPU ticket, and the hw-gate's own claim
out=$(env "${env_common[@]}" timeout 60 bash "$HOLD" -l outer-arm2 -- \
        bash -c 'GPU_TOKEN_FILE="$3" "$1" -l inner-arm2 -q "$2" -t 0.05 -- true; printf "INNER_RC=%s\n" "$?"' _ "$HOLD" "$tmp/other-queue" "$tmp/other-token" 2>&1)
rc=$?
inner_rc=$(printf '%s\n' "$out" | sed -n 's/^INNER_RC=//p' | head -1)
printf '  arm2 outer rc=%d inner rc=%s\n' "$rc" "${inner_rc:-<none>}"
if [ "${inner_rc:-}" != 0 ]; then echo "  FAIL arm2: a different-queue nest was refused (inner rc=${inner_rc:-<none>}, want 0)"; fails=$((fails+1)); fi

# 3 + 4. outside any ticket the marker is empty; inside one the payload sees the queue and the label
out=$(env "${env_common[@]}" timeout 60 "$HOLD" -l marker-arm -- \
        bash -c 'printf "QUEUE=%s\nHELD_LABEL=%s\n" "${GPU_HOLD_HELD_QUEUE:-<unset>}" "${GPU_HOLD_HELD_LABEL:-<unset>}"' 2>&1)
seen_queue=$(printf '%s\n' "$out" | sed -n 's/^QUEUE=//p' | head -1)
seen_label=$(printf '%s\n' "$out" | sed -n 's/^HELD_LABEL=//p' | head -1)
printf '  arm3/4 payload QUEUE=%s HELD_LABEL=%s\n' "${seen_queue:-<none>}" "${seen_label:-<none>}"
if [ "${seen_queue:-<unset>}" = "<unset>" ]; then echo "  FAIL arm4: GPU_HOLD_HELD_QUEUE is not exported to the payload"; fails=$((fails+1)); fi
if [ "${seen_label:-<unset>}" != "marker-arm" ]; then echo "  FAIL arm4: GPU_HOLD_HELD_LABEL is '${seen_label:-<unset>}', want marker-arm"; fails=$((fails+1)); fi
plain=$(env -u GPU_HOLD_HELD_QUEUE -u GPU_HOLD_HELD_LABEL bash -c 'printf "%s\n" "${GPU_HOLD_HELD_QUEUE:-<unset>}"')
printf '  arm3 outside-ticket marker: %s\n' "$plain"
if [ "$plain" != "<unset>" ]; then echo "  FAIL arm3: the marker leaks outside a ticket"; fails=$((fails+1)); fi

# 4b. THE TOKEN IS THE KEY TOO (engine#2727 review): `-q` moves the queue without moving the token, so a nest on a
#     DIFFERENT queue still waits for the token its parent holds. Outer on q1, payload runs `-q q2` with the inherited
#     token: refused by name, and no ticket is left in q2.
mkdir -p "$tmp/q1" "$tmp/q2"
out=$(env "${env_common[@]}" GPU_QUEUE_DIR="$tmp/q1" GPU_TOKEN_FILE="$tmp/q1-token" timeout 60 bash "$HOLD" -l outer-arm4b -- \
        bash -c '"$1" -l inner-arm4b -q "$2" -t 0.05 -- true; printf "INNER_RC=%s\n" "$?"' _ "$HOLD" "$tmp/q2" 2>&1)
rc=$?
inner_rc=$(printf '%s\n' "$out" | sed -n 's/^INNER_RC=//p' | head -1)
left=$(ls "$tmp/q2" 2>/dev/null | grep -c 'inner-arm4b' || true)
printf '  arm4b outer rc=%d inner rc=%s tickets-left-in-q2=%s\n' "$rc" "${inner_rc:-<none>}" "$left"
if [ "${inner_rc:-}" != 2 ]; then echo "  FAIL arm4b: a different-queue nest on the SAME token was not refused (inner rc=${inner_rc:-<none>}, want 2)"; fails=$((fails+1)); fi
printf '%s\n' "$out" | grep -q "moves the queue but not the token" || { echo "  FAIL arm4b: the refusal does not say why the queue alone is the wrong key"; fails=$((fails+1)); }
if [ "$left" != 0 ]; then echo "  FAIL arm4b: the refused invocation left a ticket in q2"; fails=$((fails+1)); fi

# 5. the override still refuses to be silent: a same-queue nest with GPU_HOLD_ALLOW_NESTED=1 waits instead of exiting 2
out=$(env "${env_common[@]}" GPU_HOLD_WAIT=3 timeout 60 bash "$HOLD" -l outer-arm5 -- \
        bash -c 'GPU_HOLD_ALLOW_NESTED=1 "$1" -l inner-arm5 -w 2 -t 0.05 -- true; printf "INNER_RC=%s\n" "$?"' _ "$HOLD" 2>&1)
inner_rc=$(printf '%s\n' "$out" | sed -n 's/^INNER_RC=//p' | head -1)
printf '  arm5 inner rc with the override=%s\n' "${inner_rc:-<none>}"
if [ "${inner_rc:-}" = 2 ]; then echo "  FAIL arm5: GPU_HOLD_ALLOW_NESTED=1 did not lift the refusal"; fails=$((fails+1)); fi

if [ "$fails" = 0 ]; then
    echo "gpu_hold_nested: 6 arms pass"
    exit 0
fi
echo "gpu_hold_nested: $fails failure(s)"
exit 1
