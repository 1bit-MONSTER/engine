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

# tests/gpu_hold_stale_token_test.sh — a token whose holder is gone is reaped, WHOEVER wrote it.
#
# Measured on 2026-11-16: the owner stopped `npu-serve-keeper`, which left `~/NPU-TOKEN` reading
# "npu-serve-keeper pid 298235" with that pid gone. Every NPU ticket then queued behind a dead process — mine
# waited 246 s — because reap_token() only dropped tokens whose content matched `gpu-hold*`. The keeper is a
# LEGITIMATE token holder (npu-hold:22), so its token is exactly the one that strands the queue when it stops.
#
# The pid is the identity; the prefix is not. Two arms, because the fix has two ways to be wrong:
#   1. a foreign holder with a dead pid IS reaped, and the ticket is taken rather than waiting out -w;
#   2. a foreign holder with a LIVE pid is NOT reaped — the fix must not steal a live lane's token.
#
# Everything happens under a temporary HOME, so the real ~/GPU-QUEUE and ~/GPU-TOKEN are never touched.
#
#   <gpu-hold> -> 0 both arms hold, 1 one of them failed, 77 cannot run here
set -u

HOLD=${1:?usage: gpu_hold_stale_token_test.sh <path-to-gpu-hold>}
[ -r "$HOLD" ] || { echo "77: cannot read $HOLD"; exit 77; }
command -v kill > /dev/null 2>&1 || { echo "77: no kill(1)"; exit 77; }

tmp=$(mktemp -d) || exit 77
trap 'rm -rf "$tmp"' EXIT
fail=0

# A pid that cannot be alive: the kernel's own pid_max bound is 2^22, and this is above it.
dead_pid=4294967

run_ticket() {   # <token-content> <label> -> prints the log, echoes an rc on the last line
    local content=$1 label=$2 home=$tmp/$2
    mkdir -p "$home"/.cache "$home/gpu-queue"
    printf '%s\n' "$content" > "$home/GPU-TOKEN"
    HOME="$home" GPU_QUEUE_DIR="$home/gpu-queue" GPU_TOKEN_FILE="$home/GPU-TOKEN" \
        timeout 30 bash "$HOLD" -t 1 -w 6 -l "$label" -- true > "$tmp/$label.log" 2>&1
    echo "rc=$?" >> "$tmp/$label.log"
    cat "$tmp/$label.log"
}

# Arm 1: foreign holder, dead pid -> reaped, and the ticket is taken.
out=$(run_ticket "npu-serve-keeper pid $dead_pid 2026-11-16T21:30:00Z" dead)
if ! printf '%s' "$out" | grep -q "dropped stale token"; then
    echo "FAIL dead-holder: no 'dropped stale token' line -- a foreign held token was not reaped"
    echo "  log: $(printf '%s' "$out" | tail -3 | tr '\n' ' ')"
    fail=1
elif ! printf '%s' "$out" | grep -q "rc=0"; then
    echo "FAIL dead-holder: the token was reaped but the ticket was not taken: $(printf '%s' "$out" | tail -2 | tr '\n' ' ')"
    fail=1
else
    echo "ok   a token left by a dead foreign holder is reaped and the ticket is taken"
fi

# Arm 2: foreign holder, LIVE pid -> NOT reaped. Our own process is the live holder.
out=$(run_ticket "npu-serve-keeper pid $$ 2026-11-16T21:30:00Z" live)
if printf '%s' "$out" | grep -q "dropped stale token"; then
    echo "FAIL live-holder: a LIVE holder's token was reaped -- the fix must not steal a live lane's token"
    fail=1
elif ! printf '%s' "$out" | grep -q "rc="; then
    echo "FAIL live-holder: no rc recorded"
    fail=1
else
    rc=$(printf '%s' "$out" | sed -n 's/^rc=//p' | tail -1)
    # Not reaped, so the ticket waits out -w and exits 75 (never came round) or 1; anything but "taken" is fine.
    echo "ok   a LIVE foreign holder's token is left alone (rc=$rc, waited out -w as it should)"
fi

exit "$fail"
