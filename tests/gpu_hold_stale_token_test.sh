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
# A token that names NO holder is a third case, and it stranded the whole box on 2026-11-19: a 0-byte ~/GPU-TOKEN
# left by a crash cycle parked ticket ab2 for 4828 s, because the dead-pid arm cannot see it and the grant path's
# `set -o noclobber` cannot overwrite it. It is reaped once it has been unchanged for GPU_HOLD_TOKEN_GRACE seconds:
#   3. an EMPTY token (0 bytes) is reaped and the ticket is taken;
#   4. a TRUNCATED token ("gpu-hold waste pi", written mid-crash) is reaped the same way;
#   5. an unreadable token that is still FRESH is left alone — a holder writing its token must not be robbed.
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

run_ticket() {   # <token-content> <label> [old] [grace] [wait] -> prints the log, echoes an rc on the last line
    local content=$1 label=$2 when=${3:-} grace=${4:-} wait=${5:-6} home=$tmp/$2
    mkdir -p "$home"/.cache "$home/gpu-queue"
    # "-" writes a genuinely 0-byte token, which is what a crash cycle leaves; anything else is a token body.
    if [ "$content" = - ]; then : > "$home/GPU-TOKEN"; else printf '%s\n' "$content" > "$home/GPU-TOKEN"; fi
    [ "$when" = old ] && touch -d '-1 hour' "$home/GPU-TOKEN"
    HOME="$home" GPU_QUEUE_DIR="$home/gpu-queue" GPU_TOKEN_FILE="$home/GPU-TOKEN" ${grace:+GPU_HOLD_TOKEN_GRACE="$grace"} \
        timeout 30 bash "$HOLD" -t 1 -w "$wait" -l "$label" -- true > "$tmp/$label.log" 2>&1
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

# Arms 3 and 4: a token that names no holder -- the state a crash leaves behind -- is a leftover, not a claim.
for shape in empty truncated; do
    if [ "$shape" = empty ]; then body=-; else body="gpu-hold waste pi"; fi
    out=$(run_ticket "$body" "$shape" old)
    if ! printf '%s' "$out" | grep -q "dropped unreadable token"; then
        echo "FAIL $shape-token: an unreadable token that names no holder was not reaped -- every ticket spins on it"
        echo "  log: $(printf '%s' "$out" | tail -3 | tr '\n' ' ')"
        fail=1
    elif ! printf '%s' "$out" | grep -q "rc=0"; then
        echo "FAIL $shape-token: token reaped but the ticket was not taken: $(printf '%s' "$out" | tail -2 | tr '\n' ' ')"
        fail=1
    else
        echo "ok   a$([ "$shape" = empty ] && echo "n empty" || echo " truncated") token that names no holder is reaped and the ticket is taken"
    fi
done

# Arm 5: the grace. A holder writes its token once; a fresh unreadable file may be mid-write, so it is left alone.
out=$(run_ticket - fresh-write 30 4)
if printf '%s' "$out" | grep -q "dropped unreadable token"; then
    echo "FAIL fresh-token: a token written seconds ago was reaped -- a holder mid-write can be robbed"
    fail=1
else
    echo "ok   a fresh unreadable token is left alone until the grace passes"
fi

exit "$fail"
