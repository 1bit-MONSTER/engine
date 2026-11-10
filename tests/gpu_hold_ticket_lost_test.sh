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
# Copyright 2026 The 1bit-MONSTER Authors.
# SPDX-License-Identifier: Apache-2.0
#
# tests/gpu_hold_ticket_lost_test.sh — a queued ticket that is removed must fail fast, not wait out -w.
#
# Why: engine #1635. gpu-hold waits for `oldest_ticket == my ticket`. If a reaper (or another lane) removes the
# ticket while its owner waits, that can never become true again, so the loop used to sit until -w expired —
# 5400 s by default — which reads exactly like "queued behind a busy box". The i1346 lane lost two arms to it
# in 20 minutes. This drives the real scripts/gpu-hold against a private queue whose ticket is deleted
# mid-wait, and asserts it exits on the next poll (not at -w), with 75 and a message naming the ticket.
#
# Exit: 0 = the loss is noticed and reported, 1 = it waited out -w or said nothing, 77 = cannot run here.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
hold=$here/scripts/gpu-hold
[ -x "$hold" ] || { echo "skip: no $hold"; exit 77; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
queue=$work/queue
token=$work/token
mkdir -p "$queue"

# another label holds the token, and its ticket is older than ours: the head of the queue is never us.
printf 'pid=%s\nlabel=other\ncmd=true\nenqueued=now\n' $$ > "$queue/00000000000000000001.other"
printf 'gpu-hold other pid %s now\n' $$ > "$token"

# -w 300 is deliberately far longer than the test: without the guard this hangs for five minutes.
: > "$work/log"
GPU_HOLD_POLL=1 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_COOLDOWN=0 GPU_HOLD_TCTL_START=0 \
GPU_HOLD_IGNORE_THERMAL_HOLD=1 \
    "$hold" -q "$queue" -w 300 -t 0.05 -l ours true >> "$work/log" 2>&1 &
pid=$!

# wait until our ticket is written (the tool logs 'queued'), then delete exactly that file
for _ in $(seq 1 50); do
    grep -q "queued 'ours'" "$work/log" 2> /dev/null && break
    sleep 0.2
done
ourticket=$(ls "$queue" | grep '\.ours$' | head -n 1)
if [ -z "$ourticket" ]; then
    echo "skip: gpu-hold never queued a ticket here"; kill "$pid" 2> /dev/null; exit 77
fi
start=$(date +%s)
rm -f "$queue/$ourticket"

# it must notice within a couple of polls, not at -w
for _ in $(seq 1 100); do
    kill -0 "$pid" 2> /dev/null || break
    sleep 0.1
done
elapsed=$(( $(date +%s) - start ))
wait "$pid"
rc=$?

fails=0
if kill -0 "$pid" 2> /dev/null; then
    echo "FAIL gpu-hold is still queued after its ticket was removed (${elapsed}s)"; fails=$((fails + 1))
fi
[ "$rc" = 75 ] || { echo "FAIL expected exit 75 (the timeout status), got $rc"; fails=$((fails + 1)); }
[ "$elapsed" -le 10 ] || { echo "FAIL took ${elapsed}s to notice; -w was 300s, so this is not the fast path"; fails=$((fails + 1)); }
case $(cat "$work/log") in
    *"$ourticket is gone from"*) ;;
    *) echo "FAIL the log does not name the lost ticket:"; sed 's/^/    /' "$work/log"; fails=$((fails + 1)) ;;
esac
case "$ourticket" in
    *ours) ;;
    *) echo "FAIL sanity: read ticket '$ourticket'"; fails=$((fails + 1)) ;;
esac

# Second arm (engine #1635 point 2): a REAL reaper, not a hand deletion. Our ticket is rewritten to carry a dead pid,
# the way a wrapper's pid can outlive its usefulness; another gpu-hold then drops it as stale. The victim must (a) exit
# fast as before and (b) quote the note the reaper left, so the log says WHY the position vanished.
queue2=$work/queue2
token2=$work/token2
mkdir -p "$queue2"
# sorts AFTER the victim ticket: oldest_ticket() returns at the first NON-stale ticket, so a live ticket
# ahead of the victim would hide it from the reaper (found the hard way - the first draft hung here).
printf 'pid=%s\nlabel=other\n' $$ > "$queue2/99999999999999999999.other"
printf 'gpu-hold other pid %s now\n' $$ > "$token2"
: > "$work/log2"
GPU_HOLD_POLL=1 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_COOLDOWN=0 GPU_HOLD_TCTL_START=0 \
GPU_HOLD_IGNORE_THERMAL_HOLD=1 GPU_TOKEN_FILE=$token2 \
    "$hold" -q "$queue2" -w 300 -t 0.05 -l victim true >> "$work/log2" 2>&1 &
vpid=$!
for _ in $(seq 1 50); do
    grep -q "queued 'victim'" "$work/log2" 2> /dev/null && break
    sleep 0.2
done
vticket=$(ls "$queue2" | grep '\.victim$' | head -n 1)
if [ -z "$vticket" ]; then
    echo "skip: victim never queued"; kill "$vpid" 2> /dev/null; exit 77
fi
# a pid that cannot be alive: 4194304 is above the default pid_max range in practice, and kill -0 must fail
printf 'pid=4194304\nlabel=victim\n' > "$queue2/$vticket"
start2=$(date +%s)
GPU_HOLD_POLL=1 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_COOLDOWN=0 GPU_HOLD_TCTL_START=0 \
GPU_HOLD_IGNORE_THERMAL_HOLD=1 GPU_TOKEN_FILE=$token2 \
    "$hold" -q "$queue2" -w 20 -t 0.05 -l reaper true >> "$work/log2" 2>&1 &
rpid=$!
for _ in $(seq 1 100); do
    kill -0 "$vpid" 2> /dev/null || break
    sleep 0.1
done
elapsed2=$(( $(date +%s) - start2 ))
kill "$rpid" 2> /dev/null
wait "$vpid" 2> /dev/null
rc2=$?
# the note is per ticket and the victim removes it after quoting it, so nothing accumulates in a shared queue dir
[ -f "$queue2/.reaped.$vticket" ] && { echo "FAIL the victim did not remove its own .reaped note"; fails=$((fails + 1)); }
case $(ls "$queue2" | grep -c '^\.reaped\.') in
    0) ;;
    *) echo "FAIL .reaped notes left behind in the queue"; fails=$((fails + 1)) ;;
esac
case $(cat "$work/log2") in
    *"[$vticket:"*"dropped by pid"*) ;;
    *"$vticket: owner pid"*) ;;
    *) echo "FAIL the victim does not quote the reaper's note:"; sed 's/^/    /' "$work/log2"; fails=$((fails + 1)) ;;
esac
[ "$elapsed2" -le 15 ] || { echo "FAIL the reaped arm took ${elapsed2}s"; fails=$((fails + 1)); }
[ "$rc2" = 75 ] || { echo "FAIL the reaped arm exited $rc2, expected 75"; fails=$((fails + 1)); }

[ "$fails" = 0 ] && { echo "PASS gpu-hold: a removed ticket exits 75 in ${elapsed}s naming $ourticket; a reaped one exits 75 in ${elapsed2}s quoting the reaper"; exit 0; }
echo "FAIL gpu-hold ticket-loss: $fails case(s)"
exit 1
