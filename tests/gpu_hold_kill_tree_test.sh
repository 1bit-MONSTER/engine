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

# A hold that ends must end everything it started, at any depth (engine#1624).
#
# Two defects, each necessary and each invisible without the other:
#   1. kill_tree walked `ps --ppid` again AFTER signalling, so once the root died its children were already
#      reparented and the KILL pass found only the corpse;
#   2. the main shell ran `kill "$watchdog"` the moment the payload's root died, which killed the watchdog
#      inside its own grace loop, so the KILL pass never ran at all.
#
# The payload below is the shape of every orphan this fleet has produced — a root that dies on TERM with a
# child that ignores it — and the abort is forced through GPU_HOLD_TCTL_FILE so the test needs no device and
# no heat. Usage: gpu_hold_kill_tree_test.sh <path-to-gpu-hold>
set -u

HOLD=${1:?usage: gpu_hold_kill_tree_test.sh <path-to-gpu-hold>}
[ -x "$HOLD" ] || { echo "not executable: $HOLD"; exit 1; }

tmp=$(mktemp -d)
pids=$tmp/pids
cleanup() {
    for p in $(cat "$pids" 2> /dev/null); do kill -9 "$p" 2> /dev/null; done
    rm -rf "$tmp"
}
trap cleanup EXIT

fail=0
pass() { printf 'ok   %s\n' "$*"; }
bad() { printf 'FAIL %s\n' "$*"; fail=1; }

# --- case 1: the abort path, with a grandchild that ignores TERM ------------------------------------
mkdir -p "$tmp/q1"
printf '95000' > "$tmp/tctl1" # millidegrees: tctl() reports 95C
cat > "$tmp/inner.sh" << 'INNER'
#!/bin/sh
trap '' TERM
echo $$ >> "$PIDS"
for _ in $(seq 1 300); do sleep 1; done
INNER
cat > "$tmp/root.sh" << 'ROOT'
#!/bin/sh
PIDS="$PIDS" "$1" &
wait
ROOT
chmod +x "$tmp/inner.sh" "$tmp/root.sh"
: > "$pids"
PIDS="$pids" \
GPU_QUEUE_DIR="$tmp/q1" GPU_TOKEN_FILE="$tmp/token1" GPU_HOLD_THERMAL_FLAG="$tmp/none" \
GPU_HOLD_TCTL_FILE="$tmp/tctl1" GPU_HOLD_TCTL_ABORT=50 GPU_HOLD_TCTL_START=200 \
GPU_HOLD_COOLDOWN=0 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_MINUTES=5 GPU_HOLD_WAIT=10 \
    "$HOLD" -l killtree-test -- "$tmp/root.sh" "$tmp/inner.sh" > "$tmp/out1" 2>&1
if [ ! -s "$pids" ]; then
    bad "case 1: the payload never started (test is inconclusive, not a pass)"
    cat "$tmp/out1"
else
    survivors=""
    for p in $(cat "$pids"); do
        kill -0 "$p" 2> /dev/null && survivors="$survivors $p"
    done
    if [ -n "$survivors" ]; then
        bad "case 1: descendant(s)$survivors outlived the ticket"
        grep -E 'ABORT|max hold' "$tmp/out1" || true
    else
        pass "case 1: a TERM-ignoring grandchild does not outlive its ticket"
    fi
fi

# --- case 2: the max-hold path reaches the same tree kill -------------------------------------------
mkdir -p "$tmp/q2"
printf '40000' > "$tmp/tctl2"
: > "$pids"
PIDS="$pids" \
GPU_QUEUE_DIR="$tmp/q2" GPU_TOKEN_FILE="$tmp/token2" GPU_HOLD_THERMAL_FLAG="$tmp/none" \
GPU_HOLD_TCTL_FILE="$tmp/tctl2" GPU_HOLD_TCTL_ABORT=200 GPU_HOLD_TCTL_START=200 \
GPU_HOLD_COOLDOWN=0 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_MINUTES=0.03 GPU_HOLD_WAIT=10 \
    "$HOLD" -l killtree-maxhold -- "$tmp/root.sh" "$tmp/inner.sh" > "$tmp/out2" 2>&1
rc=$?
survivors=""
for p in $(cat "$pids" 2> /dev/null); do kill -0 "$p" 2> /dev/null && survivors="$survivors $p"; done
if [ -n "$survivors" ]; then
    bad "case 2: descendant(s)$survivors outlived a max hold"
elif [ "$rc" != 124 ]; then
    bad "case 2: a max hold should exit 124 (as timeout does), got $rc"
else
    pass "case 2: a max hold ends the tree and still reports 124"
fi

# --- case 3: the ordinary path is not slowed down by the fix ----------------------------------------
mkdir -p "$tmp/q3"
printf '40000' > "$tmp/tctl3"
started=$(date +%s)
GPU_QUEUE_DIR="$tmp/q3" GPU_TOKEN_FILE="$tmp/token3" GPU_HOLD_THERMAL_FLAG="$tmp/none" \
GPU_HOLD_TCTL_FILE="$tmp/tctl3" GPU_HOLD_TCTL_ABORT=200 GPU_HOLD_TCTL_START=200 \
GPU_HOLD_COOLDOWN=0 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_MINUTES=5 GPU_HOLD_WAIT=10 \
    "$HOLD" -l killtree-fast -- true > "$tmp/out3" 2>&1
rc=$?
elapsed=$(( $(date +%s) - started ))
# waiting for a *bounded* watchdog exit must not become the normal path: 5 s is the 2 s poll plus slack
if [ "$rc" != 0 ]; then
    bad "case 3: a trivial payload should exit 0, got $rc"
elif [ "$elapsed" -gt 5 ]; then
    bad "case 3: an ordinary hold took ${elapsed}s; the watchdog wait leaked into the normal path"
else
    pass "case 3: an ordinary hold still ends promptly (${elapsed}s)"
fi

exit "$fail"
