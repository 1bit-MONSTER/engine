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

# tests/gpu_hold_hold_dir_test.sh — the queue and the token must resolve under the SAME base.
#
# The fleet directory (`$ONEBIT_HOLD_DIR`, /var/lib/1bit-fleet) makes the GPU queue fleet-wide, so a runner under
# another account queues on the same token as an owner's tmux lane. If the two paths resolve under *different* bases --
# which they did while the queue tested `$HOLD/gpu-queue` and the token tested `$HOLD` -- then on a box where the
# directory exists before `gpu-queue` does, two callers hold different token files and both run GPU work. That is the
# silent double-hold the directory exists to prevent, so it is asserted here rather than reasoned about.
#
# Everything happens under a temporary HOME, so the real ~/GPU-QUEUE and ~/GPU-TOKEN are never touched.
#
#   <gpu-hold> -> 0 all three directory states agree, 1 a split was found, 77 cannot run here
set -u

HOLD=${1:?usage: gpu_hold_hold_dir_test.sh <path-to-gpu-hold>}
[ -x "$HOLD" ] || { echo "not executable: $HOLD"; exit 1; }
command -v timeout > /dev/null || { echo "skip: no timeout"; exit 77; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fails=0

# One run per directory state. HOME is the temp dir, so both fallback paths land inside it.
run_state() {  # <state-name> <prepare-cmd, uses $state as the temp home>
    local name=$1 prep=$2 state="$tmp/$1"
    mkdir -p "$state"
    # $state, never $HOME: the prep runs before HOME is overridden, and using $HOME here would create directories in
    # the real home (it did, on the first run of this test).
    ( state="$state"; eval "$prep" )
    local out
    out=$(HOME="$state" ONEBIT_HOLD_DIR="$state/hold" GPU_HOLD_COOLDOWN=0 GPU_HOLD_RELEASE_WAIT=0 \
          GPU_HOLD_WAIT=20 GPU_HOLD_MINUTES=1 GPU_HOLD_THERMAL_FLAG="$state/none" \
          timeout 60 "$HOLD" -l probe -- true 2>&1)
    local rc=$?
    # the token path is logged on take ("took <path>") and on release; the queue base is where the mkdir put it
    local tok
    tok=$(printf '%s\n' "$out" | sed -n 's/^gpu-hold\[[0-9]*\] \(took\|released\) \([^ ]*\).*/\2/p' | head -n 1)
    printf '  %-22s rc=%d token=%s\n' "$name" "$rc" "${tok:-<none>}"
    if [ -z "$tok" ]; then
        echo "    FAIL no token path in the log - the run did not get far enough to test the resolution"
        fails=$((fails + 1))
        return
    fi
    local tok_base; tok_base=$(dirname "$tok")
    # the queue must have been created under the same base the token came from
    if [ -d "$tok_base/gpu-queue" ] || [ -d "$tok_base/GPU-QUEUE" ]; then
        :
    else
        echo "    FAIL token is under $tok_base but no queue directory was created there - the two resolved apart"
        fails=$((fails + 1))
    fi
}

echo "=== queue and token resolve under one base"
run_state "fleet-installed"   'mkdir -p "$state/hold/gpu-queue"'
run_state "fleet-dir-only"    'mkdir -p "$state/hold"'
run_state "no-fleet-dir"      ':'

# and the two lines in the script must test the same thing, which is the invariant the states above observe
q_pred=$(grep -E '^QUEUE=' "$HOLD" | sed -n 's/.*\[ -d "\$HOLD\([^"]*\)" \].*/\1/p')
t_pred=$(grep -E '^TOKEN=' "$HOLD" | sed -n 's/.*\[ -d "\$HOLD\([^"]*\)" \].*/\1/p')
echo "  predicates after \$HOLD: queue=[$q_pred] token=[$t_pred]"
if [ "$q_pred" != "$t_pred" ]; then
    echo "    FAIL QUEUE and TOKEN test different things after \$HOLD (queue=[$q_pred] token=[$t_pred]) - a split state is possible"
    fails=$((fails + 1))
fi

[ "$fails" = 0 ] && echo "ok   queue and token resolve under the same base in all three directory states"
exit "$fails"
