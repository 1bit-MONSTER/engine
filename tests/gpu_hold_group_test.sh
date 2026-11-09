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
# The payload's tree must end even when the payload's ROOT is already gone when the teardown runs (engine#1649).
#
# The tree walk is anchored on the root: `descendants "$root"` after the root has exited is empty, so a snapshot of it
# is empty too and neither TERM nor KILL reaches anything. That is not a corner case here - it is what a lane does when
# it starts a server and returns (`gpu-hold -- bash -c 'server &'`), and it is how a 14 h llama-server held 51 GiB of
# GTT past its ticket. A process GROUP survives its leader, so the group kill covers exactly that shape.
#
# No device, no ticket, ~5 s: a private queue and token file, a `sleep` grandchild that ignores TERM, a payload that
# exits by itself. The pid this test signals is read from a file the payload wrote and is only ever signalled if
# /proc/<pid>/cmdline still carries this run's marker.
set -uo pipefail

hold=$(dirname "$0")/../scripts/gpu-hold
[ -x "$hold" ] || hold=$(dirname "$0")/../scripts/gpu-hold
[ -f "$hold" ] || { echo "skip: no scripts/gpu-hold"; exit 77; }
command -v ps > /dev/null 2>&1 || { echo "skip: no ps"; exit 77; }

scratch=$(mktemp -d)
marker="gpu-hold-group-$$-$RANDOM"
cleanup() {
    p=$(cat "$scratch/pid" 2> /dev/null || true)
    if [ -n "${p:-}" ] && kill -0 "$p" 2> /dev/null && tr '\0' ' ' < "/proc/$p/cmdline" 2> /dev/null | grep -q "$marker"; then
        kill -9 "$p" 2> /dev/null
    fi
    rm -rf "$scratch"
}
trap cleanup EXIT
mkdir -p "$scratch/q"
cat > "$scratch/payload.sh" <<PAYLOAD
#!/bin/bash
( trap '' TERM; exec -a $marker sleep 300 ) &
echo \$! > "$scratch/pid"
sleep 1
exit 0
PAYLOAD
chmod +x "$scratch/payload.sh"

GPU_QUEUE_DIR=$scratch/q GPU_TOKEN_FILE=$scratch/token GPU_HOLD_LOG=$scratch/log \
GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=1 GPU_HOLD_KILL_GRACE=2 \
GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE=$scratch/tctl GPU_HOLD_THERMAL_FLAG=$scratch/none \
    bash "$hold" -t 30 -l "$marker" -- "$scratch/payload.sh" > "$scratch/out" 2>&1
rc=$?

pid=$(cat "$scratch/pid" 2> /dev/null || true)
if [ -z "$pid" ]; then
    echo "FAIL the payload never recorded its grandchild (see $scratch/out)"; sed 's/^/     /' "$scratch/out"; exit 1
fi
if kill -0 "$pid" 2> /dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" 2> /dev/null | grep -q "$marker"; then
    echo "FAIL a TERM-ignoring grandchild ($pid) outlived the hold after the payload's ROOT exited by itself -"
    echo "     the tree walk is anchored on a dead root; the group kill is what covers this shape (engine#1649)"
    exit 1
fi
echo "ok   the grandchild dies even though the payload's root had already exited (hold rc=$rc)"

# `set -m` must not change what the caller sees: the payload's OWN exit status still comes back through `wait`.
cat > "$scratch/status.sh" <<'STATUS'
#!/bin/bash
exit 7
STATUS
chmod +x "$scratch/status.sh"
GPU_QUEUE_DIR=$scratch/q GPU_TOKEN_FILE=$scratch/token GPU_HOLD_LOG=$scratch/log2 \
GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=1 GPU_HOLD_KILL_GRACE=2 \
GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE=$scratch/tctl GPU_HOLD_THERMAL_FLAG=$scratch/none \
    bash "$hold" -t 30 -l "$marker-status" -- "$scratch/status.sh" > "$scratch/out2" 2>&1
src=$?
if [ "$src" = 7 ]; then
    echo "ok   a payload that exits 7 still reports 7 to the caller"
else
    echo "FAIL a payload that exits 7 reported $src through gpu-hold (set -m must not change the status)"; exit 1
fi
# The payload must not read the CALLER's stdin: `set -m` makes a background job keep the shell's stdin, and the fleet's own
# sweep shape is `while read m; do gpu-hold -- bench "$m"; done < list` - without the payload's `< /dev/null` the first job
# swallows the rest of the list and the sweep silently runs once (measured, bash 5.3.9).
cat > "$scratch/eat.sh" <<'EAT'
#!/bin/bash
cat > /dev/null
EAT
chmod +x "$scratch/eat.sh"
consumed=$(printf 'a\nb\nc\n' | { n=0; while read -r m; do
    n=$((n + 1))
    GPU_QUEUE_DIR=$scratch/q GPU_TOKEN_FILE=$scratch/token GPU_HOLD_LOG=$scratch/log3 \
    GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=1 GPU_HOLD_KILL_GRACE=2 \
    GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE=$scratch/tctl GPU_HOLD_THERMAL_FLAG=$scratch/none \
        bash "$hold" -t 30 -l "$marker-eat-$n" -- "$scratch/eat.sh" > /dev/null 2>&1
done; printf '%s' "$n"; })
if [ "$consumed" = 3 ]; then
    echo "ok   the payload cannot swallow the caller's stdin (a 3-entry sweep runs 3 times)"
else
    echo "FAIL a sweep over 3 entries ran $consumed time(s): the payload is reading the caller's stdin"; exit 1
fi
echo "gpu_hold_group: ok"
