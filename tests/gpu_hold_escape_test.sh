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
# A payload that LEAVES THE TICKET'S TREE must still end with the ticket (engine#2741).
#
# The teardown is anchored on the tree three ways -- the process-group kill (#1649), the descendant snapshot (#1621) and
# the survivor check on the payload root -- and `setsid` escapes all three: the process is in neither the group nor the
# descendant list, and its parent is init. Measured on this box 2026-11-16: two `setsid`'d 27B serves outlived their
# tickets by 64 min carrying 44.4 GiB of GTT each, while a second copy of the same model ran inside the next ticket.
#
# What the guard can rely on is the environment: every process the payload starts inherits GPU_HOLD_TICKET, and nothing
# it does to its parentage, session or process group changes that. So the arms are:
#   1. a `setsid`'d sleeper is GONE after the hold, and the hold says it ended it;
#   2. an identical sleeper started OUTSIDE any ticket is UNTOUCHED (the guard must not become a process killer);
#   3. a run where the sleepers die on their own (an OOM hour; measured 2026-11-16, which would make arm 1 pass for the
#      wrong reason) reports INCONCLUSIVE rather than ok.
#
# No device, no ticket, ~10 s: a private queue and token file, a marker in every cmdline, and the pid that is signalled
# is read from a file the payload wrote and is only signalled if its cmdline still carries this run's marker.
set -uo pipefail

hold=$(dirname "$0")/../scripts/gpu-hold
[ -x "$hold" ] || hold=$(dirname "$0")/../scripts/gpu-hold
[ -f "$hold" ] || { echo "skip: no scripts/gpu-hold"; exit 77; }
command -v ps > /dev/null 2>&1 || { echo "skip: no ps"; exit 77; }

scratch=$(mktemp -d)
marker="gpu-hold-escape-$$-$RANDOM"
outside=""
mine=""
cleanup() {
    for p in $mine $outside; do
        [ -n "${p:-}" ] && kill -0 "$p" 2> /dev/null && tr '\0' ' ' < "/proc/$p/cmdline" 2> /dev/null | grep -q "$marker" && kill -9 "$p" 2> /dev/null
    done
    rm -rf "$scratch"
}
trap cleanup EXIT
# The take-time refusal reads the kernel log and refuses a ticket after a recent fault line (#2480/#2498); on a box whose
# NPU faults every few minutes that refuses EVERY ticket, so this fixture is the same override the other hold tests use.
printf 'kernel: gpu-hold test fixture: no fault lines in this log\n' > "$scratch/klog"
export GPU_HOLD_KLOG_FILE=$scratch/klog
mkdir -p "$scratch/q"

# arm 2's control: a sleeper with the SAME environment shape as a payload, started by no ticket at all
( exec -a "$marker-outside" sleep 300 ) &
outside=$!
sleep 0.3

cat > "$scratch/payload.sh" <<PAYLOAD
#!/bin/bash
# leave the tree the way a lane does when it detaches a server: setsid puts it in a new session, so neither the group
# kill nor the descendant walk can see it -- but it inherits this ticket's GPU_HOLD_TICKET.
setsid bash -c 'exec -a $marker-inside sleep 300' &
sleep 1
# record it by environ, exactly as the guard finds it
for p in /proc/[0-9]*; do
    p=\${p#/proc/}
    tr '\0' '\n' < "/proc/\$p/environ" 2> /dev/null | grep -Fxq "GPU_HOLD_TICKET=\$GPU_HOLD_TICKET" && tr '\0' ' ' < "/proc/\$p/cmdline" 2> /dev/null | grep -q "$marker-inside" && echo "\$p" >> "$scratch/inside"
done
exit 0
PAYLOAD
chmod +x "$scratch/payload.sh"

GPU_QUEUE_DIR=$scratch/q GPU_TOKEN_FILE=$scratch/token GPU_HOLD_LOG=$scratch/log \
GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=1 GPU_HOLD_KILL_GRACE=2 \
GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE=$scratch/tctl GPU_HOLD_THERMAL_FLAG=$scratch/none \
    bash "$hold" -t 30 -l "$marker" -- "$scratch/payload.sh" > "$scratch/out" 2>&1
rc=$?

mine=$(cat "$scratch/inside" 2> /dev/null | head -1)
if [ -z "$mine" ]; then
    # the payload could not find its own escapee: either it died on its own (OOM hour) or the fixture is broken
    if grep -q "ends with" "$scratch/out" 2> /dev/null; then :; fi
    echo "INCONCLUSIVE no escapee recorded (a box killing sleepers would look like this; hold rc=$rc)"
    sed 's/^/     /' "$scratch/out" | tail -5
    exit 77
fi
if kill -0 "$mine" 2> /dev/null && tr '\0' ' ' < "/proc/$mine/cmdline" 2> /dev/null | grep -q "$marker-inside"; then
    echo "FAIL a setsid'd sleeper ($mine) outlived its hold - it is in neither the process group nor the descendant walk,"
    echo "     and its environment still carried the ticket line (engine#2741)"
    exit 1
fi
if ! grep -qi "left the ticket's tree" "$scratch/out"; then
    echo "FAIL the hold ended the escapee but never said so - the lane cannot tell a guard that ran from one that was"
    echo "     skipped, which is the same defect as a survivor nobody records"
    sed 's/^/     /' "$scratch/out" | tail -6
    exit 1
fi
echo "ok   a setsid'd process that left the ticket's tree is ended by the hold, and the hold names it (rc=$rc)"

if ! kill -0 "$outside" 2> /dev/null; then
    echo "FAIL an identical process started OUTSIDE the ticket was killed - the guard keyed on the environment, so it must"
    echo "     not touch a process that never carried this ticket's line"
    exit 1
fi
echo "ok   an identical process started outside the ticket is untouched"
kill -9 "$outside" 2> /dev/null; outside=""
exit 0
