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
# engine#2917: an unreadable /proc/<pid>/environ must be COUNTED, and the redirection's own message must not leak.
#
# The sweep's line is `line=$(tr '\0' '\n' < "$d/environ" 2> /dev/null | grep …) || continue`. `2> /dev/null` applies to
# tr/grep, NOT to `< "$d/environ"`, so a root-owned pid prints a bare `Permission denied` into the ticket log (I spent a
# turn deciding whether that line was mine) and, because `|| continue` runs before the counter, the skipped process is
# never counted -- the very warning that exists for it (gpu-hold's teardown note) under-reports.
#
# Measured, and it also refutes the issue's stated cause: bash does NOT abort the script on that redirection failure, so
# the sweep cannot be what returned 2 to a ticket. Against a real EACCES target (`/proc/1/environ`, root-owned):
#
#   ungrouped (before)   "Permission denied" printed, count 0, loop continued
#   grouped   (after)    nothing printed,            count 1, loop continued
#
# Arms:
#   1. the grouped form counts the unreadable process and prints nothing;
#   2. the ungrouped form is the arm that would have passed a counter test written only against the fixed code: it counts
#      0 and prints the line, so this test fails if someone "simplifies" the grouping away;
#   3. the grouping does not change what a READABLE environ does: the ticket line is still found.
# No device, no gpu-hold needed: this is the shell semantics the script's line depends on.
set -u
fails=0
note() { printf '  %s\n' "$*"; }

# The unreadable target is a DIRECTORY, not /proc/PID/environ: `< dir` fails the same way `< "$d/environ"` does on
# EACCES, it behaves identically for root and non-root, and the sweep's question -- does the failure get counted, and
# does its message leak? -- is answerable either way. A /proc/environ arm alone would skip forever in CI, which is a
# green row that never runs (that is engine#2912's complaint from the other side).
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ghenv.XXXXXX") || exit 77
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/unreadable"

# 1. grouped: the redirect is inside the pipeline that `||` guards -- counted once, nothing printed
out=$(bash -c '
set -u
n=0
for d in "$1"; do
    line=$( { tr "\0" "\n" < "$d"; } 2> /dev/null | grep -m1 -F "GPU_HOLD_TICKET=x" ) || { n=$((n + 1)); continue; }
done
printf "%s" "$n"
' _ "$tmp/unreadable" 2>&1)
if [ "$out" = "1" ]; then note "ok   an unreadable environ is counted once and nothing is printed"
else note "FAIL grouped form printed/counted '$out' (want '1')"; fails=$((fails + 1)); fi

# 2. ungrouped: the form before the fix -- the failure leaks a message and is NOT counted, so the engine#2744 warning
#    under-reports. This arm fails if someone removes the grouping, and it is also the measurement that refuted the
#    issue's stated cause: the loop continues either way, so the sweep cannot abort a ticket.
out=$(bash -c '
set -u
n=0
for d in "$1"; do
    line=$(tr "\0" "\n" < "$d" 2> /dev/null | grep -m1 -F "GPU_HOLD_TICKET=x") || continue
done
printf "%s" "$n"
' _ "$tmp/unreadable" 2>&1)
if [ "$out" = "0" ]; then note "ok   the ungrouped form counts nothing, so the teardown warning under-reports (and it does not abort -- the issue's stated cause needed measuring)"
else note "FAIL the ungrouped form counted '$out' (want 0)"; fails=$((fails + 1)); fi

# 3. a real EACCES target behaves the same way, when this user is allowed to see one
if [ -r /proc/1/environ ]; then
    note "skip /proc/1/environ is readable here, so the EACCES arm has nothing to measure (the directory arms above still ran)"
else
    out=$(bash -c '
set -u
n=0
for d in /proc/1; do
    line=$( { tr "\0" "\n" < "$d/environ"; } 2> /dev/null | grep -m1 -F "GPU_HOLD_TICKET=x" ) || { n=$((n + 1)); continue; }
done
printf "%s" "$n"
' 2>&1)
    if [ "$out" = "1" ]; then note "ok   a real EACCES target is counted once as well"
    else note "FAIL EACCES target printed/counted '$out' (want '1')"; fails=$((fails + 1)); fi
    # the message leak only shows up for EACCES: bash prints "Permission denied" for a failed open, and `< dir` is
    # silent, so this half cannot be asserted on the portable target above
    out=$(bash -c '
set -u
for d in /proc/1; do
    line=$(tr "\0" "\n" < "$d/environ" 2> /dev/null | grep -m1 -F "GPU_HOLD_TICKET=x") || continue
done
printf "%s" "$?"
' 2>&1)
    case $out in
        *"Permission denied"*) note "ok   the ungrouped form leaks 'Permission denied' into the ticket log, which is the line a reader has to guess about" ;;
        *) note "FAIL the ungrouped EACCES form printed '$out' (expected the leaked message)"; fails=$((fails + 1));;
    esac
fi

# 4. a readable environ still yields the ticket line
printf 'A=1\0GPU_HOLD_TICKET=want pid 1\0B=2\0' > "$tmp/environ"
found=$(bash -c '
set -u
line=$( { tr "\0" "\n" < "$1"; } 2> /dev/null | grep -m1 -F "GPU_HOLD_TICKET=want pid 1" ) || { echo MISS; exit 0; }
printf "%s" "$line"
' _ "$tmp/environ")
if [ "$found" = "GPU_HOLD_TICKET=want pid 1" ]; then note "ok   a readable environ still yields the ticket line"
else note "FAIL grouped form lost a readable environ: '$found'"; fails=$((fails + 1)); fi

if [ "$fails" -ne 0 ]; then note "$fails failure(s)"; exit 1; fi
exit 0
