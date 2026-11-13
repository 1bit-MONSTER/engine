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

# Run ONE benchmark arm under memory observation (engine#572).
#
# `tools/bench-preflight.sh` refuses to start below a MemAvailable floor, which is a statement about the moment it
# runs. On 2026-11-10 the second arm of a pp4096 pair started at 85 GiB available, another load arrived, MemAvailable
# fell to 1-3 GiB with swap full, and the arm timed out at 540 s: the number that came back (a timeout) and the arm
# that did not were both produced on a box nobody had described. This samples for the whole command and prints
# start/min/end, so an arm is either clean or visibly contaminated where its own number is printed.
#
#   tools/bench-arm.sh [--floor GiB] [--drop-gib N] -- CMD...
#       run CMD, report memory, exit with CMD's status (or 1 if the arm was contaminated)
#
# Two independent contamination tests, because they catch different things:
#   --floor GiB     an ABSOLUTE floor (default 20, the fleet's box floor). What the preflight gates on at start.
#   --drop-gib N    a DROP during the arm: MemAvailable fell by more than N GiB between start and min. This is the one
#                   that describes contamination rather than the box's level, and it does not move when another lane's
#                   memory does -- the absolute reading in a pair can drift between two arms of the same run.
#
# BENCH_RAM_FLOOR_GIB and BENCH_DROP_GIB set the two defaults (20 and 0 = off); the flags override them. The env form is
# what reaches this through `bench-preflight.sh --watch`, which execs this tool.
#
# What --drop-gib cannot see: MemAvailable counts the arm's OWN footprint, so N has to be set above what the arm itself
# maps and allocates -- and an intruder smaller than that margin is indistinguishable from the arm. On a pp4096 arm of
# a large pack the two are the same order, so there the --floor test is the sharper instrument.
#
# An interrupted arm takes its bench with it: CMD runs in its own process group, and INT, TERM or HUP to this tool
# sends TERM to that whole group (KILL after BENCH_ARM_GRACE_S, default 10 s), so a bench that spawns workers does
# not outlive the arm that started it (the 11-08 orphan held GTT for 14 h).
set -uo pipefail

floor=${BENCH_RAM_FLOOR_GIB:-20}
drop=${BENCH_DROP_GIB:-0}
grace=${BENCH_ARM_GRACE_S:-10}
while [ $# -gt 0 ]; do
    case $1 in
        --floor) floor=${2:?--floor needs a GiB value}; shift 2 ;;
        --drop-gib) drop=${2:?--drop-gib needs a GiB value}; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
[ $# -gt 0 ] || { echo "bench-arm: usage: bench-arm.sh [--floor GiB] [--drop-gib N] -- CMD..." >&2; exit 2; }

mem_avail() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
swap_free() { awk '/SwapFree/ {printf "%d", $2/1048576}' /proc/meminfo; }

child=
# Stop the child's whole process group: TERM, then KILL once the grace period is over. Safe to call twice.
stop_arm() {
    [ -n "$child" ] || return 0
    kill -TERM -- "-$child" 2>/dev/null || return 0
    local i=0
    while kill -0 -- "-$child" 2>/dev/null && [ "$i" -lt $((grace * 10)) ]; do sleep 0.1; i=$((i + 1)); done
    kill -KILL -- "-$child" 2>/dev/null
    wait "$child" 2>/dev/null
    child=
}
on_signal() {
    echo "bench-arm: $1 received, stopping the arm's process group" >&2
    stop_arm; trap - EXIT; exit "$2"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP
trap 'stop_arm' EXIT

start=$(mem_avail); min=$start; sw_start=$(swap_free); sw_min=$sw_start
# set -m puts the background job in its own process group (pgid = its pid), so stop_arm can signal the group.
set -m
"$@" &
child=$!
set +m
while kill -0 "$child" 2>/dev/null; do
    m=$(mem_avail); s=$(swap_free)
    [ "$m" -lt "$min" ] && min=$m
    [ "$s" -lt "$sw_min" ] && sw_min=$s
    sleep 0.25
done
wait "$child"; rc=$?
child=
end=$(mem_avail)
# The end reading counts too: a short arm can fall and recover between samples, and the number produced at the end is
# the one a reader would quote.
[ "$end" -lt "$min" ] && min=$end
w_end=$(swap_free)
[ "$w_end" -lt "$sw_min" ] && sw_min=$w_end

printf 'bench-arm: mem_avail start %s min %s end %s GiB; swap_free start %s min %s GiB\n' \
       "$start" "$min" "$end" "$sw_start" "$sw_min"
if [ "$min" -lt "$floor" ]; then
    printf 'bench-arm: FAIL -- MemAvailable fell to %s GiB (floor %s) DURING the arm, so this number was not taken on a quiet box: say so, or discard it\n' \
           "$min" "$floor" >&2
    exit 1
fi
if [ "$drop" -gt 0 ] && [ $((start - min)) -gt "$drop" ]; then
    printf 'bench-arm: FAIL -- MemAvailable fell %s GiB during the arm (limit %s), so another load arrived mid-run: this number was not taken on a quiet box\n' \
           "$((start - min))" "$drop" >&2
    exit 1
fi
exit "$rc"
