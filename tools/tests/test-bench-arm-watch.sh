#!/bin/bash
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
# `bench-preflight.sh --watch` exists because a start gate describes the wrong moment (engine#572): the second arm of
# the pp4096 pair started at 85 GiB MemAvailable, another load arrived mid-arm, MemAvailable fell to 1-3 GiB with swap
# full, and the arm timed out -- so both the arm and its replacement number were produced on a box nobody had described.
#
# Three arms, each testing something a wrong implementation would get wrong:
#   1. a quiet child: the arm reports start/min/end and exits with the CHILD's status, not 0;
#   2. a child that really allocates: MemAvailable falls below a floor set just under the quiet value, so the watch
#      FAILs -- the mid-arm detection, which is the whole point;
#   3. the same child with the floor left alone: no FAIL, because a drop that stays above the floor is not a
#      contamination, and a gate that fires there would be ignored.
#   4-6. an interrupted arm takes its bench with it (INT, TERM; a child that ignores TERM is KILLed after the grace),
#      grandchildren included, and the arm exits 128+signal;
#   7. BENCH_DROP_GIB reaches the drop test, which is how `bench-preflight.sh --watch` (an exec) sets it.
#
# Arm 1 passes --floor 0: its subject is the exit status, and with the default 20 GiB floor a CI runner with less
# MemAvailable than that FAILs the floor test first and reports 1 (the first CI run of this test, engine#2049).
#
# usage: test-bench-arm-watch.sh tools/bench-arm.sh
set -u
ba=${1:?usage: test-bench-arm-watch.sh path/to/bench-arm.sh}
[ -x "$ba" ] || { echo "not executable: $ba"; exit 1; }
fails=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fails=$((fails+1)); }

avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)

# 1. quiet child, non-zero exit status must survive
out=$("$ba" --floor 0 -- sh -c 'exit 7' 2>&1); rc=$?
[ "$rc" = 7 ] && ok "a quiet arm exits with the child's status (7)" || bad "child status not propagated (got $rc)"
grep -q "mem_avail start " <<<"$out" && ok "and it reports mem start/min/end" || bad "no memory line: $out"

# 2. a child that allocates: --drop-gib 1, so the drop is what trips it and the box's absolute level does not matter
# (a floor relative to a reading taken before the arm is fragile -- other lanes move MemAvailable between the two)
out=$("$ba" --drop-gib 1 --floor 1 -- python3 -c 'import time; x = bytearray(3 * 1024**3); x[::4096] = b"\x01" * (len(x) // 4096); time.sleep(2)' 2>&1); rc=$?
[ "$rc" = 1 ] && ok "a mid-arm drop below the floor FAILs the arm" || bad "mid-arm drop not caught (rc=$rc): $out"
grep -qi 'DURING the arm' <<<"$out" && ok "and says the drop happened during the arm" || bad "message does not say when: $out"

# 3. the same allocation with the default floor: no FAIL (the box has room), child status preserved
out=$("$ba" --drop-gib 64 --floor 1 -- python3 -c 'x = bytearray(3 * 1024**3); x[::4096] = b"\x01" * (len(x) // 4096)' 2>&1); rc=$?
[ "$rc" = 0 ] && ok "a 3 GiB allocation under a 64 GiB drop limit is not a contamination" || bad "false positive (rc=$rc): $out"

# 4-6. interrupt the arm. A non-interactive shell starts a background job with SIGINT ignored, and a signal ignored on
# entry cannot be trapped, so start the arm through a shim that restores the default first (as a terminal would).
alive() { pgrep -f "^sleep $1\$" >/dev/null; }
interrupt_case() {  # interrupt_case <signal> <want_rc> <name> <child command...>
    local sig=$1 want=$2 name=$3; shift 3
    BENCH_ARM_GRACE_S=1 python3 -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])' \
        bash "$ba" --floor 0 -- "$@" >/dev/null 2>&1 &
    local arm=$! i=0
    while ! alive 3172 && [ "$i" -lt 50 ]; do sleep 0.1; i=$((i + 1)); done
    alive 3172 || { bad "$name: the child never started"; kill -KILL "$arm" 2>/dev/null; return; }
    kill "-$sig" "$arm"
    # bounded: an arm with no handler can sit out the signal (bash defers it while a foreground `sleep` runs and then
    # carries on), and an unbounded wait would hang the test instead of failing it
    i=0; while kill -0 "$arm" 2>/dev/null && [ "$i" -lt 50 ]; do sleep 0.1; i=$((i + 1)); done
    if kill -0 "$arm" 2>/dev/null; then
        bad "$name: the arm ignored SIG$sig"; kill -KILL "$arm"; wait "$arm" 2>/dev/null; pkill -KILL -f '^sleep 317[12]$'
        return
    fi
    wait "$arm"; local rc=$?
    i=0; while { alive 3171 || alive 3172; } && [ "$i" -lt 30 ]; do sleep 0.1; i=$((i + 1)); done
    if alive 3171 || alive 3172; then
        bad "$name: the bench outlived the arm (rc=$rc)"; pkill -KILL -f '^sleep 317[12]$'
    elif [ "$rc" != "$want" ]; then
        bad "$name: arm exited $rc, want $want"
    else
        ok "$name"
    fi
}
interrupt_case INT 130 "SIGINT to the arm stops the bench and its worker (rc 130)" sh -c 'sleep 3171 & sleep 3172; wait'
interrupt_case TERM 143 "SIGTERM to the arm stops the bench and its worker (rc 143)" sh -c 'sleep 3171 & sleep 3172; wait'
interrupt_case TERM 143 "a bench that ignores TERM is KILLed after the grace" sh -c 'trap "" TERM; sleep 3171 & sleep 3172; wait'

# 7. the env form of --drop-gib: what `bench-preflight.sh --watch` passes through its exec
out=$(BENCH_DROP_GIB=1 "$ba" --floor 1 -- python3 -c 'import time; x = bytearray(3 * 1024**3); x[::4096] = b"\x01" * (len(x) // 4096); time.sleep(2)' 2>&1); rc=$?
[ "$rc" = 1 ] && grep -q 'limit 1)' <<<"$out" && ok "BENCH_DROP_GIB=1 reaches the drop test" || bad "BENCH_DROP_GIB not honoured (rc=$rc): $out"

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "all checks passed"
