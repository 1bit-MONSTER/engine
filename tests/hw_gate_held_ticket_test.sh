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
# tests/hw_gate_held_ticket_test.sh -- hw-gate keeps ONE box-hold ticket across consecutive GPU steps (orchestrator, 27 Nov).
#
# Run 46014683686 lost the GPU to a lane in the ~1 s between two of its steps' tickets. The held ticket's helpers
# (hold_acquire / hold_for / hold_release) are EXTRACTED FROM scripts/hw-gate.sh and driven against the REAL
# scripts/appliance/box-hold on a private BOX_HOLD_DIR. No device, no build.
#   1. three steps that fit take ONE ticket, and the lock stays held between them
#   2. a step that does not fit in the time left releases the ticket and takes a new one
#   3. the step's process holds no lock descriptor (a killed or leftover step cannot keep the box)
#   4. hold_release frees the lock; a gate killed with SIGKILL frees it too (its holder notices within seconds)
#   5. the lock never coming round -> hold_for returns box-hold's 4
#   6. HOLD_BUDGET is the time left on the ticket minus the margin
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
gate=$root/scripts/hw-gate.sh BH=$root/scripts/appliance/box-hold
T=$(mktemp -d); trap 'kill $(jobs -p) 2> /dev/null; rm -rf "$T"' EXIT
fails=0; ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; fails=$((fails + 1)); }
fn=$T/fn.sh
sed -n '/^HELD_PID="" HELD_T0=0/,/^box_ticket() {/p' "$gate" | sed '$d' > "$fn"
grep -q '^hold_for() {' "$fn" && grep -q '^hold_acquire() {' "$fn" && grep -q '^hold_release() {' "$fn" \
  || { echo "FAIL: the held-ticket helpers are no longer where this test finds them"; exit 1; }
unset GITHUB_ACTIONS GITHUB_RUN_ID
export BOX_HOLD_DIR=$T/locks BOX_HOLD_POLL=0.2 TMPDIR=$T
free_now() { ( exec 8> "$T/locks/box-gpu.lock"; flock -n 8 ) 2> /dev/null; }   # 0 = nobody holds the GPU lock
seq_n() { cat "$T/locks/box-seq" 2> /dev/null || echo 0; }
drive() {   # runs a script in a shell that has the helpers, BOX_HOLD=real box-hold
  BOX_HOLD=$BH BOX_HOLD_WAIT_S=${W:-10} HW_GATE_HOLD_SECONDS=${HS:-30} HW_GATE_HOLD_MARGIN_S=${HM:-2} bash -c ". '$fn'; $1"
}

# 1 + 3 + 6
out=$(drive 'for i in 1 2 3; do hold_for 2 || exit 9; echo "budget $HOLD_BUDGET"; [ -n "$HELD_PID" ] || echo NOHOLD
  ls -l /proc/self/fd 2> /dev/null | grep -q box-gpu.lock && echo FDLEAK
  ( exec 8> "$BOX_HOLD_DIR/box-gpu.lock"; flock -n 8 ) 2> /dev/null && echo FREE-BETWEEN; sleep 0.5; done; hold_release')
[ "$(seq_n)" = 1 ] && ok "three steps that fit: one ticket (box-hold called once)" || bad "tickets taken: $(seq_n) ($out)"
! grep -q FREE-BETWEEN <<< "$out" && ! grep -q NOHOLD <<< "$out" && ok "the lock stays held between the steps" || bad "lock not held between steps: $out"
! grep -q FDLEAK <<< "$out" && ok "the step's process holds no lock descriptor" || bad "a step holds the lock fd"
b=$(grep -m1 '^budget' <<< "$out" | cut -d' ' -f2); [ -n "$b" ] && [ "$b" -le 28 ] && [ "$b" -ge 25 ] && ok "HOLD_BUDGET = time left minus the margin ($b of 30-2 s)" || bad "budget: $out"
sleep 2; free_now && ok "hold_release frees the lock" || bad "lock still held after hold_release"

# 2. a step that does not fit -> new ticket
echo 0 > "$T/locks/box-seq"
out=$(HS=6 HM=1 drive 'hold_for 1; t0=$HELD_T0; sleep 2; hold_for 4; [ "$HELD_T0" != "$t0" ] && echo NEWTICKET; hold_release')
[ "$(seq_n)" = 2 ] && grep -q NEWTICKET <<< "$out" && ok "a step that does not fit in the time left takes a new ticket" || bad "no re-acquire: seq $(seq_n) $out"

# 4b. the gate killed with SIGKILL: its holder lets go
( BOX_HOLD=$BH BOX_HOLD_WAIT_S=10 HW_GATE_HOLD_SECONDS=60 HW_GATE_HOLD_MARGIN_S=2 bash -c ". '$fn'; hold_for 1; echo held > '$T/g'; sleep 60" & echo $! > "$T/gate.pid"; wait ) &
for _ in $(seq 50); do [ -e "$T/g" ] && break; sleep 0.2; done
free_now && bad "the gate's ticket is not held" || ok "the gate holds the lock"
kill -9 "$(cat "$T/gate.pid")"
for _ in $(seq 20); do free_now && break; sleep 0.5; done
free_now && ok "a gate killed with SIGKILL frees the lock (its holder exits)" || bad "lock still held after the gate was killed"

# 5. the lock never comes round
sh "$BH" gpu sleep 4 & sleep 0.5
W=1 drive 'hold_for 1; echo "rc $?"' > "$T/o5" 2> /dev/null
grep -q '^rc 4$' "$T/o5" && ok "the lock never coming round: hold_for returns 4" || bad "no-lock rc: $(cat "$T/o5")"
wait

[ "$fails" = 0 ] && { echo "hw_gate_held_ticket_test: pass"; exit 0; }
echo "hw_gate_held_ticket_test: $fails failure(s)"; exit 1
