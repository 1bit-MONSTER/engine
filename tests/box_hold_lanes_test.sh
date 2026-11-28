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
# tests/box_hold_lanes_test.sh -- box-hold v6 LANE RUNS (owner 2026-11-28: "give the box to ONE lane until its job is
# done"), one arm per rule, with stub jobs on a private BOX_HOLD_DIR (no device):
#   1. the FOCUS lane (oldest waiting lane ticket) runs its tickets back to back, before a lane queued in between
#   2. the focus lane has no queued cap; the others keep 2 (a lane whose ticket RUNS keeps the focus meanwhile)
#   3. the focus lane keeps the focus when it re-submits within the grace (a lane chaining steps is not cut off)
#   4. the focus passes to the next lane when the focus lane has nothing queued for the grace
#   5. after the budget the focus passes and the lane goes to the back (its tickets keep their order)
#   6. CI tickets run between the focus lane's tickets (alternating), never during one
#   7. `box-hold focus` prints the focus lane, its remaining tickets, the lane order after it and the holder
#   8. the focus goes by AGE (queue number), not by ticket name: a younger 0-b- ticket of the old board class must not
#      take it from an older 1- lane ticket (measured at the v6 install, 2026-11-28 04:32Z)
#   9. a lane whose queued tickets all have OLDER-copy waiters (no box-hold-v6 marker) is idle to v6: the next lane runs
#      (2026-11-28 05:10Z: such a focus lane left the box idle 36 min)
#  10. a holder SIGKILLed during its command does not pin the focus: the next lane runs after the grace (review cl-flm)
#  11. at the deadline, a leftover the command forked (it inherited the lock descriptors) is reaped, so the next ticket
#      runs at once; a leftover at a normal end is logged, not killed (review cl-flm)
set -uo pipefail
BH=${1:?usage: box_hold_lanes_test.sh <box-hold>}
T=$(mktemp -d) || exit 1
trap 'kill $(jobs -p) 2> /dev/null; rm -rf "${T:?}"' EXIT
unset GITHUB_ACTIONS BOX_HOLD_WAIT
export BOX_HOLD_DIR=$T/locks BOX_HOLD_POLL=0.2 BOX_HOLD_FOCUS_GRACE=1 BOX_HOLD_SPLIT_FILE=$T/no-split   # v7: switch off
fail=0; ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; fail=1; }
job() { printf 'if [ -e %q/busy ]; then echo overlap >> %q/overlap; fi; touch %q/busy; echo %s >> %q/order; sleep %s; rm -f %q/busy\n' "$T" "$T" "$T" "$1" "$T" "$2" "$T"; }
cp "$(command -v bash)" "$T/Runner.Worker"
ci() { GITHUB_ACTIONS=true BOX_HOLD_PRIORITY=ci "$T/Runner.Worker" -c 'sh "$0" "$@"; exit $?' "$BH" "$@"; }
reset() { : > "$T/order"; rm -f "$T/overlap"; mkdir -p "$T/locks"; for f in box-focus box-running box-focus-done box-last; do : > "$T/locks/$f"; done; rm -rf "${T:?}/locks/box-ages" 2> /dev/null; }
L() { BOX_HOLD_WHO="$1" sh "$BH" gpu sh -c "$(job "$2" "${3:-0.2}")" 2>> "$T/err"; }   # <label> <tag> [secs]
order() { tr '\n' ' ' < "$T/order"; }

# 1. focus runs back to back
reset; L cl-h H 1.5 & sleep 0.3
L cl-a A1 & sleep 0.2; L cl-b B1 & sleep 0.2; L cl-a A2 & sleep 0.2
wait
[ "$(order)" = "H A1 A2 B1 " ] && ok "the focus lane runs back to back: H A1 A2 B1 (A2, queued after B1, runs first)" || bad "focus run: $(order)"
[ ! -e "$T/overlap" ] && ok "  ... never two jobs at once" || bad "overlap"

# 2. no cap for the focus lane, cap 2 for the others
reset; L cl-a A0 2 & sleep 0.3                           # cl-a has the focus (its ticket runs)
for i in 1 2 3 4 5; do L cl-a "A$i" & sleep 0.15; done  # the focus lane queues its whole job
L cl-b B1 & sleep 0.15; L cl-b B2 & sleep 0.15
BOX_HOLD_WHO=cl-b sh "$BH" gpu true 2> "$T/e2"; rc=$?
[ "$rc" = 4 ] && grep -q "lane cap: 2 queued for cl-b" "$T/e2" && ok "another lane keeps the cap of 2 (3rd refused)" || bad "cap for others: rc $rc [$(cat "$T/e2")]"
grep -q "lane cap.*cl-a" "$T/err" 2> /dev/null && bad "the focus lane was capped" || ok "the focus lane queued 5 more tickets, none refused"
wait
[ "$(order)" = "A0 A1 A2 A3 A4 A5 B1 B2 " ] && ok "  ... and runs them all before the next lane" || bad "focus job: $(order)"

# 3. grace keeps the focus
reset; export BOX_HOLD_FOCUS_GRACE=3
L cl-a A1 0.5 & sleep 0.2
L cl-b B1 & sleep 1.0                                    # B waits while A1 runs; A1 ends at ~0.5 s
L cl-a A2 &                                              # A re-submits ~0.7 s after its ticket ended: within the grace
wait
[ "$(order)" = "A1 A2 B1 " ] && ok "a re-submission within the grace keeps the focus (A1 A2 B1)" || bad "grace: $(order)"
export BOX_HOLD_FOCUS_GRACE=1

# 4. the focus passes after the grace
reset; s=$(date +%s)
L cl-a A1 0.3 & sleep 0.2
L cl-b B1 & wait; d=$(( $(date +%s) - s ))
[ "$(order)" = "A1 B1 " ] && [ "$d" -le 4 ] && ok "with nothing queued for the grace, the focus passes (A1 B1, ${d}s)" || bad "pass on idle: $(order) ${d}s"
grep -q "focus: cl-a has had nothing queued" "$T/locks/box-hold.log" && ok "  ... and the hand-over is logged" || bad "no hand-over line"

# 5. the budget, with no wall-clock race (flaky under CI load twice: #3608, #3611, then #3633's build): A1 waits on a gate
#    file; once A2, A3 and B1 are all queued, the focus record is written (under its lock) as already past the budget while
#    A1 still runs (the budget is not enforced during a focus-lane ticket); then the gate opens. When A1 ends the focus
#    must pass: cl-b runs, then cl-a again, its tickets in their own order.
reset; export BOX_HOLD_FOCUS_BUDGET=600; rm -f "$T/go"
BOX_HOLD_WHO=cl-a sh "$BH" gpu sh -c "echo A1 >> $T/order; i=0; while [ ! -e $T/go ] && [ \$i -lt 600 ]; do sleep 0.05; i=\$((i + 1)); done" 2>> "$T/err" &
for _i in $(seq 1 100); do grep -qx A1 "$T/order" 2> /dev/null && break; sleep 0.1; done      # A1 runs: cl-a has the focus
# submitted ONE AT A TIME, each queued before the next (concurrent submissions race for their numbers: A3 before A2)
queued() { [ "$(cat "$T"/locks/queue/* 2> /dev/null | grep -c "^$1 ")" -ge "$2" ]; }
_q=1
L cl-a A2 & for _i in $(seq 1 100); do queued cl-a 1 && break; sleep 0.1; done; queued cl-a 1 || _q=0
L cl-a A3 & for _i in $(seq 1 100); do queued cl-a 2 && break; sleep 0.1; done; queued cl-a 2 || _q=0
L cl-b B1 & for _i in $(seq 1 100); do queued cl-b 1 && break; sleep 0.1; done; queued cl-b 1 || _q=0
[ "$_q" = 1 ] || bad "budget arm SETUP: A2, A3, B1 not all queued within 10 s each (a slow box, not a scheduling result)"
_now=$(date +%s); flock "$T/locks/box-focus.lock" sh -c "echo 'cl-a $((_now - 1000)) $_now' > '$T/locks/box-focus'"   # budget used
touch "$T/go"; wait
o=$(order); a2=${o%%A2*}; a3=${o%%A3*}
[ "${o%% *}" = A1 ] && [ "$(echo "$o" | awk '{ print $2 }')" = B1 ] && [ ${#a2} -lt ${#a3} ] \
  && ok "after its budget the focus lane goes to the back: $o (B1 before cl-a's next tickets; A2, A3 keep their order)" || bad "budget: $o"
grep -q "focus: cl-a used its 600s budget" "$T/locks/box-hold.log" && ok "  ... logged" || bad "no budget line"
unset BOX_HOLD_FOCUS_BUDGET

# 6. CI between the focus lane's tickets
reset; ci gpu sh -c "$(job C0 1)" & sleep 0.3
L cl-a A1 & sleep 0.15; L cl-a A2 & sleep 0.15; L cl-a A3 & sleep 0.15
ci gpu sh -c "$(job C1 0.2)" & sleep 0.15; ci gpu sh -c "$(job C2 0.2)" & sleep 0.15
wait
[ "$(order)" = "C0 A1 C1 A2 C2 A3 " ] && ok "CI runs between the focus lane's tickets: C0 A1 C1 A2 C2 A3" || bad "CI interleave: $(order)"
[ ! -e "$T/overlap" ] && ok "  ... never during one" || bad "overlap"

# 7. box-hold focus
reset; L cl-a A0 2 & sleep 0.3                           # cl-a has the focus
L cl-a A1 & sleep 0.2; L cl-a A2 & sleep 0.2; L cl-b B1 & sleep 0.3
out=$(sh "$BH" focus)
echo "$out" | grep -q "^focus: cl-a" && echo "$out" | grep -q "^remaining for cl-a:" && [ "$(echo "$out" | grep -c '^  1-')" = 2 ] \
  && echo "$out" | grep -q "1\. cl-b (1 queued)" && ok "box-hold focus: the focus lane, its 2 remaining tickets, then cl-b" || bad "focus print: $out"
echo "$out" | grep -q "^holder gpu: cl-a " && ok "  ... and who holds the box now (cl-a's running ticket)" || bad "no holder line: $out"
wait

# 8. focus by age, not by name
reset; mkdir -p "$T/locks/queue"
sleep 60 & p1=$!; s1=$(sed 's/.*) //' "/proc/$p1/stat" | cut -d' ' -f20)
sleep 60 & p2=$!; s2=$(sed 's/.*) //' "/proc/$p2/stat" | cut -d' ' -f20)
printf 'cl-a %s x\nbox-hold-v3\nbox-hold-v6\n' "$(date -u +%FT%TZ)" > "$T/locks/queue/1-$(printf '%012d' 5).$p1.$s1"     # older
printf 'cl-lb-q %s y\nbox-hold-v3\n' "$(date -u +%FT%TZ)" > "$T/locks/queue/0-b-$(printf '%012d' 9).$p2.$s2"           # younger, old board name
out=$(sh "$BH" focus)
echo "$out" | grep -q "^focus: cl-a" && ok "the focus goes to the OLDEST lane ticket's lane (cl-a #5), not to a younger 0-b- name (cl-lb #9)" || bad "focus by name: $out"
kill "$p1" "$p2" 2> /dev/null; wait "$p1" "$p2" 2> /dev/null

# 9. old-copy-only lane: skipped, not a stall
reset; mkdir -p "$T/locks/queue"
sleep 60 & p1=$!; s1=$(sed 's/.*) //' "/proc/$p1/stat" | cut -d' ' -f20)
printf 'cl-old %s x\nbox-hold-v3\n' "$(date -u +%FT%TZ)" > "$T/locks/queue/1-$(printf '%012d' 1).$p1.$s1"   # older copy's waiter: never starts on v6's word
s=$(date +%s); timeout 10 env BOX_HOLD_WHO=cl-b sh "$BH" gpu sh -c "$(job B1 0.2)"; rc=$?; d=$(( $(date +%s) - s ))
[ "$rc" = 0 ] && [ "$(order)" = "B1 " ] && [ "$d" -le 3 ] && ok "a lane with only older-copy waiters is skipped: B1 ran in ${d}s (no stall)" || bad "old-copy focus stall: rc $rc $(order) ${d}s"
out=$(sh "$BH" focus)   # captured: under pipefail, grep -q would SIGPIPE the writer
echo "$out" | grep -q "^older-copy waiters (skipped by v6.*: 1$" && ok "  ... and box-hold focus counts it" || bad "old-copy count: $out"
kill "$p1" 2> /dev/null; wait "$p1" 2> /dev/null

# 10. SIGKILLed holder
reset
BOX_HOLD_WHO=cl-aaa sh "$BH" both sh -c 'sleep 2' & pk=$!; sleep 0.6
kill -9 "$pk"; wait "$pk" 2> /dev/null; sleep 2   # its command ends on its own
s=$(date +%s); timeout 10 env BOX_HOLD_WHO=cl-bbb sh "$BH" both sh -c "$(job B-RAN 0.1)"; rc=$?; d=$(( $(date +%s) - s ))
[ "$rc" = 0 ] && [ "$(order)" = "B-RAN " ] && ok "a SIGKILLed holder does not pin the focus: the next lane ran (${d}s)" || bad "stale running: rc $rc $(order) [$(cat "$T/locks/box-running")]"

# 11. leftovers
reset; : > "$T/locks/box-hold.log"
s=$(date +%s); BOX_HOLD_LANE_MAX_S=1 BOX_HOLD_REAP_GRACE_S=1 BOX_HOLD_WHO=cl-fork sh "$BH" gpu sh -c 'setsid sleep 30 & sleep 30' & pf=$!   # setsid: out of timeout's process group, as a daemonizing server is
sleep 0.4; timeout 20 env BOX_HOLD_WHO=cl-next sh "$BH" gpu sh -c "$(job N 0.1)" & pn=$!   # a waiter meanwhile: its sleep holds the fds too
wait "$pf"; rc=$?; wait "$pn"; rc2=$?; d=$(( $(date +%s) - s ))
[ "$rc" = 124 ] && [ "$rc2" = 0 ] && [ "$d" -le 12 ] && grep -q "cl-fork: deadline: 1 leftover" "$T/locks/box-hold.log" \
  && ok "at the deadline only the forked leftover is reaped (not the waiter's children): the next ticket ran (${d}s), logged" || bad "deadline reap: rc $rc/$rc2 ${d}s [$(tail -2 "$T/locks/box-hold.log")]"
reset; : > "$T/locks/box-hold.log"
BOX_HOLD_WHO=cl-bg sh "$BH" gpu sh -c 'setsid sleep 1.5 & sleep 0.2; exit 0'; rc=$?
grep -q "cl-bg: ended with 1 leftover" "$T/locks/box-hold.log" && ok "a leftover at a normal end is logged (left running)" || bad "normal-end leftover: rc $rc [$(cat "$T/locks/box-hold.log")]"
sleep 1.6

[ "$fail" = 0 ] && echo "box_hold_lanes: PASS" || echo "box_hold_lanes: FAIL"
exit "$fail"
