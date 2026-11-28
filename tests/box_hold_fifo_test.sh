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
# box-hold serves waiters in order (stub jobs, a private BOX_HOLD_DIR, no device).
#   1. five waiters queued one after another behind a holder run in that order, never two at once
#   2. a waiter killed with SIGKILL while queued is skipped: the one behind it still runs
#   3. BOX_HOLD_WAIT runs out behind a holder -> exit 4, its ticket leaves the queue
#   4. BOX_HOLD_WAIT runs out while still queued behind another waiter -> exit 4 ("still queued")
#   5. the command's rc comes back; the .who files name the holder
#   6. CI class (BOX_HOLD_PRIORITY=ci): served right after the holder, ahead of lanes queued earlier; FIFO among CI
#   7. a killed CI waiter is skipped: the lanes behind it still run
#   8. a ticket of the previous name format (lanes queued before the upgrade) is served before new lanes, after CI
#   9. a CI ticket's command is capped at BOX_HOLD_CI_MAX (rc 124)
#  10. the command's stderr reaches the caller (a bare `exec 9>lock 2>/dev/null` used to discard every job's stderr)
#  11. BOX_HOLD_PRIORITY=ci set by hand (no runner) is IGNORED: a lane ticket, and box-hold says so (stderr + box-hold.log)
#  12. under a Runner.Worker but without GITHUB_ACTIONS=true: still a lane ticket
#  13. v6: the board class is folded into lane runs - BOX_HOLD_PRIORITY=board queues a LANE ticket (reported)
#  14. (v6) a lane ticket's command is capped at BOX_HOLD_LANE_MAX_S (90 min on the box; rc 124)
#  17. lane cap: 2 queued tickets of one lane -> a 3rd from that lane is refused AT ONCE (exit 4, "lane cap"), not queued
#  18. a running ticket does not count: the lane's holder + 2 queued -> a 3rd is still queued
#  19. lanes are separate: another lane, and a dashed lane name from BOX_HOLD_LANES (cl-inkling-npu vs cl-inkling)
#  20. CI and board tickets are not counted or capped; a label of no cl-/pi- lane is not capped
#  21. alternation under a stream of CI tickets: lanes waiting + CI tickets queued behind them -> CI, lane, CI, lane...
#      (lanes progress every other turn; CI never twice in a row while a lane waits); with no lane waiting CI runs back
#      to back
#  22. an older copy's lane waiter (its ticket has no box-hold-v3 marker) is NOT preferred after a priority turn: the
#      next CI ticket runs at once instead of the queue stalling on a waiter that never takes the lock (21:57Z 11-27)
#  23. (v6, orchestrator 2026-11-28) an older copy's lane ticket is SKIPPED by v6: the younger v6 ticket takes the lane turn
#  24. v5: a lane ticket whose wait ran out and is re-submitted (same label + command) keeps its ORIGINAL age: it runs
#      before a lane ticket first queued after it; once it has run, the same job queued again is a new, young ticket
#  25. v5: a lane ticket waits for ever by default (no BOX_HOLD_WAIT) and with BOX_HOLD_WAIT=0; a CI ticket keeps 3600
#  26. v5.1: an OLDER copy's ticket file (no age record, no marker) gets its record seeded; when that waiter's wait runs
#      out and the job is re-submitted (same label + command) it keeps its place: H OLD B, not H B OLD. `box-hold seed`
#      seeds by hand, and seeding twice changes nothing
set -u
BH=${1:?usage: box_hold_fifo_test.sh <box-hold>}
T=$(mktemp -d); trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$T"' EXIT
export BOX_HOLD_DIR=$T/locks BOX_HOLD_POLL=0.2 BOX_HOLD_FOCUS_GRACE=1 BOX_HOLD_SPLIT_FILE=$T/no-split   # v7: switch off whatever the box has; v6: a 1 s focus grace (300 s on the box) so lanes hand over fast
unset GITHUB_ACTIONS   # this test runs under a real Runner.Worker in CI: only ci() below may look like CI
fail=0; ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; fail=1; }
cp "$(command -v bash)" "$T/Runner.Worker"   # a process's comm is its executable's name: this is what box-hold looks for
# ci <box-hold args...>: a ticket from "real CI" (GITHUB_ACTIONS=true, under a Runner.Worker); the trailing `exit $?`
# keeps bash from exec-ing the last command (so Runner.Worker stays box-hold's ancestor) and passes its rc on
ci() { GITHUB_ACTIONS=true BOX_HOLD_PRIORITY=ci "$T/Runner.Worker" -c 'sh "$0" "$@"; exit $?' "$BH" "$@"; }
killtree() { local c; for c in $(pgrep -P "$1"); do killtree "$c"; done; kill -9 "$1" 2> /dev/null; }
job() { # <tag> <secs>: record start order, flag any overlap
  printf 'if [ -e %q/busy ]; then echo overlap >> %q/overlap; fi; touch %q/busy; echo %s >> %q/order; sleep %s; rm -f %q/busy\n' \
    "$T" "$T" "$T" "$1" "$T" "$2" "$T"
}

# 1. order
: > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
sh "$BH" gpu sh -c "$(job H 2)" & sleep 0.5
for i in 1 2 3 4 5; do BOX_HOLD_WHO=w$i sh "$BH" both sh -c "$(job $i 0.3)" & sleep 0.3; done
wait
[ "$(tr '\n' ' ' < "$T/order")" = "H 1 2 3 4 5 " ] && ok "waiters served in arrival order" || bad "order: $(tr '\n' ' ' < "$T/order")"
[ ! -e "$T/overlap" ] && ok "never two jobs at once" || bad "overlap seen"
[ -z "$(ls "$T/locks/queue")" ] && ok "queue empty afterwards" || bad "queue left: $(ls "$T/locks/queue")"

# 2. killed waiter
: > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
sh "$BH" gpu sh -c "$(job H 2)" & sleep 0.5
sh "$BH" gpu sh -c "$(job A 0.2)" & sleep 0.3
sh "$BH" gpu sh -c "$(job B 0.2)" & b=$!; sleep 0.3
sh "$BH" gpu sh -c "$(job C 0.2)" & sleep 0.3
kill -9 "$b"
wait
[ "$(tr '\n' ' ' < "$T/order")" = "H A C " ] && ok "killed waiter skipped, the next one runs" || bad "after kill: $(tr '\n' ' ' < "$T/order")"
[ -z "$(ls "$T/locks/queue")" ] && ok "dead waiter's ticket swept" || bad "queue left: $(ls "$T/locks/queue")"

# 3. timeout behind a holder
sh "$BH" gpu sleep 3 & sleep 0.5
s=$(date +%s); BOX_HOLD_WAIT=1 sh "$BH" gpu true 2> "$T/err3"; rc=$?; d=$(( $(date +%s) - s ))
[ "$rc" = 4 ] && [ "$d" -le 3 ] && ok "wait ran out behind a holder: rc 4 after ${d}s" || bad "timeout: rc $rc after ${d}s ($(cat "$T/err3"))"
wait

# 4. timeout while queued behind a waiter
sh "$BH" gpu sleep 3 & sleep 0.5
BOX_HOLD_WAIT=10 sh "$BH" gpu true & sleep 0.5
BOX_HOLD_WAIT=1 sh "$BH" gpu true 2> "$T/err4"; rc=$?
[ "$rc" = 4 ] && grep -q "still queued" "$T/err4" && ok "wait ran out while queued: rc 4, still queued" || bad "queued timeout: rc $rc ($(cat "$T/err4"))"
wait
[ -z "$(ls "$T/locks/queue")" ] && ok "timed-out ticket left the queue" || bad "queue left: $(ls "$T/locks/queue")"

# 5. rc + who
BOX_HOLD_WHO=lane-x sh "$BH" npu sh -c 'exit 5'; rc=$?
[ "$rc" = 5 ] && ok "command rc passes through" || bad "rc $rc, wanted 5"
grep -q "^lane-x " "$T/locks/box-gpu.who" && grep -q "^lane-x " "$T/locks/box-npu.who" && ok ".who files name the holder (both locks)" || bad "who: $(cat "$T"/locks/box-*.who)"

# 6. CI jumps the line, FIFO among CI
: > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
sh "$BH" gpu sh -c "$(job H 2)" & sleep 0.5
sh "$BH" gpu sh -c "$(job L1 0.2)" & sleep 0.3
sh "$BH" gpu sh -c "$(job L2 0.2)" & sleep 0.3
ci gpu sh -c "$(job C1 0.2)" & sleep 0.3
ci both sh -c "$(job C2 0.2)" & sleep 0.3
wait
[ "$(tr '\n' ' ' < "$T/order")" = "H C1 L1 C2 L2 " ] && ok "CI served next after a lane turn, then a lane (alternating), FIFO within each class" || bad "CI order: $(tr '\n' ' ' < "$T/order")"
[ ! -e "$T/overlap" ] && ok "never two jobs at once with CI in the queue" || bad "overlap seen"

# 7. a dead CI waiter is skipped
: > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
sh "$BH" gpu sh -c "$(job H 1.5)" & sleep 0.5
sh "$BH" gpu sh -c "$(job L1 0.2)" & sleep 0.3
ci gpu sh -c "$(job C1 0.2)" & c=$!; sleep 0.3
killtree "$c"; wait
[ "$(tr '\n' ' ' < "$T/order")" = "H L1 " ] && ok "killed CI waiter skipped, the lane behind it runs" || bad "after CI kill: $(tr '\n' ' ' < "$T/order")"
[ -z "$(ls "$T/locks/queue")" ] && ok "dead CI ticket swept" || bad "queue left: $(ls "$T/locks/queue")"

# 8. previous-format tickets: after CI, before new lanes
: > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
sh "$BH" gpu sh -c "$(job H 1)" & sleep 0.3
sleep 2.5 & old=$!; st=$(sed 's/.*) //' "/proc/$old/stat" | cut -d' ' -f20)
echo "old-lane $(date -u +%FT%TZ) x" > "$T/locks/queue/$(printf '%012d' 1).$old.$st"   # a live waiter of the old script
sh "$BH" gpu sh -c "$(job L1 0.2)" & sleep 0.3
ci gpu sh -c "$(job C1 0.2)" & sleep 0.3
wait
s0=$(grep -n '^C1$' "$T/order" | cut -d: -f1); s1=$(grep -n '^L1$' "$T/order" | cut -d: -f1)
[ "$(tr '\n' ' ' < "$T/order")" = "H C1 L1 " ] && ok "CI passes an old-format ticket; the new lane waits until it is gone" || bad "old-format order: $(tr '\n' ' ' < "$T/order")"
names=$(printf '%s\n' 1-000000000009.1.1 000000000001.1.1 0-000000000010.1.1 | LC_ALL=C sort | cut -c1-2 | tr '\n' ' ')
[ "$names" = "0- 00 1- " ] && ok "name order: CI < previous format < new lanes" || bad "name order: $names"

# 9. the CI cap
s=$(date +%s); BOX_HOLD_CI_MAX=1 ci gpu sleep 5; rc=$?; d=$(( $(date +%s) - s ))
[ "$rc" = 124 ] && [ "$d" -le 3 ] && ok "a CI ticket's command is capped (rc 124 after ${d}s)" || bad "CI cap: rc $rc after ${d}s"

# 10. stderr passes through
sh "$BH" gpu sh -c 'echo to-stderr >&2' 2> "$T/err10"
grep -qx to-stderr "$T/err10" && ok "the command's stderr reaches the caller" || bad "stderr lost: [$(cat "$T/err10")]"

# 11. the variable alone does not make a ticket CI
: > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
sh "$BH" gpu sh -c "$(job H 1.5)" & sleep 0.3
sh "$BH" gpu sh -c "$(job L1 0.2)" & sleep 0.3
BOX_HOLD_WHO=lane-z BOX_HOLD_PRIORITY=ci sh "$BH" gpu sh -c "$(job Z 0.2)" 2> "$T/err11" & sleep 0.3
wait
[ "$(tr '\n' ' ' < "$T/order")" = "H L1 Z " ] && ok "BOX_HOLD_PRIORITY=ci by hand: queued as a lane (behind L1)" || bad "by-hand order: $(tr '\n' ' ' < "$T/order")"
grep -q "BOX_HOLD_PRIORITY=ci ignored for lane-z" "$T/err11" && grep -q "ignored for lane-z" "$T/locks/box-hold.log" \
  && ok "the ignored request is reported on stderr and in box-hold.log" || bad "no report: [$(cat "$T/err11")]"

# 12. a Runner.Worker ancestor without GITHUB_ACTIONS=true is not CI
: > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
sh "$BH" gpu sh -c "$(job H 1.5)" & sleep 0.3
sh "$BH" gpu sh -c "$(job L1 0.2)" & sleep 0.3
BOX_HOLD_PRIORITY=ci "$T/Runner.Worker" -c 'sh "$0" "$@"; exit $?' "$BH" gpu sh -c "$(job W 0.2)" 2> /dev/null & sleep 0.3
wait
[ "$(tr '\n' ' ' < "$T/order")" = "H L1 W " ] && ok "Runner.Worker without GITHUB_ACTIONS=true: a lane" || bad "worker-only order: $(tr '\n' ' ' < "$T/order")"

# 13. v6: board is a lane ticket now
: > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
BOX_HOLD_PRIORITY=board BOX_HOLD_WHO=cl-lb-q38 sh "$BH" gpu sh -c "$(job B1 0.2)" 2> "$T/err13"; rc=$?
[ "$rc" = 0 ] && grep -q "folded into lane runs" "$T/err13" && [ "$(cat "$T/order")" = B1 ] && ok "board asks queue a lane ticket (folded into lane runs, reported)" || bad "board fold: rc $rc [$(cat "$T/err13")]"
# 14. v6: a lane ticket's command is capped
s=$(date +%s); BOX_HOLD_LANE_MAX_S=1 BOX_HOLD_WHO=cl-cap sh "$BH" gpu sleep 5; rc=$?; d=$(( $(date +%s) - s ))
[ "$rc" = 124 ] && [ "$d" -le 3 ] && ok "a lane ticket's command is capped at BOX_HOLD_LANE_MAX_S (rc 124 after ${d}s)" || bad "lane cap timeout: rc $rc after ${d}s"

# 17-20: the lane cap
printf 'cl-inkling-npu\n' > "$T/lanes"; export BOX_HOLD_LANES=$T/lanes
q() { BOX_HOLD_WHO="$1" sh "$BH" gpu sleep "${2:-0.1}" 2>> "$T/errq" & sleep 0.2; }
: > "$T/errq"
sh "$BH" gpu sleep 3 & sleep 0.3
q "cl-inkling a"; q "cl-inkling b"
s=$(date +%s); BOX_HOLD_WHO="cl-inkling d" sh "$BH" gpu true 2> "$T/err17"; rc=$?; d=$(( $(date +%s) - s ))
[ "$rc" = 4 ] && [ "$d" -le 1 ] && grep -q "lane cap: 2 queued for cl-inkling" "$T/err17" \
  && ok "lane cap: the 3rd queued ticket of a lane is refused at once (exit 4)" || bad "lane cap: rc $rc after ${d}s [$(cat "$T/err17")]"
[ -z "$(grep -l 'cl-inkling d' "$T"/locks/queue/* 2> /dev/null)" ] && ok "  ... and it was never queued" || bad "the refused ticket is in the queue"
BOX_HOLD_WHO="cl-inkling-npu x" sh "$BH" gpu true 2> "$T/err19a" & n1=$!
BOX_HOLD_WHO="cl-hf y" sh "$BH" gpu true 2> "$T/err19b" & n2=$!
sleep 0.4
for f in "$T"/locks/queue/*; do cat "$f"; done 2> /dev/null | grep -q '^cl-inkling-npu x' && grep -q '^cl-hf y' $(ls "$T"/locks/queue/* 2> /dev/null) /dev/null \
  && ok "other lanes queue normally, incl. a dashed lane name (cl-inkling-npu is not cl-inkling)" || bad "other lanes: [$(cat "$T/err19a" "$T/err19b")]"
BOX_HOLD_PRIORITY=board BOX_HOLD_WHO=cl-lb-z sh "$BH" gpu true & sleep 0.3
for i in 1 2 3 4; do BOX_HOLD_WHO="coder-llm" sh "$BH" gpu true & done; sleep 1.2   # v6: submissions are serialised by the sequence lock
[ "$(grep -l '^coder-llm' "$T"/locks/queue/* 2> /dev/null | wc -l)" = 4 ] && ok "a label of no cl-/pi- lane is not capped (4 queued)" || bad "coder-llm capped: $(grep -l '^coder-llm' "$T"/locks/queue/* 2> /dev/null | wc -l)"
ls "$T/locks/queue" | grep -q '^1-' && ok "(v6) a board request is a lane ticket" || bad "no lane ticket from the board request"
wait
# 18. the running ticket does not count
: > "$T/errq"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"; sleep 1.2   # no focus left from arm 20
BOX_HOLD_WHO="cl-inkling run" sh "$BH" gpu sleep 2 & sleep 0.3
q "cl-inkling a"
BOX_HOLD_WHO="cl-inkling c" sh "$BH" gpu true 2> "$T/err18"; rc=$?
[ "$rc" = 0 ] && ! grep -q "lane cap" "$T/err18" && ok "the lane's running ticket does not count: a 2nd queued one is accepted" || bad "running counted: rc $rc [$(cat "$T/err18")]"
wait
unset BOX_HOLD_LANES

# 21. alternation: a stream of CI tickets does not starve the lanes
rm -f "$T/locks/box-last"; : > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
sh "$BH" gpu sh -c "$(job H 2)" & sleep 0.3
for l in L1 L2 L3; do sh "$BH" gpu sh -c "$(job $l 0.2)" & sleep 0.2; done
for c in C1 C2 C3 C4; do ci gpu sh -c "$(job $c 0.2)" & sleep 0.2; done
wait
[ "$(tr '\n' ' ' < "$T/order")" = "H C1 L1 C2 L2 C3 L3 C4 " ] && ok "a stream of CI tickets: lanes progress every other turn (H C1 L1 C2 L2 C3 L3 C4)" || bad "alternation: $(tr '\n' ' ' < "$T/order")"
[ ! -e "$T/overlap" ] && ok "  ... never two jobs at once" || bad "overlap seen"
: > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
sh "$BH" gpu sh -c "$(job H 1)" & sleep 0.3
for c in C1 C2 C3; do ci gpu sh -c "$(job $c 0.2)" & sleep 0.2; done
wait
[ "$(tr '\n' ' ' < "$T/order")" = "H C1 C2 C3 " ] && ok "no lane waiting: CI tickets run back to back" || bad "CI only: $(tr '\n' ' ' < "$T/order")"

# 22. no stall on an older copy's lane waiter
rm -f "$T/locks/box-last"; : > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
ci gpu sh -c "$(job C0 1)" & sleep 0.3
sleep 6 & old=$!; st=$(sed 's/.*) //' "/proc/$old/stat" | cut -d' ' -f20)
echo "cl-old-lane $(date -u +%FT%TZ) x" > "$T/locks/queue/1-$(printf '%012d' 2).$old.$st"   # live, no marker line
ci gpu sh -c "$(job C1 0.2)" & c1=$!
s=$(date +%s); wait "$c1"; d=$(( $(date +%s) - s ))
[ "$(tr '\n' ' ' < "$T/order")" = "C0 C1 " ] && [ "$d" -le 4 ] && ok "an older copy's lane waiter is not preferred: the next CI ticket ran after ${d}s (no stall)" \
  || bad "stall on an unmarked lane waiter: $(tr '\n' ' ' < "$T/order") after ${d}s"
kill "$old" 2> /dev/null; wait 2> /dev/null

# 23. older copies' lane tickets are skipped by v6 (2026-11-28 05:10Z: waiting on one left the box idle 36 min): the
#     younger v6 ticket takes the lane turn after C0 (was v5: it waited behind the unmarked one, C0 C1 C2 L2)
rm -f "$T/locks/box-last"; : > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
ci gpu sh -c "$(job C0 1)" & sleep 0.3
sleep 3 & old=$!; st=$(sed 's/.*) //' "/proc/$old/stat" | cut -d' ' -f20)
echo "cl-old-lane $(date -u +%FT%TZ) x" > "$T/locks/queue/1-$(printf '%012d' 3).$old.$st"   # live, no marker, oldest lane
sh "$BH" gpu sh -c "$(job L2 0.2)" & l2=$!; sleep 0.2                                    # a marked lane, younger
ci gpu sh -c "$(job C1 0.2)" & sleep 0.2
ci gpu sh -c "$(job C2 0.2)" &
wait "$l2"; wait 2> /dev/null
[ "$(tr '\n' ' ' < "$T/order")" = "C0 L2 C1 C2 " ] && ok "an older copy's lane ticket is skipped: the v6 ticket takes the lane turn (C0 L2 C1 C2)" \
  || bad "old-copy skip: $(tr '\n' ' ' < "$T/order") (want C0 L2 C1 C2)"

# 24. original age across a re-submission
rm -f "$T/locks/box-last"; rm -rf "${T:?}/locks/box-ages"; : > "$T/order"
sh "$BH" gpu sh -c "$(job H 3)" & sleep 0.3
BOX_HOLD_WAIT=1 BOX_HOLD_WHO="cl-age A" sh "$BH" gpu sh -c "$(job A 0.2)" 2> /dev/null; rcA=$?   # its 1 s runs out
BOX_HOLD_WHO="cl-other B" sh "$BH" gpu sh -c "$(job B 0.2)" & sleep 0.3                    # queued AFTER A first was
BOX_HOLD_WHO="cl-age A" sh "$BH" gpu sh -c "$(job A 0.2)" & sleep 0.3                      # A again: same label + command
wait
[ "$rcA" = 4 ] && [ "$(tr '\n' ' ' < "$T/order")" = "H A B " ] && ok "a re-submitted lane ticket keeps its original age (H A B, not H B A)" \
  || bad "original age: first rc $rcA, order $(tr '\n' ' ' < "$T/order")"
: > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
sh "$BH" gpu sh -c "$(job H 1.5)" & sleep 0.3
BOX_HOLD_WHO="cl-other C" sh "$BH" gpu sh -c "$(job C 0.2)" & sleep 0.3
BOX_HOLD_WHO="cl-age A" sh "$BH" gpu sh -c "$(job A 0.2)" & sleep 0.3                      # A ran: no old age left
wait
[ "$(tr '\n' ' ' < "$T/order")" = "H C A " ] && ok "once it has run, the same job queued again is a new ticket (H C A)" || bad "age reuse after a grant: $(tr '\n' ' ' < "$T/order")"
# 25. no expiry by default for lane tickets
s=$(date +%s); sh "$BH" gpu sleep 3 & sleep 0.3
( unset BOX_HOLD_WAIT; BOX_HOLD_WHO="cl-forever x" sh "$BH" gpu true ); rc1=$?; d=$(( $(date +%s) - s ))
[ "$rc1" = 0 ] && [ "$d" -ge 2 ] && ok "a lane ticket with no BOX_HOLD_WAIT waits it out (rc 0 after ${d}s)" || bad "no-expiry default: rc $rc1 after ${d}s"
wait
grep -qF '[ -n "$w" ] || { { [ "$pfx" = 1- ] || [ "$pfx" = 0-r- ]; } && w=0 || w=3600; }' "$BH" && ok "defaults: lane + release 0 (no expiry), CI 3600" || bad "wait defaults not as specified"

# 26. v5.1: an older copy's ticket gets its age record seeded, and its re-submission keeps its place
rm -f "$T/locks/box-last"; : > "$T/order"; rm -rf "${T:?}/locks/box-ages"; : > "$T/locks/box-focus"; : > "$T/locks/box-running"; : > "$T/locks/box-focus-done"   # each ordering arm: no ages, no focus (v5/v6)
jo=$(job OLD 0.2)
sh "$BH" gpu sh -c "$(job H 2)" & sleep 0.3
sleep 30 & old=$!; st=$(sed 's/.*) //' "/proc/$old/stat" | cut -d' ' -f20)
of="$T/locks/queue/1-$(printf '%012d' 2).$old.$st"
printf '%s %s %s\n' "cl-oldjob" "$(date -u +%FT%TZ)" "sh -c $jo" > "$of"          # what an older copy wrote: no marker
BOX_HOLD_WHO="cl-other B" sh "$BH" gpu sh -c "$(job B 0.2)" & sleep 0.4                 # B queues after it (and seeds)
okey=$(printf '%s\0%s' "cl-oldjob" "sh -c $jo" | sha256sum | cut -c1-32)   # the key v5 computes for this job
v_seeded=$(cat "$T/locks/box-ages/$okey" 2> /dev/null)
n_seeded=$(ls "$T/locks/box-ages" 2> /dev/null | wc -l)
sh "$BH" seed; n_again=$(ls "$T/locks/box-ages" | wc -l)
kill "$old" 2> /dev/null; wait "$old" 2> /dev/null; rm -f "$of"                        # its wait ran out: the waiter is gone
BOX_HOLD_WHO="cl-oldjob" sh "$BH" gpu sh -c "$jo" & sleep 0.3                           # the lane re-submits the same job
wait
[ "$v_seeded" = 2 ] && [ "$n_again" = "$n_seeded" ] && ok "an older copy's ticket gets ITS age record (its key = 2, seeded at a submission; seeding again changes nothing)" \
  || bad "seeding: record for the old ticket = [$v_seeded] (want 2); $n_seeded record(s), then $n_again"
[ "$(tr '\n' ' ' < "$T/order")" = "H OLD B " ] && ok "its re-submission keeps its original place (H OLD B, not H B OLD)" || bad "seeded age: $(tr '\n' ' ' < "$T/order")"

[ "$fail" = 0 ] && echo "box_hold_fifo: PASS" || echo "box_hold_fifo: FAIL"
exit "$fail"
