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
# tests/box_hold_split_test.sh -- box-hold v7 SPLIT DEVICES (owner 2026-11-28), stub jobs on a private BOX_HOLD_DIR and a
# private switch file (BOX_HOLD_SPLIT_FILE), no device:
#   1. switch absent: today's rule - a gpu ticket holds BOTH locks; a gpu and an npu job of two lanes never overlap
#   2. switch on: a gpu job and an npu job of two lanes run at the same time; the gpu ticket does not hold the NPU lock
#   3. switch on: one lane run per device - the gpu focus lane runs back to back (A1 A2 before B1) while an npu lane
#      runs beside it; two gpu jobs never overlap
#   4. a 'both' ticket starts only when both devices are free, and blocks new starts on both while it waits
#   5. the switch is read at every poll: a gpu waiter queued while it was off takes only the GPU once it is on
#   6. `box-hold focus` prints both devices' lane runs when split
#   7. with the switch off, `box-hold focus` counts the pre-v7.1 waiters
#   8. v7.1 (2026-11-28 09:20Z): with the switch on but an older copy's waiter live, the split is NOT active (every ticket
#      takes both locks: one scheduler, the one older waiters follow); it starts by itself once the older waiter is gone
set -uo pipefail
BH=${1:?usage: box_hold_split_test.sh <box-hold>}
T=$(mktemp -d) || exit 1
trap 'kill $(jobs -p) 2> /dev/null; rm -rf "${T:?}" 2> /dev/null' EXIT
unset GITHUB_ACTIONS BOX_HOLD_WAIT
export BOX_HOLD_DIR=$T/locks BOX_HOLD_POLL=0.2 BOX_HOLD_FOCUS_GRACE=1 BOX_HOLD_SPLIT_FILE=$T/split
fail=0; ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; fail=1; }
# a job: "S tag" at start, "E tag" at end; with a 3rd arg, also probes whether the OTHER device's lock is held (by us)
job() { printf 'echo "S %s" >> %q/log; %s sleep %s; echo "E %s" >> %q/log\n' "$1" "$T" "${3:+if flock -n $T/locks/box-$3.lock true; then echo \"free $3 $1\" >> $T/probe; else echo \"held $3 $1\" >> $T/probe; fi;}" "$2" "$1" "$T"; }
L() { BOX_HOLD_WHO="$1" sh "$BH" "$2" sh -c "$(job "$3" "$4" "${5:-}")" 2>> "$T/err"; }   # <label> <dev> <tag> <secs> [probe dev]
reset() { : > "$T/log"; : > "$T/probe"; mkdir -p "$T/locks"; for f in box-focus box-running box-focus-done box-last box-focus-gpu box-focus-npu box-running-gpu box-running-npu box-focus-done-gpu box-focus-done-npu box-last-gpu box-last-npu; do : > "$T/locks/$f"; done; rm -rf "${T:?}/locks/box-ages" 2> /dev/null; }
seq_() { awk '{ printf "%s%s ", $1, $2 }' "$T/log"; }
overlap() {   # <a> <b>: did a and b run at the same time?
    awk -v a="$1" -v b="$2" '$1 == "S" { on[$2] = 1; if (on[a] && on[b]) x = 1 } $1 == "E" { on[$2] = 0 } END { exit !x }' "$T/log"; }

# 1. switch absent
reset; rm -f "$T/split"
L cl-a gpu G 1 npu & sleep 0.3; L cl-b npu N 0.3 &
wait
! overlap G N && ok "switch absent: the gpu and npu jobs of two lanes ran one after the other ($(seq_))" || bad "overlap without the switch: $(seq_)"
grep -q "held npu G" "$T/probe" && ok "  ... and the gpu ticket held the NPU lock too (the 2026-11-23 rule)" || bad "gpu ticket did not hold npu: $(cat "$T/probe")"

# 2. switch on
reset; touch "$T/split"
L cl-a gpu G 1.2 npu & sleep 0.3; L cl-b npu N 0.3 gpu &
wait
overlap G N && ok "split: the gpu and npu jobs of two lanes ran at the same time ($(seq_))" || bad "no overlap with the switch: $(seq_)"
grep -q "free npu G" "$T/probe" && grep -q "held gpu N" "$T/probe" && ok "  ... the gpu ticket did not hold the NPU lock; the npu job saw the GPU held" || bad "locks: $(cat "$T/probe")"

# 3. per-device lane runs
reset; touch "$T/split"
L cl-a gpu A1 2 & sleep 0.2; L cl-b gpu B1 0.2 & sleep 0.2; L cl-a gpu A2 0.4 & sleep 0.1; L cl-c npu C1 0.8 &   # A1 2 s: C1 (queued at 0.5 s) starts inside it even on the box
wait
g=$(awk '$1 == "S" && $2 ~ /^[AB]/ { printf "%s ", $2 }' "$T/log")
[ "$g" = "A1 A2 B1 " ] && ok "split: the gpu focus lane ran back to back: $g" || bad "gpu focus: $g ($(seq_))"
overlap A1 C1 && ok "  ... while the npu lane ran beside it" || bad "npu did not run beside: $(seq_)"
! overlap A1 B1 && ! overlap A2 B1 && ! overlap A1 A2 && ok "  ... and no two gpu jobs overlapped" || bad "gpu overlap: $(seq_)"

# 4. a 'both' ticket: the npu focus lane keeps chaining tickets (N1 N2 N3, within its grace); without the rule X would
#    wait for that run to end. Once X heads the GPU, the NPU takes no new start: X runs after N1, before N2.
reset; touch "$T/split"
L cl-g gpu G1 0.4 & sleep 0.1; L cl-n npu N1 1.6 & sleep 0.1
L cl-x both X 0.3 & sleep 0.1                       # waits: both devices busy
L cl-n npu N2 0.3 & sleep 0.1; L cl-n npu N3 0.3 & sleep 0.1   # the npu focus lane chains (no cap for it)
L cl-g2 gpu G2 0.2 &
wait
pos() { awk -v e="$1" -v t="$2" '$1 == e && $2 == t { print NR; exit }' "$T/log"; }
xs=$(pos S X) xe=$(pos E X)
[ -n "$xs" ] && [ "$xs" -gt "$(pos E G1)" ] && [ "$xs" -gt "$(pos E N1)" ] && ok "the both ticket started once both devices were free ($(seq_))" || bad "both start: $(seq_)"
[ -n "$xs" ] && [ "$(pos S N2)" -gt "$xe" ] && [ "$(pos S N3)" -gt "$xe" ] && [ "$(pos S G2)" -gt "$xe" ] \
  && ok "  ... and blocked new starts on both meanwhile: the npu focus lane's N2/N3 and G2 came after it" || bad "both blocking: $(seq_)"

# 5. switch read at every poll
reset; rm -f "$T/split"
L cl-h gpu H 1 & sleep 0.3
L cl-a gpu W 0.8 npu & sleep 0.2                    # queued while off
touch "$T/split"; wait
grep -q "free npu W" "$T/probe" && ok "a waiter queued while the switch was off took only the GPU once it was on" || bad "switch per poll: $(cat "$T/probe")"

# 6. focus prints both
reset; touch "$T/split"
L cl-a gpu A 1.5 & sleep 0.2; L cl-a gpu A2 0.1 & L cl-c npu C 1.5 & sleep 0.2; L cl-d npu D 0.1 & sleep 0.4
out=$(sh "$BH" focus)
echo "$out" | grep -q "^split devices: ON" && echo "$out" | grep -A1 "^== gpu" | grep -q "^focus: cl-a" && echo "$out" | grep -A1 "^== npu" | grep -q "^focus: cl-c" \
  && ok "box-hold focus prints both lane runs (gpu: cl-a, npu: cl-c)" || bad "focus print: $out"
wait

# 7. switch off: focus counts the waiters a split would skip (no v7 marker), so the trial knows when to switch on
reset; rm -f "$T/split"; mkdir -p "$T/locks/queue"
sleep 30 & p1=$!; s1=$(sed 's/.*) //' "/proc/$p1/stat" | cut -d' ' -f20)
printf 'cl-old %s x\nbox-hold-v3\nbox-hold-v6\n' "$(date -u +%FT%TZ)" > "$T/locks/queue/1-$(printf '%012d' 7).$p1.$s1"
out=$(sh "$BH" focus)
echo "$out" | grep -q "^pre-v7.1 waiters (the split starts only when this is 0): 1$" && ok "switch off: box-hold focus counts the pre-v7.1 waiters (1)" || bad "pre-v7.1 count: $out"
kill "$p1" 2> /dev/null; wait "$p1" 2> /dev/null

# 8. the split waits for older waiters: a REAL older waiter (a copy of this box-hold whose tickets lack the v7.1 marker),
#    no stub ticket (a stub's start time read under load 40 on the box got it swept as dead: flaky)
reset; touch "$T/split"
sed 's/box-hold-v7\\nbox-hold-v7.1\\n/box-hold-v7\\n/' "$BH" > "$T/bh-old"
grep -q 'box-hold-v7.1' "$T/bh-old" && ! grep -q 'box-hold-v7\\nbox-hold-v7.1' "$T/bh-old" || bad "could not derive the older copy"
BOX_HOLD_WHO=cl-h sh "$BH" gpu sh -c "$(job H 1.5)" & sleep 0.3                     # a holder, so the others queue
BOX_HOLD_WHO=cl-a sh "$BH" gpu sh -c "$(job G 0.3 npu)" & pg=$!; sleep 0.3           # cl-a: the next lane run
BOX_HOLD_WHO=cl-old sh "$T/bh-old" gpu sh -c "$(job O 0.3)" & sleep 0.3              # the older waiter, behind G
out=$(sh "$BH" focus)
echo "$out" | grep -q "^split devices: REQUESTED .* 1 older-copy waiter" && ok "switch on + an older waiter: focus says the split is requested, not active" || bad "requested print: $out"
wait "$pg"
grep -q "held npu G" "$T/probe" && ok "  ... and a gpu ticket takes both locks meanwhile (one scheduler)" || bad "split active beside an older waiter: $(cat "$T/probe") [order: $(seq_)]"
wait; : > "$T/probe"
L cl-a gpu G2 0.3 npu
grep -q "free npu G2" "$T/probe" && ok "  ... once it is gone the split starts by itself (gpu ticket: GPU lock only)" || bad "split did not start: $(cat "$T/probe")"

[ "$fail" = 0 ] && echo "box_hold_split: PASS" || echo "box_hold_split: FAIL"
exit "$fail"
