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
# tests/box_hold_release_test.sh -- box-hold's RELEASE class (owner 2026-11-28, v2026.49), stub jobs on a private
# BOX_HOLD_DIR and a private allow-list (BOX_HOLD_RELEASE_CONF), no device:
#   1. a release ticket runs after CI and before EVERY lane ticket, even the focus lane's queued tickets (H C R1 R2 A2 B1)
#   2. a running lane ticket is never cut: the release ticket waits for it to end
#   3. honoured: a cl-release-rel* label; a listed lane whose label carries one of ITS issues (#3580), also as lane-suffix,
#      and a dashed lane (cl-inkling-npu) by its own issue
#   4. refused (queued as a lane ticket, reported): no issue number, another lane's issue, an unlisted lane, no label, and
#      cl-inkling-npu with cl-inkling's issue (exact lane match, review cl-loomfun)
#   5. release tickets never expire by default; focus prints them
#   6. split devices on: an npu release ticket goes before the npu lane's queued ticket, while the GPU keeps its own run
#   7. a focus-lane waiter of a copy OLDER than the release class (it reads 0-r- tickets as CI) does not jump the release
#      tickets either: A1 R1 R2 A2, not A1 R1 A2 R2 (2026-11-28 18:09:31Z, cl-220)
set -uo pipefail
BH=${1:?usage: box_hold_release_test.sh <box-hold>}
T=$(mktemp -d) || exit 1
trap 'kill $(jobs -p) 2> /dev/null; rm -rf "${T:?}" 2> /dev/null' EXIT
unset GITHUB_ACTIONS BOX_HOLD_WAIT BOX_HOLD_PRIORITY
export BOX_HOLD_DIR=$T/locks BOX_HOLD_POLL=0.2 BOX_HOLD_FOCUS_GRACE=1 BOX_HOLD_SPLIT_FILE=$T/no-split BOX_HOLD_RELEASE_CONF=$T/rel.conf
printf '# test list\ncl-hrxtest 3580 3554\ncl-loom 3521\ncl-inkling 3539\ncl-inkling-npu 3335\n' > "$T/rel.conf"
printf 'cl-inkling-npu\n' > "$T/lanes.conf"; export BOX_HOLD_LANES=$T/lanes.conf   # a dashed lane name, as on the box
fail=0; ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; fail=1; }
job() { printf 'if [ -e %q/busy ]; then echo overlap >> %q/overlap; fi; touch %q/busy; echo %s >> %q/order; sleep %s; rm -f %q/busy\n' "$T" "$T" "$T" "$1" "$T" "$2" "$T"; }
cp "$(command -v bash)" "$T/Runner.Worker"
ci() { GITHUB_ACTIONS=true BOX_HOLD_PRIORITY=ci "$T/Runner.Worker" -c 'sh "$0" "$@"; exit $?' "$BH" "$@"; }
L() { BOX_HOLD_WHO="$1" sh "$BH" gpu sh -c "$(job "$2" "${3:-0.2}")" 2>> "$T/err"; }
REL() { BOX_HOLD_PRIORITY=release BOX_HOLD_WHO="$1" sh "$BH" gpu sh -c "$(job "$2" "${3:-0.2}")" 2>> "$T/err"; }
reset() { : > "$T/order"; : > "$T/err"; rm -f "$T/overlap"; mkdir -p "$T/locks"; for f in box-focus box-running box-focus-done box-last; do : > "$T/locks/$f"; done; rm -rf "${T:?}/locks/box-ages" 2> /dev/null; }
order() { tr '\n' ' ' < "$T/order"; }
klass() { ls "$T/locks/queue" | awk -v p="$1" 'index($0, p) == 1' | wc -l; }   # queued tickets with a name prefix

# 1 + 2. order
reset
L cl-a A1 1.2 & sleep 0.3                                  # cl-a has the focus; A1 runs
L cl-a A2 & sleep 0.15; L cl-b B1 & sleep 0.15             # its next ticket and another lane wait
REL "cl-hrxtest #3580 phi2" R1 & sleep 0.15; REL "cl-release-rel-check" R2 & sleep 0.15
ci gpu sh -c "$(job C 0.2)" & sleep 0.15
wait
[ "$(order)" = "A1 C R1 R2 A2 B1 " ] && ok "release after CI and before every lane ticket, the focus lane's too: $(order)" || bad "order: $(order)"
[ ! -e "$T/overlap" ] && ok "  ... the running lane ticket was not cut (never two jobs at once)" || bad "overlap"

# 3. honoured
reset; : > "$T/locks/box-hold.log"
for who in "cl-release-rel-rc2" "cl-hrxtest #3554 openelm" "cl-hrxtest-phi2 #3580" "cl-loom #3521 prefill" "cl-inkling-npu #3335 npu" "cl-inkling-npu-b #3335"; do
    REL "$who" X 0.1; done
grep -q "ignored" "$T/err" && bad "an allowed label was refused: $(cat "$T/err")" || ok "honoured: cl-release-rel*, a listed lane with its issue (also lane-suffix labels)"

# 4. refused
reset
L cl-h H 2 & sleep 0.3                                     # a holder, so the refused tickets queue
for who in "cl-hrxtest phi2-no-issue" "cl-hrxtest #3521 not-mine" "cl-other #3580" "cl-inkling-npu #3539 not-its-lane"; do REL "$who" X 0.1 & sleep 0.2; done
sleep 0.3; r=$(klass 0-r-); l=$(klass 1-)
[ "$r" = 0 ] && [ "$l" -ge 3 ] && ok "refused: no issue, another lane's issue, an unlisted lane, cl-inkling-npu with cl-inkling's issue -> lanes ($l), none in the release class" || bad "refused: $r release, $l lane queued"
[ "$(grep -c "BOX_HOLD_PRIORITY=release ignored" "$T/err")" -ge 4 ] && grep -q "release class: only labels" "$T/locks/box-hold.log" && ok "  ... each reported on stderr and in box-hold.log" || bad "not reported: $(cat "$T/err")"
wait
( unset BOX_HOLD_WHO; BOX_HOLD_PRIORITY=release sh "$BH" gpu true 2> "$T/e5" ); grep -q "ignored" "$T/e5" && ok "no label: refused" || bad "no label: $(cat "$T/e5")"

# 5. no expiry + focus
reset
L cl-h H 3 & sleep 0.3
s=$(date +%s); REL "cl-hrxtest #3580 wait" W 0.1 & pw=$!; sleep 0.5
out=$(sh "$BH" focus); echo "$out" | grep -q "^release tickets waiting: 1 " && ok "box-hold focus counts the waiting release ticket" || bad "focus: $out"
wait "$pw"; rc=$?; d=$(( $(date +%s) - s ))
[ "$rc" = 0 ] && [ "$d" -ge 2 ] && ok "a release ticket with no BOX_HOLD_WAIT waited it out (rc 0 after ${d}s)" || bad "expiry: rc $rc ${d}s"
wait

# 6. split
reset; touch "$T/split"; export BOX_HOLD_SPLIT_FILE=$T/split
S() { BOX_HOLD_WHO="$1" sh "$BH" "$2" sh -c "$(job "$3" "$4")" 2>> "$T/err"; }
S cl-g gpu G1 1.5 & sleep 0.2; S cl-n npu N1 0.8 & sleep 0.2
S cl-n npu N2 0.2 & sleep 0.15
BOX_HOLD_PRIORITY=release BOX_HOLD_WHO="cl-loom #3521 npu" sh "$BH" npu sh -c "$(job RN 0.2)" 2>> "$T/err" & sleep 0.15
wait
o=$(awk '$1 ~ /^(N|RN)/ { printf "%s ", $1 }' "$T/order")
[ "$o" = "N1 RN N2 " ] && grep -q G1 "$T/order" && ok "split: the npu release ticket went before the npu lane's N2 ($o), the GPU ran its own lane" || bad "split release: $(order)"
export BOX_HOLD_SPLIT_FILE=$T/no-split

# 7. an older copy's focus-lane waiter
reset
# the older copy: this box-hold without the release kind (as v7.1: a 0-r- ticket is "ci" to it, no release-first rule)
sed -e 's|kind = (f ~ /^0-r-/) ? "rel" : |kind = |' "$BH" > "$T/bh-old"
grep -q '"rel"' "$T/bh-old" && grep -q 'kind = (f ~ /^0-r-/)' "$T/bh-old" && bad "could not derive the older copy"
L cl-a A1 1 & sleep 0.3                                                        # cl-a has the focus; A1 runs
BOX_HOLD_WHO=cl-a sh "$T/bh-old" gpu sh -c "$(job A2 0.2)" 2>> "$T/err" & sleep 0.2    # its next ticket, by the OLDER copy
REL "cl-hrxtest #3580 r1" R1 0.6 & sleep 0.15; REL "cl-loom #3521 r2" R2 0.2 & sleep 0.15
wait
[ "$(order)" = "A1 R1 R2 A2 " ] && ok "an older copy's focus-lane waiter waits for every release ticket: $(order)" || bad "older copy jumped the release queue: $(order)"

[ "$fail" = 0 ] && echo "box_hold_release: PASS" || echo "box_hold_release: FAIL"
exit "$fail"
