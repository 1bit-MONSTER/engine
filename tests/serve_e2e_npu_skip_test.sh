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
# tests/serve_e2e_npu_skip_test.sh — the skip contract of `serve_e2e_npu`, device-free (engine#3023/#3030).
#
# That row exists so the NPU serve route is exercised somewhere. Two things have to stay true for it to mean
# anything, and neither is visible from a CI run that has no pack:
#
#   1. with no pack it answers 77 and says why, so a machine without one reports a skip instead of a red row
#      (a red row is turned off within a day) and instead of an absent row (which is how engine#3009 -- npu
#      serve announcing the port it was asked for -- reached main behind a green CI);
#   2. with a pack it does NOT answer 77. Without this arm the test above would also pass if the row skipped
#      unconditionally, and `tools/ci/ctest_skips.txt` names the row, so a permanently-skipping serve_e2e_npu
#      would be silent everywhere. That is the vacuity this file exists to prevent.
#
# No pack, no NPU, no device and no built binary are needed: every arm stops before `1bit serve` is launched, or
# launches `/bin/true` where a launch is the point.
#
# usage: bash tests/serve_e2e_npu_skip_test.sh    (exit 0 = every arm passes, 1 = a failure)
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
e2e=$root/tests/serve_e2e.sh
[ -f "$e2e" ] || { echo "serve_e2e_npu_skip_test: no $e2e" >&2; exit 2; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/nopack" "$tmp/withnpu/npu"

fails=0
pass() { echo "ok   $*"; }
fail() { echo "FAIL $*"; fails=$((fails + 1)); }

# arm <name> <want-rc> <glob the output must contain> <model arg>
arm() {
    local name=$1 want=$2 glob=$3 model=$4 out rc
    out=$(timeout 30 bash "$e2e" /bin/true "$model" npu 2>&1)
    rc=$?
    if [ "$rc" != "$want" ]; then fail "$name: rc=$rc want=$want -- $out"
    elif [[ $out != *$glob* ]]; then fail "$name: said '$out', wanted '$glob'"
    else pass "$name: rc=$rc"
    fi
}

# The box decides what a skip may SAY (engine#3049). A `/dev/accel` guard, when it lands, answers 77 before the pack
# check can -- so on a machine without an NPU the pack sentence is unreachable, and asserting it made three arms go red
# on the one change this file exists to survive. The probe is the same file such a guard reads, so the arm follows the
# box rather than a wording.
dev=0; [ -e /dev/accel/accel0 ] && dev=1
packmsg="ONEBIT_SERVE_TEST_NPU_PACK"
[ "$dev" = 1 ] || packmsg=""
arm "no pack configured -> skip" 77 "$packmsg" ""
# `main` answers the same message for an empty pack and for a path that is not a directory (one condition, one
# sentence), so this arm pins the status and not the wording.
arm "pack is not a directory -> skip" 77 "$packmsg" "$tmp/nope"

# The anti-vacuity arm. `/bin/true` ignores its arguments, so nothing binds a port and the script ends either by
# noticing the process is gone (rc 1) or on the `timeout` above (rc 124); both mean the pack guard let it through to
# the launch, which is all this asserts. 77 would mean the row cannot run even when a pack is configured.
#
# A skip FOR THE DEVICE is allowed and must stay allowed: `serve_e2e.sh` may grow a "/dev/accel present" guard (it was
# proposed on engine#3028's review), and on a machine without an NPU that guard must answer 77 rather than red -- a CI
# runner has no `/dev/accel` at all. So the assertion is "never the PACK reason", not "never 77": that stays true today
# (no device guard: it proceeds) and after such a guard lands (it skips naming the device). Which also means the
# vacuity guarantee this arm gives is about the pack guard only, and a device-less runner cannot give the full one.
out=$(timeout 5 bash "$e2e" /bin/true "$tmp/withnpu" npu 2>&1)
rc=$?
if [ "$rc" = 77 ] && [ "$dev" = 1 ]; then
    fail "a configured pack was skipped on a box that HAS /dev/accel/accel0 -- with the device present the only excuse is the pack: $out"
elif [ "$rc" = 77 ] && [[ $out == *ONEBIT_SERVE_TEST_NPU_PACK* ]]; then
    fail "a configured pack was skipped for the PACK reason -- the pack guard is broken: $out"
elif [ "$rc" = 77 ]; then
    pass "a configured pack skipped for the device (no /dev/accel here): $out"
else
    pass "a configured pack is not skipped: rc=$rc (reached the launch)"
fi

# The relaxation that lets the row carry an empty pack must not leak to the devices that require a model.
out=$(timeout 30 bash "$e2e" /bin/true "" cpu 2>&1)
rc=$?
if [ "$rc" = 2 ]; then pass "cpu with no model still exits 2 (usage)"
else fail "cpu with no model: rc=$rc want=2 -- $out"
fi

if [ "$fails" -eq 0 ]; then echo "serve_e2e_npu_skip_test: pass"; exit 0; fi
echo "serve_e2e_npu_skip_test: $fails failure(s)"
exit 1
