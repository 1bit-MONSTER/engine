#!/bin/sh
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
# fork_ctest_nest_test.sh -- the ratchet must refuse to run inside itself (engine#3000).
#
# On 2026-11-20 a caller exported ONEBIT_FORK_CTEST=1 and ran an engine ctest whose tree also contained this row. The
# script ctested that tree, the row ran again, and each entry added one sh, one ctest and one tee: ~36,600 processes,
# swap exhausted, load average 500, first OOM at 17:35:52Z, force-reboot. The guard is a depth marker; this test is the
# arm for it, and it needs no device, no build tree and no fork checkout -- the refusal happens before anything is read.
#
#   fork_ctest_nest_test.sh            # the guard refuses
#   fork_ctest_nest_test.sh --self-test  # the same, named as a self-test for tools/ci conventions
set -u
hold="$1"
[ -n "$hold" ] || hold="$(cd "$(dirname "$0")/.." && pwd)/tools/ci/fork_ctest_known_fail.sh"

fail=0
fail_msg() { echo "FAIL $*"; fail=1; }

# 1. already inside itself -> REFUSED, rc 2, and nothing else attempted
out=$(ONEBIT_FORK_CTEST=1 ONEBIT_FORK_CTEST_DEPTH=1 /bin/sh "$hold" 2>&1)
rc=$?
[ "$rc" = 2 ] || fail_msg "a nested run should refuse with rc 2, got $rc"
printf '%s' "$out" | grep -q "REFUSED: already running inside itself" \
    || fail_msg "the refusal does not say it is already running inside itself: $(printf '%s' "$out" | head -1)"
printf '%s' "$out" | grep -q "engine#3000" \
    || fail_msg "the refusal does not name the incident it exists for"

# 2. the marker is not a skip: rc 2 is "cannot tell", which a caller must not read as "not applicable" (77)
printf '%s' "$out" | grep -q "SKIP" && fail_msg "a nested run must not report SKIP"

# 3. a first entry is not refused: without the marker it gets past the guard (whatever it does next depends on the
#    box, so this only asserts the guard itself did not fire)
out2=$(ONEBIT_FORK_CTEST=0 /bin/sh "$hold" 2>&1)
rc2=$?
[ "$rc2" = 2 ] && printf '%s' "$out2" | grep -q "REFUSED" && fail_msg "a first entry was refused: $(printf '%s' "$out2" | head -1)"

if [ $fail = 0 ]; then
    echo "fork_ctest_nest_test: ok (a nested run refuses with rc 2, a first entry is not refused)"
fi
exit $fail
