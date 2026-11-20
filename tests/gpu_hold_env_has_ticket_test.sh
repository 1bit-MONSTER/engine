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
# engine#2917: `env_has_ticket()` answers 0 / 1 / 2 -- "the environ names this ticket", "it does not", "it could not be
# read at all" -- and the three must stay distinguishable, because the sweep has to COUNT an unreadable process (an
# escapee could be hiding there, engine#2744's review) while a plain non-match must not be counted, and neither may abort
# the sweep.
#
# The function is extracted from the installed script and sourced, so this tests the real code rather than a copy of it:
# a copy is what let the weaker `any()` guard in the fa-split check survive its own test (engine#2925).
#
# Arms:
#   1. a readable environ that carries the line -> 0;
#   2. a readable environ without it -> 1 (and specifically not 0, the failure that hides an escapee);
#   3. a DIRECTORY as the path -> 2: `<` succeeds on a directory and the read is what fails with EISDIR, which is why
#      the probe is a one-byte read and not an open;
#   4. a real EACCES target -> 2, when this user cannot read one (root in CI cannot, and then arm 3 carries the point);
#   5. the binding arm: the script's redirect is inside the group whose stderr is discarded, because the shell prints its
#      own "Permission denied" for the `<` before `tr` runs and `2> /dev/null` on `tr` does not reach it.
set -u
HOLD=${1:-$(dirname "$0")/../scripts/gpu-hold}
[ -f "$HOLD" ] || { echo "  SKIP no gpu-hold at $HOLD"; exit 77; }

fails=0
note() { printf '  %s\n' "$*"; }

tmp=$(mktemp -d "${TMPDIR:-/tmp}/ghehv.XXXXXX") || exit 77
trap 'rm -rf "$tmp"' EXIT

# source the real function out of the script
sed -n '/^env_has_ticket()/,/^}/p' "$HOLD" > "$tmp/fn.sh"
if ! grep -q "^env_has_ticket()" "$tmp/fn.sh"; then
    note "FAIL env_has_ticket() is not in $HOLD -- the name moved, so this test is looking at nothing"
    exit 1
fi
# shellcheck disable=SC1090
. "$tmp/fn.sh"

printf 'A=1\0GPU_HOLD_TICKET=want pid 1\0B=2\0' > "$tmp/carries"
printf 'A=1\0B=2\0' > "$tmp/plain"
mkdir -p "$tmp/adir"

env_has_ticket "$tmp/carries" "GPU_HOLD_TICKET=want pid 1"; rc=$?
[ "$rc" = 0 ] && note "ok   a readable environ carrying the line answers 0" || { note "FAIL carrying line answered $rc (want 0)"; fails=$((fails + 1)); }

env_has_ticket "$tmp/plain" "GPU_HOLD_TICKET=want pid 1"; rc=$?
[ "$rc" = 1 ] && note "ok   a readable environ without it answers 1, not 0" || { note "FAIL non-match answered $rc (want 1)"; fails=$((fails + 1)); }

env_has_ticket "$tmp/adir" "GPU_HOLD_TICKET=want pid 1"; rc=$?
[ "$rc" = 2 ] && note "ok   a directory answers 2 -- unreadable, and the sweep must count it" || { note "FAIL directory answered $rc (want 2)"; fails=$((fails + 1)); }

if [ -r /proc/1/environ ]; then
    note "skip /proc/1/environ is readable here, so the EACCES arm has nothing to measure (the EISDIR arm above still ran)"
else
    env_has_ticket /proc/1/environ "GPU_HOLD_TICKET=want pid 1"; rc=$?
    [ "$rc" = 2 ] && note "ok   a real EACCES target answers 2 as well" || { note "FAIL EACCES target answered $rc (want 2)"; fails=$((fails + 1)); }
fi

if grep -q 'line=$( { tr' "$HOLD"; then note "ok   the redirect is inside the group whose stderr is discarded"
else note "FAIL the redirect is not grouped, so 'Permission denied' prints at the reader and the failure is not distinguishable"; fails=$((fails + 1)); fi

if [ "$fails" -ne 0 ]; then note "$fails failure(s)"; exit 1; fi
exit 0
