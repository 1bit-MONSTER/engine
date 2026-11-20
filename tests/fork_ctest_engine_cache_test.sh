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
#
# fork_ctest_engine_cache_test.sh -- engine#3000: refuse the ctest directory that CAUSES the nesting, and keep the stamp
# the ratchet reads written before the ratchet runs.
#
# The depth marker (engine#3000, merged in #3004) refuses a nested run once it starts. These two arms are about the two
# things it cannot see:
#   1. the ctest directory was configured from an ENGINE source, so its ctest registers this same row -- the
#      misconfiguration, refused before any ctest starts;
#   2. the provenance stamp is written by a step AFTER the one that reads it, so the first run after a pin move answers
#      "cannot tell" (rc 2) and #3004's `|| exit 1` turns that into a red step blaming a stamp instead of a test.
#
# Device-free and ctest-free: every arm stops at a guard, or reads the workflow text.
#
# Usage: fork_ctest_engine_cache_test.sh [repo-root]
set -u
root=${1:-.}
script="$root/tools/ci/fork_ctest_known_fail.sh"
workflow="$root/.github/workflows/hrx-build.yml"
[ -f "$script" ] || { echo "not a file: $script"; exit 2; }
[ -f "$workflow" ] || { echo "not a file: $workflow"; exit 2; }

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT INT TERM HUP
fails=0
chk() { if [ "$2" = "$3" ]; then printf 'ok   %s\n' "$1"; else printf 'FAIL %s: got %s, want %s\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi; }

# 1. a ctest directory configured from the ENGINE is refused by name, before any ctest runs
mkdir -p "$scratch/enginehx/hrx/llama"
printf 'CMAKE_HOME_DIRECTORY:INTERNAL=/somewhere/engine\n' > "$scratch/enginehx/hrx/llama/CMakeCache.txt"
out=$(ONEBIT_FORK_CTEST=1 HRX_BUILD="$scratch/enginehx" sh "$script" 2>&1)
chk "a ctest dir configured from the engine refuses (rc 2)" "$?" 2
case "$out" in
    *"registers this same row"*) chk "and it says what running it would do" 1 1 ;;
    *)                           chk "and it says what running it would do" 1 0 ;;
esac

# 2. a ctest directory configured from the FORK passes that guard (it then stops at the submodule guard, whose message is
#    different -- which is how we know the cache guard let it through)
mkdir -p "$scratch/forkhx/hrx/llama"
printf 'CMAKE_HOME_DIRECTORY:INTERNAL=/somewhere/engine/third_party/llama.cpp\n' > "$scratch/forkhx/hrx/llama/CMakeCache.txt"
out=$(ONEBIT_FORK_CTEST=1 HRX_BUILD="$scratch/forkhx" sh "$script" 2>&1)
chk "a ctest dir configured from the fork passes the cache guard" "$?" 2
case "$out" in
    *"registers this same row"*) chk "and the reason is a different one" 1 0 ;;
    *)                           chk "and the reason is a different one" 1 1 ;;
esac

# 3. no cache at all is not a refusal: the tree may be a bare fixture, and the submodule guard still decides
mkdir -p "$scratch/nocache/hrx/llama"
out=$(ONEBIT_FORK_CTEST=1 HRX_BUILD="$scratch/nocache" sh "$script" 2>&1)
case "$out" in
    *"registers this same row"*) chk "no CMakeCache is not treated as an engine tree" 1 0 ;;
    *)                           chk "no CMakeCache is not treated as an engine tree" 1 1 ;;
esac

# 4. the depth marker from #3004 still refuses (this test does not replace that one; it checks it did not regress)
ONEBIT_FORK_CTEST=1 ONEBIT_FORK_CTEST_DEPTH=1 HRX_BUILD=/nonexistent sh "$script" > /dev/null 2>&1
chk "a nested call still refuses (rc 2)" "$?" 2

# 5. THE ORDER: the stamp the ratchet reads must be written before the step that reads it. Static, because the arm for
#    the behaviour needs a fork build tree and a pin bump.
stamp_line=$(grep -n "name: Stamp the tree with its build provenance" "$workflow" | cut -d: -f1)
ctest_line=$(grep -n "name: Run the fork's own ctest rows" "$workflow" | cut -d: -f1)
chk "the workflow names both steps" "$([ -n "$stamp_line" ] && [ -n "$ctest_line" ] && echo yes)" yes
chk "the stamp step comes BEFORE the fork-ctest step" "$([ "${stamp_line:-0}" -lt "${ctest_line:-0}" ] && echo yes || echo no)" yes

if [ "$fails" = 0 ]; then echo "fork_ctest_engine_cache: all arms ok"; else echo "fork_ctest_engine_cache: $fails failure(s)"; fi
exit $([ "$fails" = 0 ] && echo 0 || echo 1)
