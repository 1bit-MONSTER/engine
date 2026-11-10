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
# The cmake/tests-*.cmake glob in CMakeLists.txt, run as it stands (engine#1979).
#
# The block between `# >>> registration glob` and `# <<< registration glob` is cut out of the real CMakeLists.txt and
# configured in a scratch project, so what is tested is the text main configures with, not a copy of it. Each case
# says which tests ctest must then list:
#   - a new file nothing names                                   -> its test is registered (the point of the glob),
#   - the same file named only in a COMMENT                      -> still registered. A substring search skipped it
#     here: the family's tests left the configuration and no gate noticed (review of engine#1983),
#   - a file CMakeLists.txt include()s by name, four spellings   -> registered once; a second include would stop the
#     configure with `add_test given test NAME ... which already exists`,
#   - tests-a.b.cmake beside an include of tests-axb.cmake       -> both registered: the dot is not a wildcard.
#
# usage: tests/registration_glob_test.sh <cmake> <path to CMakeLists.txt>   (configure only: no compiler, no device)
set -uo pipefail
CMAKE=${1:?usage: registration_glob_test.sh <cmake> <CMakeLists.txt>}
TOP=${2:?usage: registration_glob_test.sh <cmake> <CMakeLists.txt>}
CTEST="$(dirname "$CMAKE")/ctest"
[ -x "$CTEST" ] || CTEST=ctest
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
BLOCK=$(sed -n '/^# >>> registration glob$/,/^# <<< registration glob$/p' "$TOP")
if [ -z "$BLOCK" ] || ! printf '%s\n' "$BLOCK" | grep -q 'file(GLOB'; then
    echo "FAIL: no '# >>> registration glob' block with a file(GLOB in $TOP"; exit 1
fi
fails=0
# case <what> <expected test names, space separated, sorted> <CMakeLists prelude> <registration file>...
case_() {
    local what=$1 want=$2 prelude=$3; shift 3
    local d="$T/p$((++n))"; mkdir -p "$d/cmake"
    local f
    for f in "$@"; do
        printf 'add_test(NAME %s COMMAND true)\n' "$(printf '%s' "${f%.cmake}" | tr -c 'A-Za-z0-9_\n' '_')" > "$d/cmake/$f"
    done
    { printf 'cmake_minimum_required(VERSION 3.20)\nproject(regglob NONE)\nenable_testing()\n%s\n%s\n' "$prelude" "$BLOCK"
    } > "$d/CMakeLists.txt"
    local got
    if ! "$CMAKE" -S "$d" -B "$d/b" > "$d/log" 2>&1; then
        got="CONFIGURE FAILED: $(grep -m1 -A1 'CMake Error' "$d/log" | tr '\n' ' ')"
    else
        got=$(cd "$d/b" && "$CTEST" -N | sed -n 's/^ *Test *#[0-9]*: //p' | sort | tr '\n' ' ' | sed 's/ $//')
    fi
    if [ "$got" = "$want" ]; then echo "  ok    $what"; else echo "  FAIL  $what: got '$got', wanted '$want'"; fails=$((fails + 1)); fi
}
n=0
case_ "a new file nothing names is registered" "tests_one" "" tests-one.cmake
case_ "a file named only in a comment is still registered" "tests_one" \
      "# history: this used to be include(cmake/tests-one.cmake) right here" tests-one.cmake
case_ "a file named in a trailing comment is still registered" "tests_one" \
      "set(X 1) # was include(cmake/tests-one.cmake)" tests-one.cmake
case_ "a file included by name is registered once" "tests_one tests_two" \
      "include(cmake/tests-one.cmake)" tests-one.cmake tests-two.cmake
case_ "an indented include under if() counts as an include" "tests_one" \
      "$(printf 'if(TRUE)\n    include(cmake/tests-one.cmake)\nendif()')" tests-one.cmake
case_ 'a ${CMAKE_SOURCE_DIR}/ include counts as an include' "tests_one" \
      'include(${CMAKE_SOURCE_DIR}/cmake/tests-one.cmake)' tests-one.cmake
case_ "a quoted include counts as an include" "tests_one" \
      'include("cmake/tests-one.cmake")' tests-one.cmake
case_ "the dot in a file name is not a wildcard" "tests_a_b tests_axb" \
      "include(cmake/tests-axb.cmake)" tests-a.b.cmake tests-axb.cmake
case_ "a file the glob does not match is not registered" "tests_one" "" tests-one.cmake other.cmake
echo "registration_glob_test: $fails failure(s)"
[ "$fails" -eq 0 ]
