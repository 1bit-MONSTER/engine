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
# payload-modes must REQUIRE the binaries it names to be executable (engine#2422).
#
# The row printed the modes and judged only group/other-writability, so a payload whose `1bit/1bit` was 0644 read
# `PASS payload-modes (modes: 644 1bit/1bit …)` while the row that would start the engine read `SKIP hrx-placement`
# (`[ -x ]` found no runnable binary). Green over a payload that cannot run.
#
# Three arms, because the fix has two ways to be wrong and only one of them is loud:
#   * 0644 binaries must FAIL and NAME them -- otherwise the row still passes what it exists to catch;
#   * 0755 binaries must PASS with the modes still printed -- otherwise it fails healthy payloads;
#   * a payload with no binaries must still PASS -- absence is hrx-device's claim, not this row's, and turning it into a
#     failure here would red every payload that ships the engine somewhere this row does not look.
#
# The payloads are real directories (empty files are enough: the floor wants >100 entries, and nothing here reads a
# byte of them), built under ~/.cache because /tmp is tmpfs on this box and `tools/ci/tmp_path_check.py` flags new /tmp
# sites by design.
#
# usage: release_check_payload_modes_test.sh <path-to-release-check.sh>
#
# The FIRST version of this fixture ran `release-check.sh --check` three times, once per payload: 34 s here and **392 s in CI**
# (18 s per run locally became 130 s in a container), for three `[ -x ]` lines. So it now does what the neighbouring
# `release_check_hrx_init_reason_test.sh` does - extract the function and test it - plus one grep that pins the CALL SITE, because a
# correct function nothing calls is the defect this whole row is about (engine#2416/#2422 shape, and the engine#2238 lesson: a test
# should pin where the call lives, not only that the callee works).
set -u

RC=${1:?usage: release_check_payload_modes_test.sh <path-to-release-check.sh>}
[ -r "$RC" ] || { echo "not readable: $RC"; exit 1; }

FN=$(awk '/^ *payload_modes_noexec\(\) \{/{f=1} f{print} f&&/^ *\}$/{exit}' "$RC")
[ -n "$FN" ] || { echo "FAIL no payload_modes_noexec() in $RC - the row cannot require an executable engine (engine#2422)"; exit 1; }
case "$RC" in
    *"this line is never matched"*) : ;;
esac
grep -q 'payload_modes_noexec "\$PDIR"' "$RC" || { echo "FAIL payload_modes_noexec() is defined but never called on the payload - a guard nothing calls"; exit 1; }

ROOT=$(mktemp -d "${TMPDIR:-$HOME/.cache}/release-check-pm.XXXXXX") || exit 1
trap 'rm -rf "$ROOT"' EXIT
fail=0

mktree() {   # <name> <mode|none>
    mkdir -p "$ROOT/$1/1bit" "$ROOT/$1/bin"
    if [ "$2" != none ]; then
        printf 'x' > "$ROOT/$1/1bit/1bit"; printf 'x' > "$ROOT/$1/bin/lemond"
        chmod "$2" "$ROOT/$1/1bit/1bit" "$ROOT/$1/bin/lemond"
    fi
}

mktree p644 0644
got=$(bash -c "$FN"$'\npayload_modes_noexec "$1"' _ "$ROOT/p644")
case "$got" in
    *1bit/1bit*bin/lemond*) echo "ok    a 0644 engine and lemond are both named as not executable" ;;
    *) echo "FAIL  a 0644 payload was not reported: '${got}'"; fail=1 ;;
esac

mktree p755 0755
got=$(bash -c "$FN"$'\npayload_modes_noexec "$1"' _ "$ROOT/p755")
[ -z "$got" ] && echo "ok    a 0755 payload reports nothing" || { echo "FAIL  a healthy payload was reported: '${got}'"; fail=1; }

mktree pnone none
got=$(bash -c "$FN"$'\npayload_modes_noexec "$1"' _ "$ROOT/pnone")
[ -z "$got" ] && echo "ok    a payload with no binaries reports nothing (absence is another row's claim)" || { echo "FAIL  a binary-less payload was reported: '${got}'"; fail=1; }

echo "release_check_payload_modes_test: $fail failure(s)"
exit $((fail > 0))
