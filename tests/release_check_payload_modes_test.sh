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
set -u

RC=${1:?usage: release_check_payload_modes_test.sh <path-to-release-check.sh>}
[ -r "$RC" ] || { echo "not readable: $RC"; exit 1; }

ROOT=$(mktemp -d "${TMPDIR:-$HOME/.cache}/release-check-pm.XXXXXX") || exit 1
trap 'rm -rf "$ROOT"' EXIT
fail=0

mkpayload() {   # <dir> <mode or "none">
    local d=$ROOT/$1 mode=$2 i
    mkdir -p "$d/1bit" "$d/bin"
    for i in $(seq 1 120); do : > "$d/lib$i.so"; done
    find "$d" -type d -exec chmod 0755 {} +
    find "$d" -type f -exec chmod 0644 {} +
    if [ "$mode" != none ]; then
        printf 'x' > "$d/1bit/1bit"; printf 'x' > "$d/bin/lemond"
        chmod "$mode" "$d/1bit/1bit" "$d/bin/lemond"
    fi
}

run_row() {     # <dir> -> the payload-modes line
    PAYLOAD_DIR=$ROOT/$1 HW_SKIPPED=gpu,npu,npu-power,models,chat-hrx,chat-npu \
        timeout 300 bash "$RC" --check 2>&1 | grep -m1 'payload-modes' || true
}

mkpayload p644 0644
line=$(run_row p644)
case "$line" in
    FAIL*"payload-modes"*"not executable"*"1bit/1bit"*"bin/lemond"*)
        echo "ok    a 0644 engine and lemond FAIL, and both are named" ;;
    *)  echo "FAIL  a 0644 payload was not refused: ${line:-<no payload-modes line>}"; fail=1 ;;
esac

mkpayload p755 0755
line=$(run_row p755)
case "$line" in
    PASS*"payload-modes"*"modes:"*"755 1bit/1bit"*)
        echo "ok    a 0755 payload PASSes and still prints its modes" ;;
    *)  echo "FAIL  a healthy payload was not accepted with its modes: ${line:-<no payload-modes line>}"; fail=1 ;;
esac

mkpayload pnone none
line=$(run_row pnone)
case "$line" in
    PASS*"payload-modes"*)
        echo "ok    a payload with no binaries still PASSes (absence is another row's claim)" ;;
    *)  echo "FAIL  a binary-less payload changed verdict: ${line:-<no payload-modes line>}"; fail=1 ;;
esac

echo "release_check_payload_modes_test: $fail failure(s)"
exit $((fail > 0))
