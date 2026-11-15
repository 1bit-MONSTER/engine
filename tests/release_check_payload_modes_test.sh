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
# The second half of the same defect: the `payload-backend` row owns ABSENCE (the mode function owns the mode), so it needs
# the same two things - the decision is a function, and the row actually calls it. A row that exists to catch a missing
# backend and has no arm of its own is the defect this PR is about, one level up.
FN_BK=$(awk '/^ *payload_backend_missing\(\) \{/{f=1} f{print} f&&/^ *\}$/{exit}' "$RC")
[ -n "$FN_BK" ] || { echo "FAIL no payload_backend_missing() in $RC - the row cannot require a backend"; exit 1; }
grep -q 'payload_backend_missing "\$PDIR"' "$RC" || { echo "FAIL payload_backend_missing() is defined but never called on the payload"; exit 1; }

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

# The fourth binary: `serve` spawns the HRX backend for every onebit/llamacpp-hrx model, and a payload whose
# `llama-server` was 0644 answered the first chat request with "Failed to execute: ... Permission denied" (measured
# 2026-11-15) while this row - which required the other three - said nothing about it.
mkdir -p "$ROOT/pbackend/1bit/hrx/llama/bin" "$ROOT/pbackend/bin"
printf 'x' > "$ROOT/pbackend/1bit/1bit"; printf 'x' > "$ROOT/pbackend/bin/lemond"
chmod 0755 "$ROOT/pbackend/1bit/1bit" "$ROOT/pbackend/bin/lemond"
printf 'x' > "$ROOT/pbackend/1bit/hrx/llama/bin/llama-server"; chmod 0644 "$ROOT/pbackend/1bit/hrx/llama/bin/llama-server"
got=$(bash -c "$FN"$'\npayload_modes_noexec "$1"' _ "$ROOT/pbackend")
case "$got" in
    *hrx/llama/bin/llama-server*) echo "ok    a 0644 HRX backend is named even when the three named binaries are 0755" ;;
    *) echo "FAIL  a 0644 HRX backend went unreported: '${got}'"; fail=1 ;;
esac

# payload-backend's own decision, three ways: a complete backend reports nothing; an absent one names both pieces
# (the executable and the compute library, which fail at exec and at dlopen respectively); and a backend that is
# present but not executable is NOT this function's claim - the mode function above owns that.
mkdir -p "$ROOT/pbk/1bit/hrx/llama/bin"
: > "$ROOT/pbk/1bit/hrx/llama/bin/llama-server"; : > "$ROOT/pbk/1bit/hrx/llama/bin/libggml-hrx.so.0"
got=$(bash -c "$FN_BK"$'\npayload_backend_missing "$1"' _ "$ROOT/pbk")
[ -z "$got" ] && echo "ok    a complete HRX backend reports nothing missing" || { echo "FAIL  a complete backend was reported: '${got}'"; fail=1; }

mkdir -p "$ROOT/pbn/1bit"
got=$(bash -c "$FN_BK"$'\npayload_backend_missing "$1"' _ "$ROOT/pbn")
case "$got" in
    *hrx/llama/bin/llama-server*libggml-hrx*) echo "ok    a payload with no backend names the executable and the library" ;;
    *) echo "FAIL  an absent backend was under-reported: '${got}'"; fail=1 ;;
esac

mkdir -p "$ROOT/pbl/1bit/hrx/llama/bin"; : > "$ROOT/pbl/1bit/hrx/llama/bin/libggml-hrx.so.0"
got=$(bash -c "$FN_BK"$'\npayload_backend_missing "$1"' _ "$ROOT/pbl")
case "$got" in
    *llama-server*) case "$got" in
        *libggml-hrx*) echo "FAIL  a present library was reported missing: '${got}'"; fail=1 ;;
        *) echo "ok    an executable-only gap names the executable and not the library it does have" ;;
    esac ;;
    *) echo "FAIL  the missing executable was not named: '${got}'"; fail=1 ;;
esac

mktree pnone none
got=$(bash -c "$FN"$'\npayload_modes_noexec "$1"' _ "$ROOT/pnone")
[ -z "$got" ] && echo "ok    a payload with no binaries reports nothing (absence is another row's claim)" || { echo "FAIL  a binary-less payload was reported: '${got}'"; fail=1; }

echo "release_check_payload_modes_test: $fail failure(s)"
exit $((fail > 0))
