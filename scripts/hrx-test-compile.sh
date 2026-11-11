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
# hrx-test-compile.sh -- compile the fork's HRX test targets, and nothing else.
#
# Why this exists (engine#2068): `tests/hrx-backend-test.cpp` in 1bit-MONSTER/llama.cpp stopped compiling at
# llama.cpp#262 and NOBODY SAW IT, because no automated job builds that file -- it is registered under
# `if (TARGET ggml-hrx)` in the fork's tests/CMakeLists.txt, so a build without the HRX backend skips it, and no
# build asks for tests. Three host rows then failed in a binary nothing ran. A compile break in a 654 KB test
# file is the one thing a *build* catches for free and a *run* never gets the chance to.
#
# This is a compile gate only: it never runs the tests and needs no device.
#
# usage: hrx-test-compile.sh <fork-checkout> [--build-dir DIR]
# exit: 0 all HRX test targets built; 1 a target failed to build; 2 usage; 77 the tree has no HRX backend
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "usage: hrx-test-compile.sh <fork-checkout> [--build-dir DIR]" >&2
    exit 2
fi
tree=$1
shift || true
build_dir=""
while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) build_dir=${2:?--build-dir needs a path}; shift 2 ;;
        *) echo "hrx-test-compile: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -d "$tree" ] || { echo "hrx-test-compile: not a directory: $tree" >&2; exit 2; }
[ -d "$tree/tests" ] || { echo "hrx-test-compile: no tests/ in $tree" >&2; exit 2; }
[ -n "$build_dir" ] || build_dir="$tree/build-hrx-test-compile"

# Configure with the HRX backend and the tests on: those two switches are what make the file part of the build.
# GGM L_HRX's own option name is the fork's; if it is renamed, this configure fails loudly rather than skipping.
if ! cmake -S "$tree" -B "$build_dir" -DLLAMA_BUILD_TESTS=ON -DGGML_HRX=ON -DCMAKE_BUILD_TYPE=Release >"$build_dir.configure.log" 2>&1; then
    echo "hrx-test-compile: configure FAILED (see $build_dir.configure.log)" >&2
    tail -5 "$build_dir.configure.log" >&2
    exit 1
fi

# Which of the HRX test targets this tree actually declares. Asking CMake rather than grepping the CMakeLists keeps
# the gate honest when the target list changes: a target that disappears is reported, not silently dropped.
have=$(cmake --build "$build_dir" --target help 2>/dev/null | sed -n 's/^\.\.\. \([A-Za-z0-9_.-]*\)$/\1/p' | grep -x 'test-hrx-backend\|test-hrx-rope-claim-params' | sort -u)

if [ -z "$have" ]; then
    # Not "passed": no HRX target exists in this tree, so this gate proved nothing about the file it exists for.
    echo "hrx-test-compile: no HRX test targets in $tree (GGML_HRX=ON did not produce them) -- SKIP, not PASS" >&2
    exit 77
fi

rc=0
for t in $have; do
    if cmake --build "$build_dir" --target "$t" >>"$build_dir.build.log" 2>&1; then
        echo "ok   built $t"
    else
        echo "FAIL build $t (see $build_dir.build.log)" >&2
        grep -E "error:" "$build_dir.build.log" | head -3 >&2
        rc=1
    fi
done
[ "$rc" = 0 ] && echo "ok   HRX test targets compile ($(echo $have | wc -w) target(s))"
exit "$rc"
