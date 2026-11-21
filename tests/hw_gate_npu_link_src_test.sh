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
# tests/hw_gate_npu_link_src_test.sh — the NPU link gate points at a tree that has the targets it builds (engine#3079).
#
# The gate configured `third_party/llama.cpp` ("$src", the HRX UI path) and then built npu_rt_shim_test in it. That
# tree defines no such target, so the step could never pass, and it reported "the NPU runtime does not link the
# engine's NPU targets" -- a finding about the private runtime drawn from a configure that never read the engine.
# Measured 2026-11-21 on the leftover of hw-gate run 44843183313: `CMAKE_HOME_DIRECTORY:INTERNAL=.../third_party/
# llama.cpp`, `ONEBIT_NPU:UNINITIALIZED=ON`, and 0 occurrences of the target in build.ninja.
#
# Hermetic: text and one grep of the engine's own CMakeLists. No configure, no build, no device.
#
# usage: bash tests/hw_gate_npu_link_src_test.sh    (exit 0 = every arm passes, 1 = a failure)
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
gate=$root/scripts/hw-gate.sh
cmake_file=$root/CMakeLists.txt
[ -f "$gate" ] || { echo "hw_gate_npu_link_src_test: no $gate" >&2; exit 2; }

fails=0
pass() { echo "ok   $*"; }
fail() { echo "FAIL $*"; fails=$((fails + 1)); }

# The NPU link block, delimited by the note that announces it and the note that closes the pass branch.
block=$(awk '/note "private NPU runtime:/{f=1} f{print} f&&/NPU link gate: npu_rt_shim_test and npu_dmabuf_import_test link/{exit}' "$gate")
[ -n "$block" ] || { echo "hw_gate_npu_link_src_test: the NPU link block is not in $gate" >&2; exit 2; }

# 1. It configures the ENGINE tree. "$src" is third_party/llama.cpp in this script -- the bug.
if grep -q 'cmake -S "\$ROOT"' <<<"$block"; then pass "configures the engine root (-S \"\$ROOT\")"
else fail "does not configure the engine root; the NPU targets live there and nowhere else"; fi
if grep -q 'cmake -S "\$src"' <<<"$block"; then fail "still configures \$src (third_party/llama.cpp), which defines no npu_rt_shim_test"
else pass "does not configure \$src"; fi

# 2. Both targets are built, so an edit that drops one cannot pass silently.
for t in npu_rt_shim_test npu_dmabuf_import_test; do
    if grep -q -- "--target .*$t" <<<"$block"; then pass "builds $t"
    else fail "does not build $t"; fi
done

# 3. A target missing from the graph is "the check did not run", and that verdict comes BEFORE the build attempt.
didrun=$(grep -n 'the NPU link gate did not run' <<<"$block" | head -1 | cut -d: -f1)
buildline=$(grep -n 'cmake --build' <<<"$block" | head -1 | cut -d: -f1)
linkfail=$(grep -n "does not link the engine's NPU targets" <<<"$block" | head -1 | cut -d: -f1)
if [ -n "$didrun" ]; then pass "the cannot-tell verdict exists ('the NPU link gate did not run')"
else fail "no cannot-tell verdict: a tree without the targets still reads as a link failure"; fi
if [ -n "$didrun" ] && [ -n "$buildline" ] && [ "$didrun" -lt "$buildline" ]; then
    pass "cannot-tell is decided before the build is attempted"
else fail "cannot-tell does not precede the build attempt (didrun=${didrun:-none} build=${buildline:-none})"; fi
if [ -n "$linkfail" ] && [ -n "$buildline" ] && [ "$buildline" -lt "$linkfail" ]; then
    pass "a real link failure is only claimed after the build"
else fail "the link-failure message is not downstream of the build"; fi

# 3b. A BUILD DIR OF ITS OWN. This one is easy to lose in review and impossible to see in a fresh tree: the old gate's
#     cache in "${build}-npu" was generated with `-S third_party/llama.cpp`, and CMake refuses to re-home an existing
#     cache ("the current CMakeCache.txt is different than the one used to generate" -- a FATAL_ERROR), so pointing the
#     same directory at the engine would fail on every box that had already run the old gate. The fix is only a fix if
#     the configure gets a directory that has never seen the submodule.
if grep -q 'npu_build="\${build}-npu-engine"' <<<"$block"; then
    pass "configures into \"\${build}-npu-engine\", a directory the old -S \"\$src\" never poisoned"
else
    fail "the NPU configure does not use a fresh build directory -- an existing cache generated against third_party/llama.cpp cannot be re-homed (CMake FATAL_ERROR), so the gate would still fail on every box that ran the old one"
fi

# 4. The predicate is the ninja graph, not the log: the log carries the configure's own words, which is how the old
#    message came out of a configure that had not read the engine.
if grep -q "build.ninja" <<<"$block"; then pass "the cannot-tell predicate reads build.ninja"
else fail "the cannot-tell predicate does not read build.ninja"; fi

# 5. Anti-vacuity: the engine's CMakeLists really does define both targets. Without this the arms above would keep
#    passing after a rename, and the gate would cannot-tell on every run instead of building anything.
for t in npu_rt_shim_test npu_dmabuf_import_test; do
    if grep -q "add_executable($t" "$cmake_file"; then pass "CMakeLists defines $t"
    else fail "CMakeLists does not define $t -- the gate would report cannot-tell on every run"; fi
done

if [ "$fails" -eq 0 ]; then echo "hw_gate_npu_link_src_test: pass"; exit 0; fi
echo "hw_gate_npu_link_src_test: $fails failure(s)"
exit 1
