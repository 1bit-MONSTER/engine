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
# hrx_test_compile_test.sh -- arms for scripts/hrx-test-compile.sh (engine#2068).
#
# The gate's whole value is that it fails on the source that did not compile (#262) and does not report a pass when
# it never built the file. So the arms are: a tree that builds, a tree whose test source has the #262-shaped error,
# and a tree with no HRX targets at all -- the last one must exit 77, because "not built" and "built fine" are the
# two states this gate exists to keep apart.
#
# usage: hrx_test_compile_test.sh <path-to-hrx-test-compile.sh>
set -uo pipefail

S=${1:?usage: hrx_test_compile_test.sh <path-to-hrx-test-compile.sh>}
[ -x "$S" ] || { echo "not executable: $S"; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "skip: no cmake (exit 77)"; exit 77; }
work=$(mktemp -d "${TMPDIR:-/tmp}/hrxtc.XXXXXX")   # scratch, removed below
trap 'rm -rf "$work"' EXIT
fail=0
ok()  { echo "ok   $1"; }
bad() { echo "FAIL $1"; fail=1; }

# A stub "fork": the smallest tree whose CMakeLists declares the HRX test target the gate looks for, so the gate's
# own logic (configure, ask for the target, build it) is exercised without the real 654 KB source or the HRX toolchain.
mktree() {  # mktree <dir> <source-body> [declare-hrx]
    local d=$1 body=$2 hrx=${3:-yes}
    mkdir -p "$d/tests"
    cat > "$d/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(stub C CXX)
option(LLAMA_BUILD_TESTS "tests" ON)
option(GGML_HRX "hrx" ON)
if (LLAMA_BUILD_TESTS AND GGML_HRX AND ${hrx/yes/TRUE})
    add_subdirectory(tests)
endif()
EOF
    cat > "$d/tests/CMakeLists.txt" <<'EOF'
add_executable(test-hrx-backend hrx-backend-test.cpp)
EOF
    printf '%s\n' "$body" > "$d/tests/hrx-backend-test.cpp"
}

# 1. Compiles cleanly -> 0.
mktree "$work/good" 'int main(void) { return 0; }'
"$S" "$work/good" --build-dir "$work/b-good" >/dev/null 2>&1
[ $? = 0 ] && ok "a tree whose HRX test compiles exits 0" || bad "a tree whose HRX test compiles exits 0"

# 2. The #262 shape: an error in the test source -> non-zero, and the error text is surfaced.
mktree "$work/bad" 'int main(void) { const int x = "not an int"; return x; }'
out=$("$S" "$work/bad" --build-dir "$work/b-bad" 2>&1); rc=$?
[ "$rc" = 1 ] && ok "a tree with a broken HRX test exits 1" || bad "a tree with a broken HRX test exits 1 (got $rc)"
[[ $out == *test-hrx-backend* ]] && ok "and names the target that failed" || bad "and names the target that failed"
[[ $out == *error:* ]] && ok "and surfaces the compiler's error line" || bad "and surfaces the compiler's error line"

# 3. No HRX targets at all -> 77, never 0. This is the arm that keeps the gate from being vacuous.
mktree "$work/nohrx" 'int main(void) { return 0; }' no
out=$("$S" "$work/nohrx" --build-dir "$work/b-nohrx" 2>&1); rc=$?
[ "$rc" = 77 ] && ok "a tree without HRX targets exits 77 (skip, not pass)" || bad "a tree without HRX targets exits 77 (got $rc)"
[[ $out == *"SKIP, not PASS"* ]] && ok "and says so in those words" || bad "and says so in those words"

# 4. Usage contract.
"$S" >/dev/null 2>&1; [ $? = 2 ] && ok "a missing tree exits 2" || bad "a missing tree exits 2"
"$S" "$work/good" --nope >/dev/null 2>&1; [ $? = 2 ] && ok "an unknown argument exits 2" || bad "an unknown argument exits 2"
"$S" "$work/absent" >/dev/null 2>&1; [ $? = 2 ] && ok "a non-existent tree exits 2" || bad "a non-existent tree exits 2"

[ $fail = 0 ] && echo "all hrx-test-compile arms passed"
exit $fail
