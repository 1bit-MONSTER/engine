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
# fork_ctest_known_fail.sh -- run the fork's own ctest suite against the known-failure baseline.
#
# engine#2761: 15 of the fork's 112 tests fail at the fork the tree pins (52dda9bc), 13 of them
# serially, and nothing in CI ran the suite -- so the fleet was green while it was red. Turning the
# suite on naively fails the job on arrival, so the failures we have are written down in
# tools/ci/ctest_known_fail.txt and a NEW one fails this row.
#
# It is OPT-IN (engine#2848 review). A row registered in CMakeLists runs inside every `ctest` on
# every box, and this one needs a fork build tree, a submodule at the pin and then 112 tests: on a
# lane box whose $HOME/.cache/hrx-build/bench exists, a plain `ctest` either went red on a stale
# submodule (the normal state of a worktree that has not run `git submodule update`) or quietly
# started a nested `ctest -j 4` over the fork's suite, which is CPU work that belongs under
# ~/lb/thermal-run.sh rather than inside someone's test run. Set ONEBIT_FORK_CTEST=1 to run it.
#
# Two things this row refuses to do:
#
#   * It does not run when the ratchet tool is absent. The tool lands separately (engine#2825); until
#     it is on the branch this row SKIPS with 77 rather than failing for a missing file. A row that
#     red-lights every PR because its own dependency has not merged yet is worse than no row.
#
#   * It does not trust the build tree it finds. "$HRX_BUILD/hrx/llama" is reused across runs, so a
#     configure done against an older llama.cpp submodule keeps its include paths -- which is exactly
#     how the first attempt died on `ggml-backend-impl.h: file not found` while that header existed at
#     the pinned sha. The pinned submodule is verified before the suite is believed.
#
# Exit: 0 no new failure, 1 a failure the baseline does not name, 2 cannot tell, 77 not applicable.
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HRX_BUILD="${HRX_BUILD:-$HOME/.cache/hrx-build/bench}"
FORK="$HRX_BUILD/hrx/llama"

# Not applicable beats failed: say which, on stdout, so the skip is legible in the ctest log.
skip() { echo "fork_ctest_known_fail: SKIP -- $*"; exit 77; }

[ "${ONEBIT_FORK_CTEST:-0}" = 1 ] || skip "opt-in: set ONEBIT_FORK_CTEST=1 to run the fork's suite (a fork build tree plus 112 tests)"

[ -f "$ROOT/tools/ci/ctest_known_fail.py" ] || skip "tools/ci/ctest_known_fail.py is not on this tree (engine#2825 lands it)"
[ -f "$ROOT/tools/ci/ctest_known_fail.txt" ] || skip "tools/ci/ctest_known_fail.txt is not on this tree"
[ -d "$FORK" ] || skip "no fork build tree at $FORK"

cd "$FORK"

# The submodule this tree was configured against, and the submodule that is actually checked out.
WANT=$(cd "$ROOT" && git ls-tree HEAD third_party/llama.cpp 2>/dev/null | awk '{print $3}' || true)
[ -n "$WANT" ] || skip "no git tree at $ROOT, so the pinned submodule cannot be read (an exported source tree)"
GOT=$(cd "$ROOT/third_party/llama.cpp" 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo none)
if [ "$GOT" = none ]; then
    echo "fork_ctest_known_fail: cannot tell: third_party/llama.cpp is not checked out" >&2
    exit 2
fi
if [ "$WANT" != "$GOT" ]; then
    echo "fork_ctest_known_fail: cannot tell: the llama.cpp submodule is at $GOT, the tree pins $WANT" >&2
    echo "  a build tree configured against a different submodule keeps stale include paths; reconfigure" >&2
    exit 2
fi

# tee, because the ratchet reads a log and ctest does not write one by itself. `|| true` is
# load-bearing: under `set -e` a non-zero ctest ends the script HERE, before the ratchet runs, and
# every known failure would then fail the row -- the one thing the baseline exists to prevent.
ctest -j 4 --timeout 600 -E '^(hrx-ops-test|test-hrx-ops|test-hrx-ops-loom)$' 2>&1 | tee ctest.log || true

exec python3 "$ROOT/tools/ci/ctest_known_fail.py" --log ctest.log --root "$ROOT"
