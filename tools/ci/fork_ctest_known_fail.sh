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

# THE SOURCE TREE IS NOT THE TREE THAT RUNS. The guard above compares ROOT's gitlink with ROOT's submodule, and
# then this script runs `ctest` in $FORK, which is a COPY under $HRX_BUILD. A source tree aligned to the pin
# therefore satisfies that guard while the build tree stays whatever it was configured from -- and its
# failures are then reported as this pin's. Measured on 2026-11-20: with ROOT's submodule at 8d26add2c and
# $HRX_BUILD a different-pin copy, this row answered "11 NEW failure(s)" and "11 now passing", none of which
# was about the pinned source. So ask the BUILD tree where it came from, using the stamp bench_stamp.py
# writes (engine#2882), and refuse to compare when it cannot say.
STAMP="$HRX_BUILD/bench-stamp.txt"
if [ ! -f "$STAMP" ]; then
    echo "fork_ctest_known_fail: cannot tell: $STAMP is absent, so this build tree does not record which source it was built from" >&2
    echo "  a tree configured against another llama.cpp keeps stale include paths, and its failures are not this pin's" >&2
    echo "  stamp it with: tools/ci/bench_stamp.py write --tree $HRX_BUILD --root $ROOT" >&2
    exit 2
fi
STAMPED=$(sed -n 's/^llama_cpp=//p' "$STAMP" 2> /dev/null | head -1)
# The comparison below treats the stamp's value as a PREFIX of the pin, because bench_stamp.py writes a `--short=12`
# revision while the pin is a full one. That makes the length load-bearing: `llama_cpp=8` is a prefix of a pin
# beginning `8d26add2c…` and satisfied the guard on 2026-11-20 -- the row answered "18 known failure(s), no new ones"
# about a build tree nobody could identify, which is the same green tick over nothing this guard was added to stop.
# A stamp that cannot name a revision is not provenance.
if [ "${#STAMPED}" -lt 7 ]; then
    echo "fork_ctest_known_fail: cannot tell: the build tree's stamp names llama_cpp='${STAMPED}', shorter than a git short sha (7); bench_stamp.py writes 12" >&2
    exit 2
fi
case "$WANT" in
    "${STAMPED:-none}"*) : ;;          # the stamp records a short sha; the pin is a full one
    *)
        echo "fork_ctest_known_fail: cannot tell: the build tree's stamp says llama_cpp=${STAMPED:-<missing>}, the tree pins $WANT" >&2
        echo "  the suite would be run against a different fork than the one this tree pins" >&2
        exit 2 ;;
esac


# tee, because the ratchet reads a log and ctest does not write one by itself. `|| true` is
# load-bearing: under `set -e` a non-zero ctest ends the script HERE, before the ratchet runs, and
# every known failure would then fail the row -- the one thing the baseline exists to prevent.
ctest -j 4 --timeout 600 -E '^(hrx-ops-test|test-hrx-ops|test-hrx-ops-loom)$' 2>&1 | tee ctest.log || true

exec python3 "$ROOT/tools/ci/ctest_known_fail.py" --log ctest.log --root "$ROOT"
