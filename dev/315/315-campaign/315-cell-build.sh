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
# engine#315: build a measurement cell from a given source worktree. Asserts the intended source change
# is present BEFORE configuring and the diagnostic markers are in the binary AFTER, so a silent build of
# the wrong tree cannot be mistaken for a measurement.
#
#   SRC=<worktree> B=<build dir> [LOG=<log>] [EXPECT=<grep pattern in ggml-hrx.cpp>] 315-cell-build.sh
set -euo pipefail
SRC=${SRC:?source worktree}; B=${B:?build dir}; LOG=${LOG:-$B.log}
PIN=${PIN:-e44c9d01a4d5110cecaf45472a717f46cd1abbf9}
HRX=${HRX:-$HOME/wt/pin-fork-hrx/third_party/hrx-system}
EXPECT=${EXPECT:-}
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "preflight"
[ -e "$SRC/.git" ] || { echo "ABORT: $SRC is not a worktree"; exit 1; }
[ "$(git -C "$SRC" rev-parse HEAD)" = "$PIN" ] || { echo "ABORT: $SRC not at the pin"; exit 1; }
[ -d "$HRX" ] || { echo "ABORT: hrx-system missing"; exit 1; }
# Search the whole backend tree: a change may live in any translation unit (this assertion originally
# looked only at ggml-hrx.cpp and rejected a probe that lives in runtime/graph-executor.cpp).
if [ -n "$EXPECT" ]; then
  grep -rq "$EXPECT" "$SRC/ggml/src/ggml-hrx" || { echo "ABORT: expected source change '$EXPECT' is absent - wrong tree"; exit 1; }
  say "source assertion OK: '$EXPECT' present in $(grep -rl "$EXPECT" "$SRC/ggml/src/ggml-hrx" | head -1)"
fi
git -C "$SRC" diff > "$HOME/wt/$(basename "$B").patch" 2>/dev/null || true
say "worktree diff: $(git -C "$SRC" diff --shortstat | tr -d '\n')"

say "configure"
rm -rf "$B"
cmake -S "$SRC" -B "$B" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DGGML_HIP=OFF -DGGML_HRX=ON -DGGML_VULKAN=OFF \
  -DHRX_SOURCE_DIR="$HRX" \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DBUILD_TESTING=OFF \
  > "$LOG" 2>&1 || { echo "ABORT: cmake failed, see $LOG"; tail -15 "$LOG"; exit 1; }

say "build (-j12)"
cmake --build "$B" -j 12 >> "$LOG" 2>&1 || { echo "ABORT: build failed, see $LOG"; tail -25 "$LOG"; exit 1; }

say "binary identity"; "$B/bin/llama-server" --version
LIB=$(ls "$B"/bin/libggml-hrx.so* 2>/dev/null | head -1)
[ -n "$LIB" ] || { echo "ABORT: libggml-hrx.so not found"; exit 1; }
# MARKERS lets a cell assert its own marker: a probe that never compiled in would otherwise pass every
# assertion here, which is exactly what happened once.
for M in hrx-wb-fp hrx-dl-fp hrx-buf ${MARKERS:-}; do
  N=$(grep -c "$M" "$LIB" || true); say "  marker $M=$N"; [ "$N" -gt 0 ] || { echo "ABORT: $M absent"; exit 1; }
done
say "CELLBUILD_DONE bin_dir=$B/bin"
