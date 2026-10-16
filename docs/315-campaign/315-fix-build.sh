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
# engine#315: build the FIX cell. ~/wt/llama-315fix already carries the same three diagnostics as the
# control cell ~/wt/315instr3-build plus the coherence fix, so the two cells differ by exactly the fix.
# Assertions before and after the build, so a silent build of the wrong source cannot be mistaken for a
# measurement (the cpe.cpp patch-path mistake is the precedent).
set -euo pipefail
PIN=e44c9d01a4d5110cecaf45472a717f46cd1abbf9
SRC=$HOME/wt/llama-315fix
B=$HOME/wt/315fix-build
HRX=$HOME/wt/pin-fork-hrx/third_party/hrx-system
LOG=$HOME/wt/315fix-build.log
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "preflight"
[ -e "$SRC/.git" ] || { echo "ABORT: $SRC is not a worktree"; exit 1; }
[ "$(git -C "$SRC" rev-parse HEAD)" = "$PIN" ] || { echo "ABORT: $SRC not at the pin"; exit 1; }
[ -d "$HRX" ] || { echo "ABORT: hrx-system missing"; exit 1; }
G=$SRC/ggml/src/ggml-hrx/ggml-hrx.cpp
grep -q 'memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT;' "$G" || { echo "ABORT: the fix is absent from the source"; exit 1; }
if grep -A1 'HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;' "$G" | grep -q 'if (direct_host_binding)'; then
  echo "ABORT: coherence is still gated on direct_host_binding - the fix is not in effect"; exit 1
fi
say "source OK: coherence unconditional, trace marker=$(grep -c 'hrx-buf' "$G")"
git -C "$SRC" diff > "$HOME/wt/315-fix-trace.patch"
say "candidate patch sha256 $(sha256sum "$HOME/wt/315-fix-trace.patch" | cut -d' ' -f1)"
say "control cell for comparison: $(~/wt/315instr3-build/bin/llama-server --version 2>&1 | head -1)"

say "configure"
rm -rf "$B"
cmake -S "$SRC" -B "$B" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DGGML_HIP=OFF -DGGML_HRX=ON -DGGML_VULKAN=OFF \
  -DHRX_SOURCE_DIR="$HRX" \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DBUILD_TESTING=OFF \
  > "$LOG" 2>&1 || { echo "ABORT: cmake failed, see $LOG"; tail -20 "$LOG"; exit 1; }

say "build (-j12; the peer is building with ~4)"
cmake --build "$B" -j 12 >> "$LOG" 2>&1 || { echo "ABORT: build failed, see $LOG"; tail -30 "$LOG"; exit 1; }

say "binary identity"; "$B/bin/llama-server" --version
LIB=$(ls "$B"/bin/libggml-hrx.so* 2>/dev/null | head -1)
[ -n "$LIB" ] || { echo "ABORT: libggml-hrx.so not found"; exit 1; }
for M in hrx-wb-fp hrx-dl-fp hrx-buf; do
  N=$(grep -c "$M" "$LIB" || true); say "  marker $M=$N"; [ "$N" -gt 0 ] || { echo "ABORT: $M absent from the binary"; exit 1; }
done
say "FIXBUILD_DONE bin_dir=$B/bin"
