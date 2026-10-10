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
# engine#315 row probe: which token is the NaN, and where in the MoE chain does it enter?
#
# Section 47's invariant is that 15 of 16 faulting arms carry exactly one n_embd row of NaN (2048 floats)
# while the batch size varies four-fold. A count cannot say more than that, so the fingerprint was extended
# (in ~/wt/llama-315copy) to print `ne0=` (the innermost extent), `first=` (the index of the first NaN) and
# `row=` (first/ne0). This run reads those over the named per-token MoE tensors.
#
# What the answer decides:
#   the NaN row is the *last* row, in every arm  -> the token whose routing is computed last, i.e. a read
#     that is not ordered after the routing computation: the fix is to order it
#   the same row index throughout the chain      -> a per-token state, so the carrier is the token's routing
#     weight or one selected expert's activation for that token
#   differing row indices across the chain       -> the corruption is not per-token at all and the row
#     invariant is a coincidence of the sampler's single-row read
#
# Builds into a NEW directory (315row-build): overwriting a binary an in-use session is executing is the
# mistake that has already cost one session, and 315fp-build is in use by the running chain probe. Source
# edits are safe because both running and queued sessions run prebuilt binaries.
#
# Diagnostic, not acceptance: no registered criterion changes and no arm from it is eligible.
#
# Usage: 315-rowprobe.sh [arms] [tag]
set -uo pipefail
ARMS=${1:-4}; TAG=${2:-315row}
SRC=$HOME/wt/llama-315copy
B=$HOME/wt/315row-build
HRX=$HOME/wt/pin-fork-hrx/third_party/hrx-system
LOG=$HOME/wt/315row-build.log

exec 9>"$HOME/.cache/lax-decode/box.lock"
if flock -w "${ROW_LOCK_WAIT:-5400}" 9; then
  echo "row: box lock held $(date -Is)"
else
  echo "row: lock unobtainable after ${ROW_LOCK_WAIT:-5400}s - aborting rather than running concurrently"
  exit 3
fi

grep -q "first_nan" "$SRC/ggml/src/ggml-hrx/ggml-hrx.cpp" || { echo "row: ABORT - instrument not in the source"; exit 1; }
[ -d "$HRX" ] || { echo "row: ABORT - hrx-system missing"; exit 1; }

echo "row: configuring $B $(date -Is)"
rm -rf "$B"
cmake -S "$SRC" -B "$B" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DGGML_HIP=OFF -DGGML_HRX=ON -DGGML_VULKAN=OFF \
  -DHRX_SOURCE_DIR="$HRX" \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DBUILD_TESTING=OFF \
  > "$LOG" 2>&1 || { echo "row: cmake failed"; tail -20 "$LOG"; exit 1; }
echo "row: building -j10 $(date -Is)"
cmake --build "$B" -j 6 >> "$LOG" 2>&1 || { echo "row: build failed"; tail -30 "$LOG"; exit 1; }
"$B/bin/llama-server" --version

LIB=$(ls "$B"/bin/libggml-hrx.so* 2>/dev/null | head -1)
grep -qa "first=%lld" "$LIB" || { echo "row: ABORT - the new field is not in $LIB"; exit 1; }
echo "row: index field present in $(basename "$LIB")"

NAMES="ffn_moe_probs-41,ffn_moe_weights-41,ffn_moe_gate-41,ffn_moe_up-41,ffn_moe_weighted-41,ffn_moe_down-41,ffn_moe_out-41,ffn_swiglu-41,ffn_shexp-41,ffn_out-41,l_out-41,norm,result_norm,result_output"
echo "row: running $ARMS arms with the index fingerprint $(date -Is)"
CBIN="$B/bin" FBIN="$B/bin" NOLOCK=1 \
REGIME_LO=${REGIME_LO:-80} REGIME_HI=${REGIME_HI:-92} \
  bash "$HOME/wt/315-attrib.sh" "$ARMS" "$TAG" 999 \
  "GGML_HRX_SENTINEL=1 GGML_HRX_TENSOR_FP=$NAMES GGML_HRX_TENSOR_FP_NANONLY=1"

echo "=== row readout $(date -Is) ==="
for d in "$HOME"/wt/rt-det/"$TAG"-*/; do
  [ -d "$d" ] || continue
  echo "--- $(basename "$d" | sed "s/$TAG-//;s/-GLM.*//") nan_guard=$(grep -c 'HRX returned NaN logits' "$d/server.log") ---"
  grep -m6 "hrx-tensor-fp" "$d/server.log" 2>/dev/null | cut -c1-165
done
echo "Read row= against the earlier arms' nan=2048: one row of 2048 floats is the invariant to explain."
