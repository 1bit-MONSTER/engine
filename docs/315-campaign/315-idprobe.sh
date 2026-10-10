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
# engine#315 id probe: are the route ids a clean permutation, and is the NaN in ffn_moe_weights an aliased read?
#
# Why: arm 315row-p1c (in regime, gtt_start=98G) left the fault on the ids. Its route-weight gather
# ffn_moe_weights = ggml_get_rows(ffn_moe_probs, ffn_moe_topk) was NaN for two whole tokens while ffn_moe_probs
# was not, and the ids were never instrumented. Two probe defects made that unanswerable: the name filter was
# std::strstr(list, name) (so "-41" also matched "-4", covering layers 4 and 41 only) and every tensor was read
# as f32, so an i32 ids tensor was fingerprinted as nonsense.
#
# This run builds the patched instrument (~/wt/llama-315copy, patch applied by tools/patch-id-instrument.py):
#   * anchored name matching against the comma-separated list;
#   * for i32 tensors a real ids dump: first 8 values, min/max, count outside [0, ne0) = oor, and dup_pairs
#     (repeated values within a row);
#   * for f32 tensors a second hash over the FINITE subset only, so two tensors sharing a NaN payload cannot
#     look identical when their finite parts differ.
#
# Readout, pre-committed:
#   oor > 0 or dup_pairs > 0 in a faulting arm   -> the id list is not a clean permutation: the argsort's rank
#       computation (evidence/315-id-producing-kernel.md) is the site, and the stale id is what the expert
#       tables consume
#   ids clean while the matmuls are NaN          -> the argsort candidate dies and the fault is downstream, in
#       the table builders or in the matmul's descriptor use
#   hash_finite equal across layers 4 and 41     -> an aliased read of the same buffer, not a coincidence
#
# Diagnostic, not acceptance: no registered criterion changes and no arm from it is eligible.
#
# Usage: 315-idprobe.sh [arms] [tag]
set -uo pipefail
ARMS=${1:-4}; TAG=${2:-315id}
SRC=$HOME/wt/llama-315copy
B=$HOME/wt/315id-build
HRX=$HOME/wt/pin-fork-hrx/third_party/hrx-system
LOG=$HOME/wt/315id-build.log

exec 9>"$HOME/.cache/lax-decode/box.lock"
if flock -w "${IDP_LOCK_WAIT:-10800}" 9; then
  echo "id: box lock held $(date -Is)"
else
  echo "id: lock unobtainable after ${IDP_LOCK_WAIT:-10800}s - aborting rather than running concurrently"
  exit 3
fi

grep -q "hrx_fp_name_listed" "$SRC/ggml/src/ggml-hrx/ggml-hrx.cpp" || { echo "id: ABORT - instrument not in the source"; exit 1; }
[ -d "$HRX" ] || { echo "id: ABORT - hrx-system missing"; exit 1; }

echo "id: configuring $B $(date -Is)"
rm -rf "$B"
cmake -S "$SRC" -B "$B" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DGGML_HIP=OFF -DGGML_HRX=ON -DGGML_VULKAN=OFF \
  -DHRX_SOURCE_DIR="$HRX" \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DBUILD_TESTING=OFF \
  > "$LOG" 2>&1 || { echo "id: cmake failed"; tail -20 "$LOG"; exit 1; }
echo "id: building -j 6 $(date -Is)"
cmake --build "$B" -j 6 >> "$LOG" 2>&1 || { echo "id: build failed"; tail -30 "$LOG"; exit 1; }
"$B/bin/llama-server" --version

LIB=$(ls "$B"/bin/libggml-hrx.so* 2>/dev/null | head -1)
for marker in hrx-ids-fp hrx-finite-fp; do
  grep -qa "$marker" "$LIB" || { echo "id: ABORT - $marker not in $LIB"; exit 1; }
done
echo "id: ids and finite-hash instruments present in $(basename "$LIB")"

# Note on volume and perturbation: NANONLY=1 keeps the f32 tensors quiet, while the i32 branch of the
# patched instrument prints unconditionally (it is asked for by name, and an i32 tensor has no NaN). So the
# added traffic per layer graph is two small ids reads plus, only when a NaN is present, the f32 lines. The
# name list is deliberately narrow (four per layer, not the whole MoE chain) so the extra device->host reads
# stay as small a perturbation as this instrument can be.

# The node names are suffixed with the layer index (observed: ffn_moe_weights-4, l_out-41), so an ANCHORED
# match needs every layer spelled out. 64 layers is above GLM-4.7-Flash's depth and costs nothing: a name that
# matches nothing is simply never printed.
NAMES="norm,result_norm,result_output"
for l in $(seq 0 63); do
  for n in ffn_moe_topk ffn_moe_argsort ffn_moe_weights ffn_moe_probs; do
    NAMES="$NAMES,$n-$l"
  done
done
echo "id: name list $(printf '%s' "$NAMES" | tr ',' '\n' | wc -l) entries"

echo "id: running $ARMS arms with the id instrument $(date -Is)"
CBIN="$B/bin" FBIN="$B/bin" NOLOCK=1 \
REGIME_LO=${REGIME_LO:-80} REGIME_HI=${REGIME_HI:-92} \
ARM_TIMEOUT=${ARM_TIMEOUT:-1800} \
  bash "$HOME/wt/315-attrib.sh" "$ARMS" "$TAG" 999 \
  "GGML_HRX_TENSOR_FP=$NAMES GGML_HRX_TENSOR_FP_NANONLY=1"

echo "=== id readout $(date -Is) ==="
CONSOLE=$HOME/wt/$TAG-console.log
grep -hE "^  pair" "$CONSOLE" || true
echo "clean: $(grep -c 'arm l: OK' "$CONSOLE" || true)  fault: $(grep -c 'arm l: \(FAULT\|INVALID\)' "$CONSOLE" || true)  oor: $(grep -c 'arm l: OOR' "$CONSOLE" || true)"
for d in "$HOME"/wt/rt-det/"$TAG"-*/; do
  [ -d "$d" ] || continue
  b=$(basename "$d" | sed "s/$TAG-//;s/-GLM.*//")
  ids=$(grep -c "hrx-ids-fp" "$d/server.log" 2>/dev/null || echo 0)
  bad=$(grep "hrx-ids-fp" "$d/server.log" 2>/dev/null | grep -cE "oor=[1-9]|dup_pairs=[1-9]" || true)
  echo "--- $b ids_lines=$ids bad_id_lines=$bad nan_guard=$(grep -c 'HRX returned NaN logits' "$d/server.log" 2>/dev/null || true)"
  # the faulting evidence, if there is any: the first bad id line, and the ids at the faulting call
  grep "hrx-ids-fp" "$d/server.log" 2>/dev/null | grep -E "oor=[1-9]|dup_pairs=[1-9]" | head -4 | cut -c1-200
  grep -m2 "hrx-ids-fp" "$d/server.log" 2>/dev/null | cut -c1-200
done
echo "Read oor/dup_pairs= against the prediction: a stale id in the top-k view comes from a rank collision."
