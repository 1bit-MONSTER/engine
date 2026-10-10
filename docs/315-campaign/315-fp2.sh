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
# engine#315 fp2: the ORDERED locator. Only worth running if 315gsync2 shows the fault survives a sync.
#
# Why: section 53 withdrew the by-name per-node attribution because the fingerprint reads host-visible tensors
# out of a host-visible non-coherent arena through ggml_backend_tensor_get, which for those is a plain memcpy
# with no synchronisation. The per-node lines were therefore not a snapshot of one instant - proven by
# contradiction inside the arms themselves (ffn_moe_out finite downstream of a 99.95 %-NaN ffn_moe_weighted).
#
# `tools/patch-fp-sync.py` inserts one `ggml_backend_synchronize(backend)` before the fingerprint's read loop,
# which orders those reads. The trade-off is explicit: a sync per graph is itself an ordering intervention, so
# this build can inhibit the fault it is meant to locate.
#
#   gsync2 says the fix cell is CLEAN      -> the ordering is the mechanism; do not run this, there is nothing
#                                             left to locate, and the remedy is scoped around the wait
#   gsync2 says the fault SURVIVES a sync  -> run this: the sync is not the carrier, and its per-node lines are
#                                             then valid evidence about where the corruption is
#
# Usage: 315-fp2.sh [arms] [tag]
set -uo pipefail
ARMS=${1:-4}; TAG=${2:-315fp2}
SRC=$HOME/wt/llama-315copy
B=$HOME/wt/315fp2-build
HRX=$HOME/wt/pin-fork-hrx/third_party/hrx-system
LOG=$HOME/wt/315fp2-build.log

exec 9>"$HOME/.cache/lax-decode/box.lock"
if flock -w "${FP2_LOCK_WAIT:-10800}" 9; then
  echo "fp2: box lock held $(date -Is)"
else
  echo "fp2: lock unobtainable after ${FP2_LOCK_WAIT:-10800}s - aborting rather than running concurrently"
  exit 3
fi

grep -q "ggml_backend_synchronize(backend);" "$SRC/ggml/src/ggml-hrx/ggml-hrx.cpp" \
  || { echo "fp2: ABORT - the ordering patch is not in the source (tools/patch-fp-sync.py)"; exit 1; }
grep -q "hrx_fp_name_listed" "$SRC/ggml/src/ggml-hrx/ggml-hrx.cpp" \
  || { echo "fp2: ABORT - the anchored-match patch is not in the source"; exit 1; }

# Pre-committed gate: refuse to spend the box if gsync2 already answered the question.
C=$HOME/wt/315gsync2-console.log
if [ -f "$C" ]; then
  fixf=$(grep -c 'arm l: \(FAULT\|INVALID\)' "$C" || true)
  fixok=$(grep -c 'arm l: OK' "$C" || true)
  echo "fp2: gsync2 so far - fix cell OK=$fixok fault=$fixf"
  if [ "${fixok:-0}" -ge 3 ] && [ "${fixf:-0}" -eq 0 ]; then
    echo "fp2: ABORT - gsync2's fix cell is clean; the ordering is the mechanism and this locator has no question left"
    exit 0
  fi
else
  echo "fp2: WARNING - no gsync2 console yet, so the gate cannot be checked; proceeding"
fi

echo "fp2: configuring $B $(date -Is)"
rm -rf "$B"
cmake -S "$SRC" -B "$B" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DGGML_HIP=OFF -DGGML_HRX=ON -DGGML_VULKAN=OFF \
  -DHRX_SOURCE_DIR="$HRX" \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DBUILD_TESTING=OFF \
  > "$LOG" 2>&1 || { echo "fp2: cmake failed"; tail -20 "$LOG"; exit 1; }
echo "fp2: building -j 6 $(date -Is)"
cmake --build "$B" -j 6 >> "$LOG" 2>&1 || { echo "fp2: build failed"; tail -30 "$LOG"; exit 1; }
"$B/bin/llama-server" --version

LIB=$(ls "$B"/bin/libggml-hrx.so* 2>/dev/null | head -1)
for marker in hrx-ids-fp hrx-finite-fp; do
  grep -qa "$marker" "$LIB" || { echo "fp2: ABORT - $marker not in $LIB"; exit 1; }
done
echo "fp2: ordered fingerprint + ids + finite hash present in $(basename "$LIB")"

NAMES="norm,result_norm,result_output"
for l in $(seq 0 63); do
  for n in ffn_moe_topk ffn_moe_argsort ffn_moe_probs ffn_moe_weights ffn_moe_gate ffn_moe_up ffn_moe_down ffn_moe_weighted ffn_moe_out ffn_swiglu ffn_shexp ffn_out l_out; do
    NAMES="$NAMES,$n-$l"
  done
done
echo "fp2: name list $(printf '%s' "$NAMES" | tr ',' '\n' | wc -l) entries"

# REGIME_LO=80: measured, not chosen for the strictest reading. A neighbour costs ~17 GiB of GTT (a 17 GiB model
# plus its context), so the composition lands on discrete steps: neighbours at 80 GiB score arms at 98 GiB (inside
# the frozen 95-110 gate), neighbours at 97 GiB score them at 115 GiB (outside it - that is what the first
# 315gsync2 attempt did with LO=85, and its arms are archived as 315gsync2-overshoot-console.log). There is no
# setting that lands at 101-105 GiB with this granularity; 98 GiB is the in-regime point this box can hold.
echo "fp2: running $ARMS arms at REGIME_LO=${REGIME_LO:-80} $(date -Is)"
CBIN="$B/bin" FBIN="$B/bin" NOLOCK=1 \
REGIME_LO=${REGIME_LO:-80} REGIME_HI=${REGIME_HI:-92} \
ARM_TIMEOUT=${ARM_TIMEOUT:-1800} \
  bash "$HOME/wt/315-attrib.sh" "$ARMS" "$TAG" 999 \
  "GGML_HRX_TENSOR_FP=$NAMES GGML_HRX_TENSOR_FP_NANONLY=1"

echo "=== fp2 readout $(date -Is) ==="
CONSOLE=$HOME/wt/$TAG-console.log
grep -hE "^  pair" "$CONSOLE" || true
echo "fault: $(grep -c 'arm l: \(FAULT\|INVALID\)' "$CONSOLE" || true)  clean: $(grep -c 'arm l: OK' "$CONSOLE" || true)  oor: $(grep -c 'arm l: OOR' "$CONSOLE" || true)"
for d in "$HOME"/wt/rt-det/"$TAG"-*/; do
  [ -d "$d" ] || continue
  b=$(basename "$d" | sed "s/$TAG-//;s/-GLM.*//")
  echo "--- $b guard=$(grep -c 'HRX returned NaN logits' "$d/server.log" 2>/dev/null || true) ids=$(grep -c 'hrx-ids-fp' "$d/server.log" 2>/dev/null || true)"
  grep -m6 "hrx-tensor-fp" "$d/server.log" 2>/dev/null | cut -c1-200
  grep "hrx-ids-fp" "$d/server.log" 2>/dev/null | grep -E "oor=[1-9]|dup_pairs=[1-9]" | head -4 | cut -c1-200
done
echo "The per-node lines are now ordered; a T=1-only NaN row still carries no per-token information."
