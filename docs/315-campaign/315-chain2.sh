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
# engine#315 chain probe, second pass. Prepared, not launched: 315-chain.sh answers whether the NaN was
# already in block 41's input, and this one is for the case where it was not - i.e. the NaN is born inside
# the block's FFN/MoE, between ffn_norm and ffn_moe_out.
#
# Why a second pass at all: with GGML_HRX_TENSOR_FP_NANONLY=1 the fingerprint prints only nodes that carry
# NaN, in graph order, so the FIRST printed line is the earliest NaN-bearing named node. That makes the name
# list a bisection tool rather than a single query - but only for nodes that have names. The earlier window
# showed auto-named nodes (`node_2948`, `node_2985`) that no name list can match, and the MoE's own
# intermediates were not named in the first pass.
#
# This list widens both directions: `l_out-24..l_out-42` walks the residual stream back towards the birth of
# the corruption if it turns out to precede block 41, and the block-41/42 chain adds the MoE intermediates
# (build_moe_ffn in the pin names them ffn_moe_gate/up/gate_up/swiglu/weighted/down/probs/weights/topk).
# Roughly 60 device-to-host reads per graph instead of 16: more timing perturbation, which is wanted here
# because the arms have to fault - and section 41 established that these reads cannot manufacture a NaN.
#
# Diagnostic, not acceptance: no registered criterion changes and no arm from it is eligible.
#
# Usage: 315-chain2.sh [pairs] [tag]
set -uo pipefail
PAIRS=${1:-6}; TAG=${2:-315chain2}

exec 9>"$HOME/.cache/lax-decode/box.lock"
if flock -w "${CHAIN2_LOCK_WAIT:-5400}" 9; then
  echo "chain2: box lock held $(date -Is)"
else
  echo "chain2: lock unobtainable after ${CHAIN2_LOCK_WAIT:-5400}s - aborting rather than running concurrently"
  exit 3
fi

# residual stream, walking backwards from the block that first showed NaN
BACK=""
for il in 42 41 40 39 38 36 34 32 30 28 26 24; do BACK="$BACK,l_out-$il"; done
# block 41 and 42: attention input, projections, and the FFN/MoE chain in dataflow order
CHAIN=""
for il in 41 42; do
  CHAIN="$CHAIN,attn_norm-$il,Qcur-$il,Kcur-$il,Vcur-$il,ffn_inp-$il,ffn_norm-$il"
  # ffn_moe_topk is deliberately absent: it is an i32 index tensor, so reading it as floats makes
  # its NaN count meaningless (i32 bit patterns that look like NaN), and with NANONLY=1 a node
  # that does not print tells us nothing. The f32 routing tensors (probs, weights) are kept.
  CHAIN="$CHAIN,ffn_moe_probs-$il,ffn_moe_weights-$il,ffn_moe_gate-$il,ffn_moe_up-$il"
  CHAIN="$CHAIN,ffn_moe_gate_up-$il,ffn_swiglu-$il,ffn_moe_weighted-$il,ffn_moe_down-$il,ffn_moe_out-$il"
  CHAIN="$CHAIN,ffn_shexp-$il,ffn_out-$il"
done
NAMES="norm,result_norm,result_output${BACK}${CHAIN}"
echo "chain2: $(printf '%s' "$NAMES" | tr ',' '\n' | grep -c .) names"

NOLOCK=1 exec bash "$HOME/wt/315-attrib.sh" "$PAIRS" "$TAG" 999 \
  "GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_TENSOR_FP=$NAMES GGML_HRX_TENSOR_FP_NANONLY=1"
