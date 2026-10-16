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
# engine#315 chain probe. Section 40's one-bit causality test, run as an env change rather than a rebuild:
# GGML_HRX_TENSOR_FP accepts a comma-joined name substring list, so naming the block's residual, attention,
# norm and FFN tensors makes the fingerprint scan the *whole* graph for exactly those, and the first one
# carrying NaN says whether the NaN was already in the block's input (`l_out-40`, `ffn_inp-41`,
# `ffn_norm-41`) or was fabricated inside the FFN branch (`ffn_shexp-41`, `ffn_swiglu-41`,
# `ffn_moe_out-41`). Names are from the pin's own builder, src/models/deepseek.cpp:130-176 plus
# llama-graph.cpp's build_moe_ffn; this model's output norm is the unnumbered `norm`.
#
# Both instruments in this env are cleared of fabricating NaN (section 41): the copy probe only fprintfs, and
# the sentinel writes only into the 619520-byte logits allocation. What they can do is perturb timing - which
# is wanted here, because the arms have to fault for the probe to have anything to read.
#
# Diagnostic, not acceptance: no registered criterion changes and no arm from it is eligible.
#
# Usage: 315-chain.sh [pairs] [tag]     (waits for the box lock, then runs; NOLOCK=1 goes to the driver)
set -uo pipefail
PAIRS=${1:-6}; TAG=${2:-315chain}

exec 9>"$HOME/.cache/lax-decode/box.lock"
if flock -w "${CHAIN_LOCK_WAIT:-5400}" 9; then
  echo "chain: box lock held $(date -Is) after waiting"
else
  echo "chain: box lock unobtainable after ${CHAIN_LOCK_WAIT:-5400}s - aborting rather than running concurrently"
  exit 3
fi

NAMES="l_out-39,l_out-40,l_out-41,attn_norm-41,Qcur-41,Kcur-41,Vcur-41,ffn_inp-41,ffn_norm-41,ffn_shexp-41,ffn_swiglu-41,ffn_moe_out-41,ffn_out-41,attn_norm-42,ffn_inp-42,ffn_norm-42,ffn_shexp-42,ffn_moe_out-42,l_out-42,norm,result_norm,result_output"
echo "chain: names=$NAMES"

NOLOCK=1 exec bash "$HOME/wt/315-attrib.sh" "$PAIRS" "$TAG" 999 \
  "GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_TENSOR_FP=$NAMES GGML_HRX_TENSOR_FP_NANONLY=1"
