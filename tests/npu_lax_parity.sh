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
#
# npu_lax_parity.sh <prefix> <model-dir> <ref-dir>
#
# The Qwen3.6-35B-A3B lax decode (docs/npu-lax.md) against the fp64 reference: three
# positions from <|im_start|>, all 40 layers as ONE runlist submit per token, weights
# packed at load time from <model-dir>/model.q4nx. Every position must have finite
# logits, correlation > 0.9999 with the reference and the reference's argmax. Then one
# chat turn must mention Paris.
#   <prefix>   scripts/build-lax.sh's prefix
#   <ref-dir>  open_kernels/model/make_decode.py --requant --layers 40 --tokens 3 output
set -euo pipefail
prefix=$(cd "${1:?usage: npu_lax_parity.sh <prefix> <model-dir> <ref-dir>}" && pwd)
model=$(cd "${2:?model dir}" && pwd)
ref=$(cd "${3:?reference dir}" && pwd)
ok=$prefix/src/open_kernels
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

cd "$ok"
python3 model/lax_decode_cfg.py --model-dir "$model" --ref "$ref" --out "$out" \
    --lax-l "$prefix/kernels/lax_l" --lax-a "$prefix/kernels/lax_a" --per 40 --tokens 3
ln -s "$ref"/ref_logits*.bin "$out"/
python3 model/compare_decode.py --out "$out" --tokens 3

answer=$(python3 model/lax_chat.py --model-dir "$model" --lax-l "$prefix/kernels/lax_l" \
    --lax-a "$prefix/kernels/lax_a" --max-new 32 \
    "What is the capital of France? Answer in one sentence." 2>/dev/null)
echo "chat: $answer"
grep -q Paris <<< "$answer"
