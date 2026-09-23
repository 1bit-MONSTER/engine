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

# npu_lane_e2e.sh <1bit> <model-dir> <kernel-dir> <reference-logits-dir>
#
# The anchor prompt (Qwen3 chat template around "Hi"), 24 greedy steps on the
# NPU fast lane. Every step's logits must be bit-identical to the reference
# lane's (decode_logits_idx<N>.bin, float32).
set -euo pipefail
bin=$1 model=$2 kernels=$3 ref=$4
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

"$bin" npu-run --model "$model" --kernels "$kernels" \
    --ids "151644 872 198 13048 151645 198 151644 77091 198" \
    -n 24 --repetition-penalty 1 --dump-logits "$out" > "$out/tokens.txt"

same=0 total=0
for f in "$ref"/decode_logits_idx*.bin; do
    total=$((total + 1))
    if cmp -s "$f" "$out/$(basename "$f")"; then same=$((same + 1)); else echo "differs: $(basename "$f")"; fi
done
echo "logits bit-identical to the reference: $same/$total steps"
echo "tokens: $(tr '\n' ' ' < "$out/tokens.txt")"
[ "$total" -gt 0 ] && [ "$same" -eq "$total" ]
