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
# npu_lax_cpp.sh <1bit> <model-dir> <kernel-dir> <ref-dir>
#
# The Qwen3.6-35B-A3B lax decode driven by the engine itself (`1bit npu-lax`, no Python):
# three positions from the reference's residual inputs, each one runlist submit plus the
# norm and the lm head, scored against the fp64 reference (corr > 0.9999, same argmax);
# then one greedy chat turn, which must answer with Paris.
#   <kernel-dir>  scripts/build-lax.sh's <prefix>/kernels
#   <ref-dir>     open_kernels/model/make_decode.py --requant --tokens 3 output
# The NPU is shared: run it under the box's lock, e.g. flock <lockfile> ctest -R npu_lax_e2e.
set -euo pipefail
bin=${1:?usage: npu_lax_cpp.sh <1bit> <model-dir> <kernel-dir> <ref-dir>}
model=${2:?model dir} kernels=${3:?kernel dir} ref=${4:?reference dir}

"$bin" npu-lax --model "$model" --kernels "$kernels" --parity "$ref" --tokens 3

answer=$("$bin" npu-lax --model "$model" --kernels "$kernels" -n 32 \
    "What is the capital of France? Answer in one sentence.")
echo "chat: $answer"
grep -q Paris <<< "$answer"
