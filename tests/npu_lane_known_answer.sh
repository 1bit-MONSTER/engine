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
# The fast lane's known-answer check (docs/npu.md, "Checking the NPU"): one fixed prompt through
# `1bit npu-run`, greedy, and the generated ids against a golden. It needs an NPU, a Q4NX model
# and the lane's kernels (a private component, from ONEBIT_NPU_LANE_KERNELS_DIR), so it runs on
# Strix Halo, not in CI. The lane is deterministic, so the golden is exact: any change to the
# kernels or to the host side moves it.
#
# usage: tests/npu_lane_known_answer.sh <1bit> <model dir> <kernel dir> <golden>
#   golden: line 1 the prompt ids, line 2 the ids it must produce, both space-separated.
set -uo pipefail
bin=${1:?usage: npu_lane_known_answer.sh <1bit> <model dir> <kernel dir> <golden>}
model=${2:?}
kernels=${3:?}
golden=${4:?}

mapfile -t lines <"$golden"
prompt=${lines[0]:-}
want=${lines[1]:-}
if [ -z "$prompt" ] || [ -z "$want" ]; then
    echo "FAIL npu_lane_known_answer: $golden needs a prompt line and an expected line"
    exit 1
fi
# --no-eos: a golden of n ids must mean n decoded steps, not an early stop that happens to match.
n=$(wc -w <<<"$want")
out=$("$bin" npu-run --model "$model" --kernels "$kernels" --ids "$prompt" -n "$n" --no-eos 2>/dev/null)
got=$(tr '\n' ' ' <<<"$out" | xargs)
if [ "$got" = "$want" ]; then
    echo "PASS npu_lane_known_answer: $n ids"
    exit 0
fi
echo "FAIL npu_lane_known_answer"
echo "  prompt: $prompt"
echo "  want:   $want"
echo "  got:    $got"
exit 1
