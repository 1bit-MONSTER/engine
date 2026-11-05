#!/bin/bash
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
# npu_padded_grid_test.sh <c++-compiler> <source-root> <test.cpp>
#
# The padded-grid contract (engine#1097): pure decoder plus index math, no runtime, no device, no
# model. Same reason as npu_pb16i_layout_test.sh and npu_t2_layout_test.sh: the check compiles itself
# with the build's own compiler at test time, so it cannot become "Not Run" by being missing from
# CI's explicit target list. Unlike the t2 contract it needs the shipped decoder, so
# npu/forward/src/q4nx_dequant.c is compiled (as C) and linked in.
set -u
CXX=${1:?usage: npu_padded_grid_test.sh <c++-compiler> <source-root> <test.cpp>}
ROOT=${2:?}
SRC=${3:?}
work=$(mktemp -d "${TMPDIR:-/tmp}/npu-padded-grid.XXXXXX")
trap 'rm -rf "$work"' EXIT
"$CXX" -x c -std=c11 -O1 -fopenmp -Wall -Wextra -I "$ROOT/npu/forward/include" \
    -c "$ROOT/npu/forward/src/q4nx_dequant.c" -o "$work/q4nx_dequant.o" \
    || { echo "npu_padded_grid_test: decoder compile failed"; exit 1; }
"$CXX" -std=c++17 -O1 -fopenmp -Wall -Wextra -I "$ROOT/npu/forward/include" \
    "$SRC" "$work/q4nx_dequant.o" -o "$work/t" \
    || { echo "npu_padded_grid_test: compile failed"; exit 1; }
"$work/t"
