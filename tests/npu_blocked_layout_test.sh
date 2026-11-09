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
# npu_blocked_layout_test.sh <c++-compiler> <include-dir> <test.cpp>
#
# The device weight layout contract (q4nx_weight_layout) is pure index math: no
# runtime, no device, no NPU toolchain.  CI builds an explicit target list, so a
# test that wants its own executable is one "Not Run" away from never executing;
# this wrapper compiles the check with the build's own compiler at test time
# instead, which keeps the contract tested on the hosted runner without adding a
# target to that list.
set -u
CXX=${1:?usage: npu_blocked_layout_test.sh <c++-compiler> <include-dir> <test.cpp>}
INC=${2:?}
SRC=${3:?}
work=$(mktemp -d "${TMPDIR:-/tmp}/npu-blocked-layout.XXXXXX")
trap 'rm -rf "$work"' EXIT
# Two packers join the link (engine #1416): the row-major reference (q4nx_pack.c) and the extracted production packer
# (q4nx_pack_blocked.cpp). q4nx_dequant.c joins too, because q4nx_pack.c's fused q4_1 packer (#1713) decodes its tiles
# with it. The test asserts that the production image is the reference image permuted by bpos, so the layout contract
# and the packer that feeds the microkernel are checked against each other rather than in isolation.
LIB=${LIB_DIR:-$(cd "$(dirname "$SRC")/../npu/forward/src" 2>/dev/null && pwd)}
"$CXX" -std=c++17 -O1 -Wall -Wextra -I "$INC" "$SRC" "$LIB/q4nx_pack.c" "$LIB/q4nx_dequant.c" "$LIB/q4nx_pack_blocked.cpp" \
    -fopenmp -lm -o "$work/t" || {
    echo "npu_blocked_layout_test: compile failed"; exit 1; }
"$work/t"
