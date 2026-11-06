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
# npu_moe_container_test.sh <c++-compiler> <source-root> <test.cpp>
#
# The MoE container-convention contract (npu/forward/include/moe_container.h, engine#1035) is pure
# name/config/geometry logic plus the shipped chunk decoder: no runtime, no device, no NPU
# toolchain. It compiles itself with the build's own compiler at test time, so it cannot become
# "Not Run" by being absent from CI's explicit target list.
set -u
CXX=${1:?usage: npu_moe_container_test.sh <c++-compiler> <source-root> <test.cpp>}
ROOT=${2:?}
SRC=${3:?}
work=$(mktemp -d "${TMPDIR:-/tmp}/npu-moe-container.XXXXXX")
trap 'rm -rf "$work"' EXIT
"$CXX" -std=c++17 -O1 -Wall -Wextra -Wno-unknown-pragmas -I "$ROOT/npu/forward/include" \
    "$SRC" "$ROOT/npu/forward/src/q4nx_dequant.c" -o "$work/t" || {
        echo "npu_moe_container_test: compile failed"; exit 1; }
"$work/t"
