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
# npu_pb16i_layout_test.sh <c++-compiler> <include-dir> <test.cpp>
#
# The pb16i weight image contract (q4nx_weight_layout "pb16i") is pure index math: no
# runtime, no device, no NPU toolchain.  Same reason as npu_blocked_layout_test.sh:
# the check compiles itself with the build's own compiler at test time, so it cannot
# become "Not Run" by being absent from CI's explicit target list.
set -u
CXX=${1:?usage: npu_pb16i_layout_test.sh <c++-compiler> <include-dir> <test.cpp>}
INC=${2:?}
SRC=${3:?}
work=$(mktemp -d "${TMPDIR:-/tmp}/npu-pb16i-layout.XXXXXX")
trap 'rm -rf "$work"' EXIT
"$CXX" -std=c++17 -O1 -Wall -Wextra -I "$INC" "$SRC" -o "$work/t" || {
    echo "npu_pb16i_layout_test: compile failed"; exit 1; }
"$work/t"
