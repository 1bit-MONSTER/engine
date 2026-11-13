#!/bin/sh
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
# tests/audio_kernel_plan.sh -- the launch plan a device route reads (engine#1783).
#
# The plan is the artifact the encoder's device arm consumes: one entry per projection, in the order the graph runs them, naming the
# ELF each one needs at the width the ENGINE stages it to. Three properties make it usable, and each has been wrong somewhere today:
# the list is not empty (an empty plan must not read as a servable pack), `order` is dense from 0 (a route indexes by it), and every
# staged width is on the engine's grid -- K a multiple of 256, N of 512 (`npu/forward/src/q4nx_pack.c`) -- because the forward asks
# for the staged name and a plan naming the model's own width sends a route after a file that is not there.
#
# Usage: audio_kernel_plan.sh <audio_kernel_shapes> <pack dir>
set -u
tool=${1:?audio_kernel_shapes binary}
pack=${2:-}
[ -n "$pack" ] || pack=${ONEBIT_TEST_WHISPER_PACK:-}
[ -n "$pack" ] || { echo "SKIP: no pack (pass one, or set ONEBIT_TEST_WHISPER_PACK)"; exit 77; }
[ -d "$pack" ] || { echo "SKIP: $pack is not a directory"; exit 77; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
plan=$tmp/plan.json
if ! "$tool" "$pack" --write-plan "$plan" > "$tmp/out.txt" 2>&1; then
    echo "FAIL the plan could not be written:"; sed 's/^/    /' "$tmp/out.txt"; exit 1
fi
fail=0
ok()  { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1"; fail=1; }

n=$(grep -c '"order":' "$plan" || true)
[ "$n" -gt 0 ] && ok "the plan lists $n projection(s)" || bad "the plan lists no projection -- nothing would launch"

orders=$(grep -o '"order": [0-9]*' "$plan" | grep -o '[0-9]*' | tr '\n' ' ')
want=$(seq 0 $((n - 1)) | tr '\n' ' ')
[ "$orders" = "$want" ] && ok "...ordered 0..$((n - 1)) with no gaps" || bad "order is not dense from 0: $orders"

offk=$(grep -o '"staged_k": [0-9]*' "$plan" | grep -o '[0-9]*' | awk '$1 % 256 != 0' | wc -l)
offn=$(grep -o '"staged_n": [0-9]*' "$plan" | grep -o '[0-9]*' | awk '$1 % 512 != 0' | wc -l)
[ "$offk" -eq 0 ] && ok "every staged K is a multiple of 256" || bad "$offk staged K value(s) off the grid"
[ "$offn" -eq 0 ] && ok "every staged N is a multiple of 512" || bad "$offn staged N value(s) off the grid"

# The plan's ELF names are the family's, and they differ from the model's own widths exactly where staging bites.
bad_elf=$(grep -o '"elf_b8": "[^"]*"' "$plan" | grep -vc 'full_i8_[A-Z]*_K[0-9]*_N[0-9]*_b8\.elf' || true)
[ "$bad_elf" -eq 0 ] && ok "every elf_b8 is a full_i8_<role>_K<k>_N<n>_b8.elf name" \
                     || bad "$bad_elf elf_b8 name(s) do not look like a family member"
if grep -q '"staged_n": 1024' "$plan"; then
    grep -q '_N1024_b8.elf' "$plan" && ok "a staged N1024 is named N1024 (768 -> 1024), not the model's 768" \
                                  || bad "a launch is staged to 1024 but its ELF name is not"
fi

[ "$fail" -eq 0 ] && { echo "audio kernel plan: ok"; exit 0; }
echo "audio kernel plan: FAILED"; exit 1
