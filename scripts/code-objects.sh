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
# Count the code objects bundled into a fat library: its embedded ELF headers.
#
# engine#1443: the packers asked `strings -a libggml-hrx.so | grep -c amdgcn-amd-amdhsa`, which tests a string
# that appears only in some ways of assembling a fatbin, not the property it names. The libraries this box
# actually ships carry 56 and 39 bundled code objects and **zero** occurrences of that string -- so the check
# refused every real drop and would have accepted one with a third of the kernels. Counting the 4-byte ELF
# header is the property itself and needs no toolchain.
#
# usage: scripts/code-objects.sh FILE   -> prints the count (0 for a file with no bundled object)
set -u
f=${1:?usage: code-objects.sh FILE}
[ -f "$f" ] || { echo 0; exit 1; }
pat=$(printf '\177ELF')
# -o makes grep print one line per match, so `wc -l` counts matches -- `grep -c` would count *lines*, which is
# one for a file that happens to hold several headers on one line (measured: three headers, -c said 1).
n=$(LC_ALL=C grep -a -o "$pat" "$f" 2>/dev/null | wc -l)
echo "${n:-0}"
