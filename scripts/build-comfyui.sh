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
# build-comfyui.sh <prefix>
#
# Builds ComfyUI.cpp pinned in third_party/comfyui.cpp (1bit-MONSTER/comfyui.cpp, GPL-3.0)
# into <prefix>/comfyui_cpp. It is a separate program: the engine runs it as a child
# process (`1bit comfy`) and never links it (docs/comfyui.md). Needs LibTorch; the
# build takes torch's CMake config from `python3 -c "import torch"`, or from Torch_DIR.
set -euo pipefail
prefix=${1:?usage: build-comfyui.sh <prefix>}
root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/third_party/comfyui.cpp
[ -f "$src/CMakeLists.txt" ] || { echo "third_party/comfyui.cpp is empty: git submodule update --init third_party/comfyui.cpp"; exit 1; }
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)
cmake -S "$src" -B "$prefix/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$prefix/build" -j"$(nproc)" --target comfyui_cpp
cp "$prefix/build/comfyui_cpp" "$prefix/comfyui_cpp"
echo "built $prefix/comfyui_cpp (comfyui.cpp $(git -C "$src" rev-parse --short=12 HEAD 2>/dev/null || echo unknown))"
