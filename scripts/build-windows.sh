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
# build-windows.sh <out-dir>
#
# Cross-builds the engine for Windows 10+ x64 on Linux (docs/windows.md): <out-dir>/1bit.exe;
# from the engine's llama.cpp pin (third_party/llama.cpp, the tree the HRX build uses, built with
# no GPU backend), <out-dir>/llama-server.exe on the CPU. 1bit.exe finds llama-server.exe beside
# it. --device onnx uses the ryzenai-server.exe Lemonade installs (its ryzenai-server backend),
# so the engine builds no ONNX server for Windows (docs/onnx.md). The Windows package has no GPU
# route for now: the engine's GPU route is HRX, which has no Windows build yet, and the engine
# builds no Vulkan (RFC #213 stage 3, docs/hrx.md). Everything it downloads is
# pinned and checked against its sha256: llvm-mingw (clang 23: the engine is C++26) and PCRE2 (the
# tokenizer). Host tools: cmake, git, curl.
set -euo pipefail
out=$(realpath -m "${1:?usage: build-windows.sh <out-dir>}")
root=$(cd "$(dirname "$0")/.." && pwd)
work=$out/work
mkdir -p "$work" "$out"
jobs=$(nproc)

LLVM_MINGW=https://github.com/mstorsjo/llvm-mingw/releases/download/20260922/llvm-mingw-20260922-ucrt-ubuntu-22.04-x86_64.tar.xz
LLVM_MINGW_SHA=bb7bb7654b33d5aa8712acb837c963b2e0c56352560c76105270a3268c665c21
PCRE2=https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.48/pcre2-10.48.tar.gz
PCRE2_SHA=ebcc25aadf2a51fa1fefa9b8bc9e7a79b3dae86870a0f1152a22e42befd46888

# fetch <url> <sha256> <file name>: download once, verify every time
fetch() {
    local f="$work/$3"
    if ! { [ -f "$f" ] && echo "$2  $f" | sha256sum -c --quiet 2>/dev/null; }; then
        curl -fsSL -o "$f.part" "$1"
        echo "$2  $f.part" | sha256sum -c --quiet || { echo "build-windows.sh: $1: checksum mismatch"; exit 1; }
        mv "$f.part" "$f"
    fi
    echo "$f"
}
unpack() {   # unpack <archive> <dir>
    [ -d "$2" ] || { mkdir -p "$2.tmp" && tar xf "$1" -C "$2.tmp" --strip-components=1 && mv "$2.tmp" "$2"; }
}

# 1. the toolchain, and a CMake toolchain file for it
unpack "$(fetch $LLVM_MINGW $LLVM_MINGW_SHA llvm-mingw.tar.xz)" "$work/llvm-mingw"
T=$work/llvm-mingw
prefix=$work/prefix
mkdir -p "$prefix/include" "$prefix/lib"
cat > "$work/toolchain.cmake" <<TC
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER $T/bin/x86_64-w64-mingw32-clang)
set(CMAKE_CXX_COMPILER $T/bin/x86_64-w64-mingw32-clang++)
set(CMAKE_RC_COMPILER $T/bin/x86_64-w64-mingw32-windres)
set(CMAKE_FIND_ROOT_PATH $T/x86_64-w64-mingw32 $prefix)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
TC
printf 'set(CMAKE_C_COMPILER cc)\nset(CMAKE_CXX_COMPILER c++)\n' > "$work/host.cmake"

# 2. PCRE2, static
if [ ! -f "$prefix/lib/libpcre2-8.a" ]; then
    unpack "$(fetch $PCRE2 $PCRE2_SHA pcre2.tar.gz)" "$work/pcre2"
    cmake -S "$work/pcre2" -B "$work/pcre2/build" -DCMAKE_TOOLCHAIN_FILE="$work/toolchain.cmake" \
        -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DPCRE2_BUILD_TESTS=OFF -DPCRE2_BUILD_PCRE2GREP=OFF > /dev/null
    cmake --build "$work/pcre2/build" -j"$jobs" > /dev/null
    cmake --install "$work/pcre2/build" > /dev/null
fi

# 3. 1bit.exe: serve, route (no NPU lane or HRX on Windows yet)
PKG_CONFIG_LIBDIR=$prefix/lib/pkgconfig cmake -S "$root" -B "$work/engine" -DCMAKE_TOOLCHAIN_FILE="$work/toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release -DONEBIT_NPU=OFF -DONEBIT_HRX=OFF > "$work/engine.cmake.log"
cmake --build "$work/engine" --target onebit -j"$jobs" > "$work/engine.build.log" 2>&1 || { tail -20 "$work/engine.build.log"; exit 1; }
cp "$work/engine/1bit.exe" "$out/"

# 4. llama-server.exe on the CPU (GGUF, --device cpu), from the llama.cpp pin, without HRX
git -C "$root" submodule update --init --depth 1 third_party/llama.cpp
cmake -S "$root/third_party/llama.cpp" -B "$work/llama" -DCMAKE_TOOLCHAIN_FILE="$work/toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_HRX=OFF -DGGML_VULKAN=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_EXE_LINKER_FLAGS=-static \
    "-DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00" "-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00" \
    > "$work/llama.cmake.log"
cmake --build "$work/llama" --target llama-server -j"$jobs" > "$work/llama.build.log" 2>&1 || { tail -20 "$work/llama.build.log"; exit 1; }
cp "$work/llama/bin/llama-server.exe" "$out/"

ls -la "$out"/*.exe
