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
# from the Vulkan pin (third_party/llama.cpp-vulkan), <out-dir>/llama-server.exe with the Vulkan
# backend; and from third_party/ryzenai-server, <out-dir>/ryzenai-server.exe with Microsoft's ONNX
# Runtime GenAI and ONNX Runtime DLLs (--device onnx, CPU). 1bit.exe finds its backends beside it.
# Everything it downloads is pinned and checked against its sha256: llvm-mingw (clang 23: the
# engine is C++26), PCRE2 (the tokenizer), Vulkan-Headers, the Vulkan loader's export list
# (vulkan-1.def -> the import library, so no Windows machine or Vulkan SDK is needed) and
# SPIRV-Headers. Host tools: cmake, git, curl, a native C/C++ compiler (for llama.cpp's
# shader generator) and glslc (shaderc).
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
VK_TAG=vulkan-sdk-1.4.357.0
VK_HEADERS=https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/$VK_TAG.tar.gz
VK_HEADERS_SHA=e87dce08116151f6b6d7de6b6faf41498e87e6cf848ff16fa3bd5402190ad4a3
VK_DEF=https://raw.githubusercontent.com/KhronosGroup/Vulkan-Loader/$VK_TAG/loader/vulkan-1.def
VK_DEF_SHA=9ba339b7f5ee2df28487698a6840ecf095fac58b415c2c89ad9f161e9d316378
SPIRV_HEADERS=https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/$VK_TAG.tar.gz
SPIRV_HEADERS_SHA=4d703067a7e06331ccb37bdfed3f9b7879cc61969a2689ae95c95db34a47ff07
OGA_WIN=https://github.com/microsoft/onnxruntime-genai/releases/download/v0.11.2/onnxruntime-genai-0.11.2-win-x64.zip
OGA_WIN_SHA=31aeeb4fa7e1d9bf284f6215d60e0025d534b99add7b0daeaccd729ee8ad1595
ORT_WIN=https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip
ORT_WIN_SHA=0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd

command -v glslc >/dev/null || { echo "build-windows.sh: glslc (shaderc) is required"; exit 1; }

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

# 3. Vulkan: the headers, the loader's import library from its export list, SPIRV-Headers (header
# only: installed into the Windows prefix, where the cross compiler looks)
unpack "$(fetch $VK_HEADERS $VK_HEADERS_SHA vulkan-headers.tar.gz)" "$work/vulkan-headers"
cp -r "$work/vulkan-headers/include/." "$prefix/include/"
"$T/bin/llvm-dlltool" -d "$(fetch $VK_DEF $VK_DEF_SHA vulkan-1.def)" -l "$prefix/lib/libvulkan-1.a" -m i386:x86-64
unpack "$(fetch $SPIRV_HEADERS $SPIRV_HEADERS_SHA spirv-headers.tar.gz)" "$work/spirv-headers"
cmake -S "$work/spirv-headers" -B "$work/spirv-headers/build" -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DSPIRV_HEADERS_ENABLE_TESTS=OFF > /dev/null
cmake --install "$work/spirv-headers/build" > /dev/null

# 4. 1bit.exe: serve, route, comfy (no NPU lane or HRX on Windows yet)
PKG_CONFIG_LIBDIR=$prefix/lib/pkgconfig cmake -S "$root" -B "$work/engine" -DCMAKE_TOOLCHAIN_FILE="$work/toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release -DONEBIT_NPU=OFF -DONEBIT_HRX=OFF > "$work/engine.cmake.log"
cmake --build "$work/engine" --target onebit -j"$jobs" > "$work/engine.build.log" 2>&1 || { tail -20 "$work/engine.build.log"; exit 1; }
cp "$work/engine/1bit.exe" "$out/"

# 5. llama-server.exe with Vulkan, from the Vulkan pin. The pinned llama.cpp uses std::function
# without <functional> in ggml-vulkan-types.h, which libc++ does not include transitively: it is
# included from the command line until upstream adds it.
git -C "$root" submodule update --init third_party/llama.cpp-vulkan
cmake -S "$root/third_party/llama.cpp-vulkan" -B "$work/llama" -DCMAKE_TOOLCHAIN_FILE="$work/toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_VULKAN=ON \
    -DGGML_VULKAN_SHADERS_GEN_TOOLCHAIN="$work/host.cmake" -DSPIRV-Headers_DIR="$prefix/share/cmake/SPIRV-Headers" \
    -DVulkan_INCLUDE_DIR="$prefix/include" -DVulkan_LIBRARY="$prefix/lib/libvulkan-1.a" \
    -DVulkan_GLSLC_EXECUTABLE="$(command -v glslc)" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_EXE_LINKER_FLAGS=-static \
    "-DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00" "-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00 -include functional" \
    > "$work/llama.cmake.log"
cmake --build "$work/llama" --target llama-server -j"$jobs" > "$work/llama.build.log" 2>&1 || { tail -20 "$work/llama.build.log"; exit 1; }
cp "$work/llama/bin/llama-server.exe" "$out/"

# 6. ryzenai-server.exe (--device onnx), from its pin, against Microsoft's Windows releases of ONNX
# Runtime GenAI and ONNX Runtime (MIT; CPU). Upstream's CMake is used unmodified through a small
# wrapper that clears its MSVC-only /SUBSYSTEM:CONSOLE link flag (MinGW links console programs by
# default), and a one-line Wbemidl.h shim covers the case of MinGW's wbemidl.h.
git -C "$root" submodule update --init third_party/ryzenai-server
unzip_to() { [ -d "$2" ] || { mkdir -p "$2.tmp" && (cd "$2.tmp" && cmake -E tar xf "$1") && mv "$2.tmp" "$2"; }; }
unzip_to "$(fetch $OGA_WIN $OGA_WIN_SHA oga-win.zip)" "$work/oga-win"
unzip_to "$(fetch $ORT_WIN $ORT_WIN_SHA ort-win.zip)" "$work/ort-win"
oga=$(ls -d "$work"/oga-win/*/)
ort=$(ls -d "$work"/ort-win/*/)
mkdir -p "$work/rz-wrap" "$work/rz-shim" "$work/oga-root"
echo '#include <wbemidl.h>' > "$work/rz-shim/Wbemidl.h"
cat > "$work/rz-wrap/CMakeLists.txt" <<'RZ'
cmake_minimum_required(VERSION 3.20)
project(ryzenai-server-mingw CXX)
add_subdirectory(${RYZENAI_SERVER_SRC} rz)
set_target_properties(ryzenai-server PROPERTIES LINK_FLAGS "")
RZ
cmake -S "$work/rz-wrap" -B "$work/rz" -DRYZENAI_SERVER_SRC="$root/third_party/ryzenai-server" \
    -DCMAKE_TOOLCHAIN_FILE="$work/toolchain.cmake" -DCMAKE_BUILD_TYPE=Release -DOGA_ROOT="$work/oga-root" \
    -DOGA_INCLUDE="${oga}include" -DOGA_LIB="${oga}lib/onnxruntime-genai.lib" -DCMAKE_EXE_LINKER_FLAGS=-static \
    "-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00 -I$work/rz-shim" > "$work/rz.cmake.log"
cmake --build "$work/rz" -j"$jobs" > "$work/rz.build.log" 2>&1 || { tail -20 "$work/rz.build.log"; exit 1; }
cp "$work/rz/bin/ryzenai-server.exe" "${oga}lib/onnxruntime-genai.dll" "${ort}lib/onnxruntime.dll" \
   "${ort}lib/onnxruntime_providers_shared.dll" "$out/"

ls -la "$out"/*.exe "$out"/*.dll
