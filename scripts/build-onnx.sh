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
# build-onnx.sh <prefix>
#
# Builds ryzenai-server, pinned in third_party/ryzenai-server (lemonade-sdk/ryzenai-server, MIT),
# into <prefix>/ryzenai-server, with the libraries it loads beside it. It is the engine's ONNX
# Runtime GenAI backend (`1bit serve --device onnx`, docs/onnx.md), run as a child process.
#
# By default it builds against Microsoft's ONNX Runtime GenAI 0.11.2 and ONNX Runtime 1.23.2
# (both MIT), fetched pinned and checked against their sha256: models run on the CPU. For the
# NPU and hybrid modes, AMD's Ryzen AI Software (its own licence, installed separately) provides
# ONNX Runtime GenAI with the Vitis AI provider: set RYZENAI_INSTALL_PATH to it, and the build
# uses that instead.
#
# ONNX_WEBGPU=1 runs models on the GPU on Linux: it builds ONNX Runtime 1.23.2 from source with its
# WebGPU provider (Dawn, over Vulkan) in place of the CPU release library; Microsoft publishes no
# Linux build of it. A model then runs on the GPU when its genai_config.json asks for "webgpu"
# (ONNX Runtime GenAI's builder: -e webgpu). The build needs Node.js (set NODE_DIR to its bin/ if
# `node` is not on PATH), takes about 20 minutes and about 6 GB in <prefix>/work.
set -euo pipefail
prefix=${1:?usage: build-onnx.sh <prefix>}
root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/third_party/ryzenai-server
if [ ! -f "$src/CMakeLists.txt" ] && git -C "$root" rev-parse --git-dir > /dev/null 2>&1; then
    echo "fetching third_party/ryzenai-server"
    git -C "$root" submodule update --init third_party/ryzenai-server
fi
[ -f "$src/CMakeLists.txt" ] || { echo "third_party/ryzenai-server is empty: git submodule update --init third_party/ryzenai-server"; exit 1; }
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)
work=$prefix/work
mkdir -p "$work"

OGA=https://github.com/microsoft/onnxruntime-genai/releases/download/v0.11.2/onnxruntime-genai-0.11.2-linux-x64.tar.gz
OGA_SHA=0f63ef0fd3ba6a5c49b0b7d58e28e734c8a95deb8162221f50c716773341b286
ORT=https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
ORT_SHA=1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b
ORT_SRC_TAG=v1.23.2
ORT_SRC_COMMIT=a83fc4d58cb48eb68890dd689f94f28288cf2278

fetch() {   # fetch <url> <sha256> <file>: download once, verify every time
    local f="$work/$3"
    if ! { [ -f "$f" ] && echo "$2  $f" | sha256sum -c --quiet 2>/dev/null; }; then
        curl -fsSL -o "$f.part" "$1"
        echo "$2  $f.part" | sha256sum -c --quiet || { echo "build-onnx.sh: $1: checksum mismatch"; exit 1; }
        mv "$f.part" "$f"
    fi
    echo "$f"
}

if [ -n "${RYZENAI_INSTALL_PATH:-}" ]; then
    oga_root=$RYZENAI_INSTALL_PATH       # Ryzen AI Software: NPU, hybrid and CPU
    ort_lib=""
else
    # Microsoft's releases, in the layout ryzenai-server's build looks for (<root>/deployment/lib)
    rm -rf "$work/oga" "$work/ort" && mkdir -p "$work/oga" "$work/ort"
    tar xzf "$(fetch $OGA $OGA_SHA oga.tgz)" -C "$work/oga" --strip-components=1
    tar xzf "$(fetch $ORT $ORT_SHA ort.tgz)" -C "$work/ort" --strip-components=1
    oga_root=$work/oga-root
    mkdir -p "$oga_root/deployment"
    ln -sfn "$work/oga/lib" "$oga_root/deployment/lib"
    ort_lib=$work/ort/lib
    if [ "${ONNX_WEBGPU:-0}" = 1 ]; then
        [ -n "${NODE_DIR:-}" ] && export PATH="$NODE_DIR:$PATH"
        command -v node > /dev/null || { echo "build-onnx.sh: ONNX_WEBGPU needs Node.js (set NODE_DIR)"; exit 1; }
        ort_src=$work/ort-src
        if [ ! -d "$ort_src/.git" ]; then
            git clone -q --depth 1 --recurse-submodules --shallow-submodules -b "$ORT_SRC_TAG" \
                https://github.com/microsoft/onnxruntime "$ort_src"
        fi
        [ "$(git -C "$ort_src" rev-parse HEAD)" = "$ORT_SRC_COMMIT" ] || { echo "build-onnx.sh: $ort_src is not ONNX Runtime $ORT_SRC_COMMIT"; exit 1; }
        echo "building ONNX Runtime $ORT_SRC_TAG with the WebGPU provider (log: $work/ort-webgpu.log)"
        # CMake 4 rejects some dependencies' old minimum versions; GCC 15 no longer includes
        # <cstdint> transitively, which ONNX Runtime 1.23 relies on
        (cd "$ort_src" && ./build.sh --config Release --build_dir "$work/ort-webgpu" --build_shared_lib \
            --use_webgpu --parallel "$(nproc)" --skip_tests --compile_no_warning_as_error \
            --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 onnxruntime_BUILD_UNIT_TESTS=OFF \
            "CMAKE_CXX_FLAGS=-include cstdint") > "$work/ort-webgpu.log" 2>&1 \
            || { tail -20 "$work/ort-webgpu.log"; exit 1; }
        ort_lib=$work/ort-webgpu/Release
        # the notices of what the library carries: ONNX Runtime's dependencies, and Dawn
        cp "$ort_src/ThirdPartyNotices.txt" "$prefix/onnxruntime-ThirdPartyNotices.txt"
        cp "$work/ort-webgpu/Release/_deps/dawn-src/LICENSE" "$prefix/dawn-LICENSE"
    fi
fi

# DT_RPATH $ORIGIN: the binary finds its libraries beside it, and so does ONNX Runtime GenAI when
# it loads libonnxruntime.so (a RUNPATH would not cover that dlopen)
cmake -S "$src" -B "$work/build" -DCMAKE_BUILD_TYPE=Release -DOGA_ROOT="$oga_root" \
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--disable-new-dtags -Wl,-rpath,\$ORIGIN" > "$work/cmake.log"
cmake --build "$work/build" -j"$(nproc)" > "$work/build.log" 2>&1 || { tail -20 "$work/build.log"; exit 1; }
cp -a "$work/build/bin/." "$prefix/"
[ -n "$ort_lib" ] && cp -a "$ort_lib"/libonnxruntime.so* "$prefix/"
echo "built $prefix/ryzenai-server (ryzenai-server $(git -C "$src" rev-parse --short=12 HEAD 2>/dev/null || echo unknown))"
