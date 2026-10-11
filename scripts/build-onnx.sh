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
# ONNX Runtime GenAI with the RyzenAI provider: set RYZENAI_INSTALL_PATH to it, and the build
# uses that instead.
#
# Linux NPU (docs/onnx.md, "Linux NPU"): RYZENAI_INSTALL_PATH is the venv Ryzen AI Software's
# install_ryzen_ai.sh made (`-p <venv>`); its deployment/lib holds libonnxruntime-genai.so,
# libonnxruntime.so and libonnxruntime_providers_ryzenai.so. ryzenai-server's build copies that
# whole directory beside the binary (DT_RPATH $ORIGIN finds it), and this script then clears the
# executable-stack request from any copy that carries one: a 7.x kernel refuses to load such a
# library (`patchelf --clear-execstack`, on the copies only; the venv is never modified). At run
# time `1bit serve` points the child at the venv and XRT with ONEBIT_RYZENAI_ROOT / ONEBIT_XRT_ROOT.

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
fi

# DT_RPATH $ORIGIN: the binary finds its libraries beside it, and so does ONNX Runtime GenAI when
# it loads libonnxruntime.so (a RUNPATH would not cover that dlopen)
cmake -S "$src" -B "$work/build" -DCMAKE_BUILD_TYPE=Release -DOGA_ROOT="$oga_root" \
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--disable-new-dtags -Wl,-rpath,\$ORIGIN" > "$work/cmake.log"
cmake --build "$work/build" -j"$(nproc)" > "$work/build.log" 2>&1 || { tail -20 "$work/build.log"; exit 1; }
cp -a "$work/build/bin/." "$prefix/"
[ -n "$ort_lib" ] && cp -a "$ort_lib"/libonnxruntime.so* "$prefix/"

# Executable stacks: a library whose PT_GNU_STACK asks for RWE fails to load on Linux 7.x
# (and makes the whole process's stack executable before that). Scan the copies beside the
# binary; clear the request on those that carry it. Only our copies change.
if [ -n "${RYZENAI_INSTALL_PATH:-}" ]; then
    rwe=()
    for so in "$prefix"/*.so "$prefix"/*.so.*; do
        [ -f "$so" ] && [ ! -L "$so" ] || continue
        readelf -lW "$so" 2>/dev/null | grep -q 'GNU_STACK.*RWE' && rwe+=("$so")
    done
    if [ ${#rwe[@]} -gt 0 ]; then
        command -v patchelf > /dev/null || { echo "build-onnx.sh: ${#rwe[@]} copied libraries request an executable stack (${rwe[*]##*/}) and patchelf is not installed to clear it: apt install patchelf"; exit 1; }
        for so in "${rwe[@]}"; do patchelf --clear-execstack "$so"; echo "cleared executable stack: ${so##*/}"; done
    else
        echo "executable-stack scan: none of the copied libraries request one"
    fi
fi
echo "built $prefix/ryzenai-server (ryzenai-server $(git -C "$src" rev-parse --short=12 HEAD 2>/dev/null || echo unknown))"
