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
# package-linux.sh <build> <out-dir> <version>
#
# Packs a build into <out-dir>/1bit-<version>-linux-x86_64.tar.zst: 1bit, and every backend the
# build has, at the same paths relative to 1bit as in the build tree, which is where 1bit looks
# once the build tree is gone (app/built_path.h). Each program and library finds its libraries
# beside itself ($ORIGIN). The NPU lane has no separate runtime library to ship (no XRT;
# weekly.sh checks). HRX still needs TheRock (/opt/rocm-therock) on the
# machine, as it does when built there. The engine is pure HRX with HIP kernels on the GPU: no
# Vulkan, ROCm backend, Metal, ZINC or CUDA. This tarball is not published on its own: weekly.sh
# unpacks it into the Lemonade + onebit package as 1bit/ (docs/releases.md).
set -euo pipefail

[ $# -ge 3 ] || { sed -n '16,27p' "$0"; exit 2; }
build=$(cd "$1" && pwd) out=$2 ver=$3
src=$(cd "$(dirname "$0")/.." && pwd)
name=1bit-$ver-linux-x86_64
stage=$(mktemp -d)/$name
trap 'rm -rf "$(dirname "$stage")"' EXIT
mkdir -p "$stage" "$out"

# shellcheck source=scripts/pack-elf.sh
. "$src/scripts/pack-elf.sh"

# <rel dir> <program ...>: the programs, and every shared library in that directory
put() {
    local rel=$1; shift
    [ -d "$build/$rel" ] || return 0
    mkdir -p "$stage/$rel"
    local f
    for f in "$@"; do [ -f "$build/$rel/$f" ] && cp -a "$build/$rel/$f" "$stage/$rel/"; done
    find "$build/$rel" -maxdepth 1 \( -name '*.so' -o -name '*.so.*' \) -exec cp -a {} "$stage/$rel/" \;
    echo "  $rel: $(ls "$stage/$rel" | wc -l) files"
}

cp -a "$build/1bit" "$stage/"
put hrx/llama/bin llama-server llama-bench
put ds4/hip ds4-server ds4 ds4-bench
put onnx ryzenai-server
# the licence texts build-onnx.sh puts beside it (ryzenai-server, ONNX Runtime GenAI, ONNX Runtime)
if [ -d "$stage/onnx" ]; then
    for f in "$build"/onnx/*-LICENSE.txt "$build"/onnx/*-ThirdPartyNotices.txt; do
        if [ -f "$f" ]; then cp "$f" "$stage/onnx/"; fi
    done
fi

# Every program and library finds its libraries beside itself (scripts/pack-elf.sh): a library it
# loads from elsewhere in the build tree (HRX's libhrx, libloomc) or LLVM's OpenMP runtime
# (libomp.so, which ggml's CPU code links when TheRock's amdclang builds llama.cpp) is copied beside
# it; RUNPATH entries outside the build tree (TheRock in /opt) stay, since HRX needs TheRock anyway.
pack_localize "$stage" "$build"
patchelf --remove-rpath "$stage/1bit"

# symbol tables and debug info out of every program and library: the release ships compiled code
# only (docs/releases.md, "Private components"); the dynamic symbols they link by stay
pack_strip "$stage"

# nothing may still point into the build machine's home
bad=$(pack_build_rpaths "$stage" "$build")
[ -z "$bad" ] || { echo "RPATH into the build machine left:"; echo "$bad"; exit 1; }

cp "$src/LICENSE" "$src/NOTICE" "$src/LICENSE-BINARY" "$src/NOTICE-BINARY" "$stage/"
cat > "$stage/README.txt" <<EOF
1bit engine $ver for Linux x86_64 (Strix Halo / Ryzen AI): https://1bit.gg/

  ./1bit serve -m model.gguf --port 8000      (--device auto: HRX0, else the CPU)

Keep the directory layout: 1bit finds its backends beside itself.
HRX needs TheRock in /opt/rocm-therock; the NPU needs the amdxdna
kernel driver. Docs: https://1bit.gg/  Code: https://github.com/1bit-MONSTER/engine
EOF
tar -C "$(dirname "$stage")" -cf - "$name" | zstd -q -T4 -19 -o "$out/$name.tar.zst"
ls -la "$out/$name.tar.zst"
