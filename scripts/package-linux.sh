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
# beside itself ($ORIGIN). The NPU lane has no separate runtime library to ship: its runtime is
# compiled into 1bit (scripts/weekly.sh checks the package's runtime dependencies), and it carries
# the GGUF-on-NPU helper scripts (scripts/, tools/): public
# Apache-2.0 source, the one exception package_checks makes to the tarball's no-source rule.
# HRX still needs TheRock (/opt/rocm-therock) on the
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

# the sealed NPU kernel sets (scripts/weekly.sh, WEEKLY_NPU_KERNELS): encrypted artifacts, packed
# beside 1bit in npu/kernels, where it looks for them
# Only the artifact surface is shipped: <set>/layout.json and <set>/layer1_*/layer1.elf. The set
# directory also holds layer1.work/ (plaintext MLIR, object files, intermediate ELFs) because that is
# where it was built; copying the tree would ship the kernel method and feed those static ELFs to
# patchelf. package_checks asserts this allowlist on the tarball.
if [ -d "$build/npu/kernels" ]; then
    mkdir -p "$stage/npu/kernels"
    for set in "$build"/npu/kernels/*/; do
        setname=$(basename "$set")   # not $name: that is the package name tar uses below
        [ -f "$set/layout.json" ] || continue
        mkdir -p "$stage/npu/kernels/$setname"
        cp -a "$set/layout.json" "$stage/npu/kernels/$setname/"
        for layer in "$set"/layer1_*/; do
            [ -f "$layer/layer1.elf" ] || continue
            mkdir -p "$stage/npu/kernels/$setname/$(basename "$layer")"
            cp -a "$layer/layer1.elf" "$stage/npu/kernels/$setname/$(basename "$layer")/"
        done
    done
    echo "  npu/kernels: $(ls "$stage/npu/kernels" | wc -l) sets, $(find "$stage/npu/kernels" -name layer1.elf | wc -l) sealed kernels"
fi

# The NPU lane's GGUF path runs public helper scripts (the repack, its declared-scale guard and the
# lane packer). 1bit looks for them beside itself (app/built_path.h, packaged_path), so they ship at
# the same relative path a build tree has them. They are Apache-2.0 and already published in this
# repository -- the tarball still carries no source of a private component.
if [ -d "$build/npu" ]; then
    mkdir -p "$stage/scripts" "$stage/tools"
    cp "$src/scripts/repack_gguf.py" "$src/scripts/check_repack_config.py" "$stage/scripts/"
    cp "$src/tools/gguf_to_q4nx.py" "$stage/tools/"
    echo "  scripts, tools: the GGUF-on-NPU helpers ($(ls "$stage/scripts" "$stage/tools" | wc -l) files)"
fi

# Every program and library finds its libraries beside itself (scripts/pack-elf.sh): a library it
# loads from elsewhere in the build tree (HRX's libhrx, libloomc) or LLVM's OpenMP runtime
# (libomp.so, which ggml's CPU code links when TheRock's amdclang builds llama.cpp) is copied beside
# it; RUNPATH entries outside the build tree (TheRock in /opt) stay, since HRX needs TheRock anyway.
pack_localize "$stage" "$build"
# pack_localize gives every program and library $ORIGIN plus whatever SDK entries its link needed; the
# engine binary at the top of the tree is the one it does not visit (depth 1), so drop its rpath here.
patchelf --remove-rpath "$stage/1bit"

# symbol tables and debug info out of every program and library: the release ships compiled code
# only (docs/releases.md, "Private components"); the dynamic symbols they link by stay
pack_strip "$stage"

# nothing may still point into the build machine's home, carry the compiler's identification, or load
# through an absolute RUNPATH
bad=$(pack_build_rpaths "$stage" "$build")
[ -z "$bad" ] || { echo "RPATH into the build machine left:"; echo "$bad"; exit 1; }
hygiene=$(pack_artifact_hygiene "$stage")
[ -z "$hygiene" ] || { echo "artifact hygiene:"; echo "$hygiene"; exit 1; }

cp "$src/LICENSE" "$src/NOTICE" "$src/LICENSE-BINARY" "$src/NOTICE-BINARY" "$stage/"
# the stock Lemonade PPA's lemond (Ubuntu 26.04) runs our HRX llama-server through lemonade-hrx
# (docs/lemonade.md, "Ubuntu 26.04: the PPA's lemond"); install-lemonade-ppa.sh wires its drop-in
if [ -x "$stage/hrx/llama/bin/llama-server" ]; then
    cp "$src/scripts/lemonade-hrx" "$src/scripts/install-lemonade-ppa.sh" "$stage/"
    mkdir -p "$stage/lemonade"
    cp "$src/config/lemonade/1bit-hrx.conf" "$stage/lemonade/"
fi
cat > "$stage/README.txt" <<EOF
1bit engine $ver for Linux x86_64 (Strix Halo / Ryzen AI): https://1bit.gg/

  ./1bit serve -m model.gguf --port 8000      (--device auto: HRX0, else the CPU)
$( [ -d "$stage/npu/kernels" ] && printf '  ./1bit serve -m <Qwen3 NPU model dir> --port 8000   (the 1bit NPU engine, XDNA 2 NPU)\n  ./1bit serve -m model.gguf --device npu --port 8000  (a GGUF repacked for the NPU: needs python3\n                                                       with numpy and gguf, and the Q4NX converter\n                                                       in ONEBIT_Q4NX_CONVERTER or ~/np-model-generic)\n' )

Ubuntu 26.04 with the Lemonade PPA's lemond: sudo ./install-lemonade-ppa.sh
  (installs to /opt/1bit; lemond's HRX recipe then runs this llama-server)

Keep the directory layout: 1bit finds its backends beside itself.
HRX needs TheRock in /opt/rocm-therock; the NPU needs the amdxdna
kernel driver. Docs: https://1bit.gg/  Code: https://github.com/1bit-MONSTER/engine
EOF
tar -C "$(dirname "$stage")" -cf - "$name" | zstd -q -T4 -19 -o "$out/$name.tar.zst"
ls -la "$out/$name.tar.zst"
