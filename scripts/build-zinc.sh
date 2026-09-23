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
# build-zinc.sh <prefix> [backend]
#
# Builds ZINC pinned in third_party/zinc (upstream zolotukhin/zinc) into
# <prefix>/<backend>/bin/zinc. backend is one of vulkan (default), rocm, cuda:
#   vulkan  needs glslc and the Vulkan loader
#   rocm    needs a ROCm/TheRock root in ROCM_PATH (default /opt/rocm)
#   cuda    needs the CUDA toolkit in CUDA_HOME (default /usr/local/cuda); an
#           NVIDIA box (on WSL2, libcuda comes from /usr/lib/wsl/lib)
#
# Zig is the exact release zinc's build.zig.zon names as minimum_zig_version,
# downloaded once into <prefix>/zig-<version> and checked against the sha256
# ziglang.org publishes. Nothing is installed system-wide (docs/zinc.md).
set -euo pipefail
prefix=${1:?usage: build-zinc.sh <prefix> [vulkan|rocm|cuda]}
backend=${2:-vulkan}
case "$backend" in vulkan|rocm|cuda) ;; *) echo "unknown backend: $backend"; exit 1 ;; esac
root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/third_party/zinc
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)

[ -f "$src/build.zig" ] || { echo "third_party/zinc is empty: git submodule update --init third_party/zinc"; exit 1; }

ver=$(sed -n 's/.*minimum_zig_version *= *"\([^"]*\)".*/\1/p' "$src/build.zig.zon")
[ -n "$ver" ] || { echo "no minimum_zig_version in third_party/zinc/build.zig.zon"; exit 1; }
arch=$(uname -m)
zig_dir=$prefix/zig-$ver
if [ ! -x "$zig_dir/zig" ]; then
    key="$arch-linux"
    read -r url sha < <(curl -fsSL https://ziglang.org/download/index.json |
        python3 -c 'import json,sys; e=json.load(sys.stdin)[sys.argv[1]][sys.argv[2]]; print(e["tarball"], e["shasum"])' "$ver" "$key")
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    curl -fsSL "$url" -o "$tmp/zig.tar.xz"
    echo "$sha  $tmp/zig.tar.xz" | sha256sum -c --quiet
    mkdir -p "$zig_dir"
    tar -xJf "$tmp/zig.tar.xz" -C "$zig_dir" --strip-components=1
fi
"$zig_dir/zig" version

commit=$(git -C "$src" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
out=$prefix/$backend
# Keep zig's caches out of the source tree so the submodule stays clean.
(cd "$src" && "$zig_dir/zig" build -Dbackend="$backend" -Doptimize=ReleaseFast \
    -Dcommit="$commit" \
    --cache-dir "$prefix/.zig-cache/$backend" \
    --global-cache-dir "$prefix/.zig-global-cache" \
    --prefix "$out")

echo "ZINC ($backend, $commit): $out/bin/zinc"
"$out/bin/zinc" --version
