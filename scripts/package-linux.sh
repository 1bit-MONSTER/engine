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
# package-linux.sh <build> <out-dir> <version> [xrt-root]
#
# Packs a build into <out-dir>/1bit-<version>-linux-x86_64.tar.zst (docs/releases.md): 1bit, and
# every backend the build has, at the same paths relative to 1bit as in the build tree, which
# is where 1bit looks once the build tree is gone (app/built_path.h). Each program and library
# finds its libraries beside itself ($ORIGIN). With the NPU lane, XRT's libraries and the XDNA
# plugin from [xrt-root] go in xrt/lib. HRX and the ROCm backends still need TheRock
# (/opt/rocm-therock) on the machine, as they do when built there.
set -euo pipefail

[ $# -ge 3 ] || { sed -n '16,25p' "$0"; exit 2; }
build=$(cd "$1" && pwd) out=$2 ver=$3 xrt=${4:-}
src=$(cd "$(dirname "$0")/.." && pwd)
name=1bit-$ver-linux-x86_64
stage=$(mktemp -d)/$name
trap 'rm -rf "$(dirname "$stage")"' EXIT
mkdir -p "$stage" "$out"

is_elf() { [ -f "$1" ] && [ "$(head -c 4 "$1" | od -An -c | tr -d ' ')" = '177ELF' ]; }

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
put vulkan/llama/bin llama-server llama-bench
put hrx/llama/bin llama-server llama-bench
put lean/llama/bin llama-server llama-quantize llama-bench
put lean/llama-rocm/bin llama-server llama-bench
for b in vulkan rocm cuda; do
    put "zinc/$b/bin" zinc
    [ -d "$build/zinc/$b/share" ] && cp -a "$build/zinc/$b/share" "$stage/zinc/$b/"   # its shaders
done
for b in rocm cuda cpu; do put "ds4/$b" ds4-server ds4 ds4-bench; done
put onnx ryzenai-server

# Every program and library finds its libraries beside itself. A library it loads from elsewhere
# in the build tree (HRX's libhrx, libloomc) is copied beside it; RUNPATH entries outside the
# build tree (TheRock) stay, since the package needs what the build machine had there.
fix() {  # fix <staged file> <file in the build>
    local f=$1 orig=$2 keep="" e dep
    while read -r dep; do
        case "$dep" in "$build"/*) [ -e "$(dirname "$f")/$(basename "$dep")" ] || {
            cp -L "$dep" "$(dirname "$f")/"; queue+=("$(dirname "$f")/$(basename "$dep")|$dep"); } ;; esac
    done < <(ldd "$orig" 2>/dev/null | awk '$2 == "=>" && $3 ~ /^\// {print $3}')
    for e in $(readelf -d "$orig" | sed -n 's/.*R\(UN\)\{0,1\}PATH.*\[\(.*\)\]/\2/p' | tr ':' ' '); do
        case "$e" in '$ORIGIN'*|"$build"*|"$HOME"*|'') ;; *) keep="$keep:$e" ;; esac
    done
    patchelf --set-rpath "\$ORIGIN$keep" "$f"
}
queue=()
while IFS= read -r -d '' f; do
    is_elf "$f" && queue+=("$f|$build/${f#"$stage"/}")
done < <(find "$stage" -mindepth 2 -type f -print0)
while [ ${#queue[@]} -gt 0 ]; do
    item=${queue[0]}; queue=("${queue[@]:1}")
    fix "${item%%|*}" "${item#*|}"
done

if [ -n "$xrt" ] && readelf -d "$stage/1bit" | grep -q libxrt; then
    mkdir -p "$stage/xrt/lib"
    cp -a "$xrt"/lib/libxrt_coreutil.so* "$xrt"/lib/libxrt_core.so* "$stage/xrt/lib/" 2>/dev/null || true
    cp -a "$xrt"/lib/libxrt_driver_xdna.so* "$stage/xrt/lib/" 2>/dev/null || true
    while IFS= read -r -d '' f; do is_elf "$f" && patchelf --set-rpath '$ORIGIN' "$f"; done \
        < <(find "$stage/xrt/lib" -type f -print0)
    patchelf --set-rpath '$ORIGIN/xrt/lib' "$stage/1bit"
    echo "  xrt/lib: $(ls "$stage/xrt/lib" | wc -l) files"
else
    patchelf --remove-rpath "$stage/1bit"
fi

# nothing may still point into the build machine's home
bad=$(find "$stage" -type f -print0 | while IFS= read -r -d '' f; do
    is_elf "$f" && readelf -d "$f" | grep -E 'R(UN)?PATH' | grep -E "$HOME|$build" | sed "s|^|$f: |"; done || true)
[ -z "$bad" ] || { echo "RPATH into the build machine left:"; echo "$bad"; exit 1; }

cp "$src/LICENSE" "$src/NOTICE" "$stage/"
cat > "$stage/README.txt" <<EOF
1bit engine $ver for Linux x86_64 (Strix Halo / Ryzen AI): https://1bit.gg/

  ./1bit serve -m model.gguf --device vulkan --port 8000

Keep the directory layout: 1bit finds its backends beside itself.
HRX and the ROCm backends need TheRock in /opt/rocm-therock; the NPU needs the amdxdna
kernel driver. Docs: https://1bit.gg/  Code: https://github.com/1bit-MONSTER/engine
EOF
tar -C "$(dirname "$stage")" -cf - "$name" | zstd -q -T0 -19 -o "$out/$name.tar.zst"
ls -la "$out/$name.tar.zst"
