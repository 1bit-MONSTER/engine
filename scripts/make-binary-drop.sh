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
# make-binary-drop.sh <engine checkout> <out dir>
#
# Packs the public binary drop for one engine pin (docs/releases.md, "The public binary drop"): the
# compiled proprietary parts an outside source build cannot compile for itself, and nothing else.
# The drop carries binaries only; no source of the private components, and nothing named like it.
#
#   hrx/llama/bin/        the HRX llama-server, llama-bench and every library the release package
#                         carries beside them (scripts/package-linux.sh), libggml-hrx.so with the
#                         private HIP kernels compiled in
#   npu/libnpu_rt_api.a   the NPU runtime behind the engine's public npu/npu_rt_api.h
#   npu/include/          npu_rt_api.h, the engine's own public header
#   MANIFEST              every pin, the build host, the compilers, sha256 of every file
#   LICENSE-BINARY NOTICE-BINARY LICENSE NOTICE
#
# Where the pieces come from, in order of preference:
#   hrx   --build <dir>          a release build's hrx/llama/bin (WEEKLY_DRY_RUN output is fine)
#         --from-package <tar>   an already packed engine release (1bit-<version>-linux-x86_64.tar.zst)
#   npu   --npu-lib <file>       a prebuilt libnpu_rt_api.a
#         --npu-rt <dir>         the private runtime's sources, built here with its build-api-lib.sh
#
# Then it runs the same checks the weekly package runs (scripts/weekly.sh package_checks): runtime deps
# self-contained, the HIP kernels really are compiled in, no source files, nothing named like the private
# sources, and the two binary licenses are present. The drop is written as
# <out>/binary-drop-<version>.tar.zst plus <out>/SHA256SUMS, and the script prints the tarball path.
# It publishes nothing.
set -euo pipefail

me=make-binary-drop
usage() {
    echo "usage: $me <engine checkout> <out dir> [--build <dir>] [--from-package <tar.zst>]" >&2
    echo "          [--npu-lib <libnpu_rt_api.a>] [--npu-rt <private npu_rt dir>] [--version <v>]" >&2
    exit 2
}
die() { echo "$me: $*" >&2; exit 1; }
# shellcheck source=scripts/pack-elf.sh
. "$(cd "$(dirname "$0")" && pwd)/pack-elf.sh"

src=${1:-}; out=${2:-}
[ -n "$src" ] && [ -n "$out" ] || usage; shift 2
build="" package="" npu_lib="" npu_rt="" version=""
while [ $# -gt 0 ]; do
    case "$1" in
        --build) build=${2:?}; shift 2 ;;
        --from-package) package=${2:?}; shift 2 ;;
        --npu-lib) npu_lib=${2:?}; shift 2 ;;
        --npu-rt) npu_rt=${2:?}; shift 2 ;;
        --version) version=${2:?}; shift 2 ;;
        -h|--help) usage ;;
        *) die "unknown argument '$1'" ;;
    esac
done
src=$(cd "$src" && pwd)
mkdir -p "$out"
out=$(cd "$out" && pwd)

for f in LICENSE-BINARY NOTICE-BINARY LICENSE NOTICE config/npu-runtime.config config/gpu-kernels.config \
         scripts/package-linux.sh; do
    [ -f "$src/$f" ] || die "$src has no $f: not an engine checkout"
done

val() { sed -n "s/^$1=//p" "$2" | tail -1; }
npu_pin=$(val sha "$src/config/npu-runtime.config")
gpu_pin=$(val sha "$src/config/gpu-kernels.config")
engine_sha=$(git -C "$src" rev-parse HEAD 2>/dev/null) || die "$src is not a git checkout (need the engine pin)"
[ -n "$version" ] || version=$(git -C "$src" rev-parse --short HEAD)

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
stage=$work/binary-drop-$version
mkdir -p "$stage/hrx/llama/bin" "$stage/npu/include"

# ── the HRX tree: exactly what package-linux.sh puts in the release's hrx/llama/bin ──────────────
hrx_src="" hrx_from=""
if [ -n "$build" ]; then
    [ -d "$build/hrx/llama/bin" ] || die "$build has no hrx/llama/bin (build with ONEBIT_HRX=ON and ONEBIT_GPU_PRIVATE)"
    hrx_src=$build/hrx/llama/bin; hrx_from="build $build"
elif [ -n "$package" ]; then
    [ -f "$package" ] || die "no such engine package: $package"
    mkdir -p "$work/pkg"
    tar -C "$work/pkg" -xf "$package" --use-compress-program=unzstd
    hrx_src=$(find "$work/pkg" -path '*/hrx/llama/bin' -type d)
    hrx_src=${hrx_src%%$'\n'*}
    [ -n "$hrx_src" ] || die "$package carries no hrx/llama/bin"
    hrx_from="package $(basename "$package")"
else
    die "no HRX source: pass --build <release build dir> or --from-package <engine release tarball>"
fi
for f in llama-server; do [ -f "$hrx_src/$f" ] || die "$hrx_src has no $f"; done
[ -f "$hrx_src/libggml-hrx.so" ] || die "$hrx_src has no libggml-hrx.so"
# The release's rule, not a new one: the programs and every shared library beside them.
cp -a "$hrx_src/llama-server" "$stage/hrx/llama/bin/"
for f in "$hrx_src"/*.so "$hrx_src"/*.so.*; do [ -e "$f" ] && cp -a "$f" "$stage/hrx/llama/bin/"; done
[ -f "$hrx_src/llama-bench" ] && cp -a "$hrx_src/llama-bench" "$stage/hrx/llama/bin/"
# A packed release is already self-contained. A build tree is not: run package-linux.sh's own
# steps on it (scripts/pack-elf.sh): the libraries it loads from elsewhere in the build tree and
# LLVM's OpenMP runtime go beside it, RUNPATH becomes $ORIGIN, and the symbol tables come out.
if [ -n "$build" ]; then
    build=$(cd "$build" && pwd)
    pack_localize "$stage" "$build"
    pack_strip "$stage/hrx"
fi
echo "  hrx/llama/bin: $(ls "$stage/hrx/llama/bin" | wc -l) files from $hrx_from"

# ── the NPU runtime library and its public header ────────────────────────────────────────────────
if [ -z "$npu_lib" ] && [ -n "$npu_rt" ]; then
    b=$npu_rt/build-api-lib.sh
    [ -f "$b" ] || die "$npu_rt has no build-api-lib.sh (not the private npu_rt directory)"
    npu_lib=$(bash "$b" "$src" "$work/npurt-build")
elif [ -z "$npu_lib" ] && [ -n "${BINARY_DROP_NPU_RT:-}" ]; then
    npu_lib=$(bash "$BINARY_DROP_NPU_RT/build-api-lib.sh" "$src" "$work/npurt-build")
fi
[ -n "$npu_lib" ] && [ -f "$npu_lib" ] || die "no NPU runtime library: pass --npu-lib <libnpu_rt_api.a> or --npu-rt <dir>"
cp -a "$npu_lib" "$stage/npu/libnpu_rt_api.a"
header=$src/npu/npu_rt_api.h
[ -f "$header" ] || die "$src has no npu/npu_rt_api.h: the engine pin predates the public NPU header"
cp -a "$header" "$stage/npu/include/npu_rt_api.h"

cp -a "$src/LICENSE" "$src/NOTICE" "$src/LICENSE-BINARY" "$src/NOTICE-BINARY" "$stage/"

# ── checks: the same class the weekly package runs, against this layout ──────────────────────────
bad=0
is_elf() { pack_is_elf "$1"; }
fail() { echo "FAIL $*"; bad=1; }

# every ELF is a compiled binary with no source-name leakage
while IFS= read -r -d '' f; do
    is_elf "$f" || continue
    n=$(readelf -d "$f" 2>/dev/null | grep -Ec 'NEEDED.*(libxrt|libopenvino|libonnxruntime)' || true); [ "$n" = 0 ] || fail "$f needs an external accelerator runtime"
    n=$(ldd "$f" 2>/dev/null | grep -Ec 'libxrt|libopenvino|libonnxruntime' || true);           [ "$n" = 0 ] || fail "$f: ldd resolves an external accelerator runtime"
    n=$(strings -a "$f" | grep -Ec 'libxrt|libopenvino|libonnxruntime' || true);                [ "$n" = 0 ] || fail "$f names an external accelerator runtime"
    n=$(readelf -d "$f" 2>/dev/null | grep -E 'R(UN)?PATH' | grep -cE "$HOME|/home/|$work" || true)
    [ "$n" = 0 ] || fail "$f has an RPATH into a build machine home"
done < <(find "$stage" -type f -print0)
[ "$(find "$stage" \( -name 'libxrt*' -o -name 'libopenvino*' -o -name 'libonnxruntime*' \) | wc -l)" = 0 ] || fail "the drop carries an external accelerator runtime"

# every library the programs load is in the drop (or is the TheRock runtime HRX documents)
missing=$(pack_missing_libs "$stage")
[ -z "$missing" ] || fail "libraries ldd cannot find: $(echo "$missing" | tr '\n' ';')"

# the private HIP kernels really are compiled into the HRX ggml library
n=$(nm -D --defined-only "$stage/hrx/llama/bin/libggml-hrx.so" | grep -c ggml_hrx_hip_addon_register || true)
[ "$n" != 0 ] || fail "libggml-hrx.so does not export ggml_hrx_hip_addon_register"
n=$(strings -a "$stage/hrx/llama/bin/libggml-hrx.so" | grep -c amdgcn-amd-amdhsa || true)
[ "$n" != 0 ] || fail "libggml-hrx.so carries no HIP code objects"

# the NPU library exposes the public API and nothing that names the private sources
n=$(nm -g --defined-only "$stage/npu/libnpu_rt_api.a" 2>/dev/null | grep -c ' T ' || true)
[ "$n" != 0 ] || fail "libnpu_rt_api.a defines no global symbols"
n=$(nm -g --defined-only "$stage/npu/libnpu_rt_api.a" 2>/dev/null | awk '$2=="T"{print $3}' | grep -cvE '^_ZN[K]*10npu_rt_api' || true)
[ "$n" = 0 ] || fail "libnpu_rt_api.a exports $n symbols outside the npu_rt_api namespace"
n=$(strings -a "$stage/npu/libnpu_rt_api.a" | grep -cE '/home/|npu_rt\.cpp' || true)
[ "$n" = 0 ] || fail "libnpu_rt_api.a names a private source path"

# no source files, in any tree
n=$(find "$stage" -type f \( -name '*.hip' -o -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
        -o -name '*.h' -o -name '*.hpp' -o -name '*.inc' -o -name '*.cu' -o -name '*.py' -o -name '*.cmake' \) \
        -not -path '*/npu/include/npu_rt_api.h' | wc -l)
[ "$n" = 0 ] || fail "source files in the drop"

# nothing named like the private sources: the pins' require= names, plus every tracked file name
# when a private checkout happens to be at hand (the same rule weekly.sh applies)
names=$(for pin in npu-runtime gpu-kernels; do
            sed -n 's/^require=//p' "$src/config/$pin.config" | tr ' ' '\n'
        done | grep -v '^$' | xargs -r -n1 basename | LC_ALL=C sort -u)
n=$(find "$stage" -type f -printf '%f\n' | LC_ALL=C sort -u | LC_ALL=C comm -12 - <(echo "$names") | wc -l)
[ "$n" = 0 ] || fail "$n files named like the private sources"

for f in LICENSE-BINARY NOTICE-BINARY LICENSE NOTICE; do
    [ -f "$stage/$f" ] || fail "$f is not in the drop"
done

# ── MANIFEST: every pin, the host, the compilers, sha256 of every file ───────────────────────────
llama_pin=$(git -C "$src" ls-tree HEAD third_party/llama.cpp 2>/dev/null | awk '{print $3}')
hrx_pin=$(git -C "$src" ls-tree HEAD third_party/hrx-system 2>/dev/null | awk '{print $3}')
{
    echo "# The public binary drop for engine $engine_sha (docs/releases.md)."
    echo "# Binaries under the 1bit Binary License (LICENSE-BINARY); third-party code in NOTICE-BINARY."
    echo "# Build with -DONEBIT_BINARY_DROP=<this directory or its tarball URL> (config/binary-drop.config)."
    echo "drop_version=$version"
    echo "engine_sha=$engine_sha"
    echo "llama_cpp_pin=$llama_pin"
    echo "hrx_system_pin=$hrx_pin"
    echo "gpu_kernels_pin=$gpu_pin"
    echo "npu_runtime_pin=$npu_pin"
    echo "npu_runtime_lib_source=${npu_rt:-$npu_lib}"
    echo "build_host=$(uname -srm)"
    echo "build_compiler_cc=$(${CC:-cc} --version 2>/dev/null | head -1)"
    echo "build_compiler_cxx=$(${CXX:-c++} --version 2>/dev/null | head -1)"
    echo "hrx_build=$hrx_from"
    echo "# sha256  size  path"
    ( cd "$stage" && find . -type f ! -name MANIFEST -printf '%P\n' | LC_ALL=C sort | \
        while IFS= read -r f; do echo "$(sha256sum "$f" | cut -d' ' -f1)  $(stat -c%s "$f")  $f"; done )
} > "$stage/MANIFEST"

echo "  MANIFEST: $(grep -cE '^[0-9a-f]{64}  ' "$stage/MANIFEST") files, pins engine=${engine_sha:0:7} llama=${llama_pin:0:7} hrx=${hrx_pin:0:7} gpu=${gpu_pin:0:7} npu=${npu_pin:0:7}"
[ -f "$stage/MANIFEST" ] || fail "MANIFEST was not written"
[ $bad = 0 ] && echo "  checks: ok" || die "checks failed, nothing written"

# ── the tarball and its SHA256SUMS ──────────────────────────────────────────────────────────────
name=binary-drop-$version
# Rebuild cleanly: the target is this script's own output, and zstd refuses an existing one.
rm -f "$out/$name.tar.zst" "$out/SHA256SUMS"
tar -C "$work" -cf - "$name" | zstd -q -T4 -19 -o "$out/$name.tar.zst"
( cd "$out" && sha256sum "$name.tar.zst" > SHA256SUMS )
echo "  $out/$name.tar.zst ($(stat -c%s "$out/$name.tar.zst") B)"
echo "  sha256 $(cut -d' ' -f1 "$out/SHA256SUMS")"
echo "drop $version written: $(grep -cE '^[0-9a-f]{64}  ' "$stage/MANIFEST") files. Nothing published."
echo "$out/$name.tar.zst"
