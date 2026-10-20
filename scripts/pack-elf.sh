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
# pack-elf.sh: sourced by scripts/package-linux.sh and scripts/make-binary-drop.sh, which pack
# build-tree programs and libraries into a tree that must run once the build tree is gone.
#
#   pack_is_elf <file>              true for an ELF file
#   pack_localize <stage> <build>   every ELF under <stage> (below its top level; <stage>/<rel> was
#                                   copied from <build>/<rel>) finds its libraries beside itself:
#                                   a library it loads from the build tree (HRX's libhrx, libloomc)
#                                   or a toolchain runtime named in PACK_BUNDLE_LIBS (default
#                                   libomp.so*: LLVM's OpenMP runtime, which ggml's CPU code links
#                                   when TheRock's amdclang builds llama.cpp) is copied beside it,
#                                   and its RUNPATH becomes $ORIGIN plus the entries outside the
#                                   build tree and $HOME (TheRock in /opt, which HRX needs anyway)
#   pack_strip <dir>                strips symbol tables and debug info from every ELF under <dir>
#   pack_build_rpaths <dir> <build> prints each ELF under <dir> whose RPATH/RUNPATH still points
#                                   into <build> or $HOME
#   pack_missing_libs <dir>         prints "<file>: <lib> ..." for each ELF under <dir> whose
#                                   libraries ldd cannot find (ROCR_VISIBLE_DEVICES=-1: nothing
#                                   touches a GPU), except PACK_EXTERNAL_LIBS, the TheRock runtime
#                                   the package documents as a requirement (HIP, hipBLAS(Lt),
#                                   rocBLAS, HSA: DwarfStar's ds4 links them)

PACK_BUNDLE_LIBS=${PACK_BUNDLE_LIBS:-'libomp.so libomp.so.*'}
PACK_EXTERNAL_LIBS=${PACK_EXTERNAL_LIBS:-'libamdhip64.so* libhipblas.so* libhipblaslt.so* librocblas.so* libhsa-runtime64.so* librocm_smi64.so* librocsolver.so* libroctx64.so*'}

pack_is_elf() { [ -f "$1" ] && [ "$(head -c 4 "$1" | od -An -c | tr -d ' ')" = '177ELF' ]; }

_pack_matches() {  # <name> <patterns>
    local p
    for p in $2; do
        # shellcheck disable=SC2254
        case "$1" in $p) return 0 ;; esac
    done
    return 1
}

pack_localize() {
    local stage=$1 build=$2 item f orig keep e dep dest
    local queue=()
    while IFS= read -r -d '' f; do
        pack_is_elf "$f" && queue+=("$f|$build/${f#"$stage"/}")
    done < <(find "$stage" -mindepth 2 -type f -print0)
    while [ ${#queue[@]} -gt 0 ]; do
        item=${queue[0]}; queue=("${queue[@]:1}")
        f=${item%%|*} orig=${item#*|} keep=""
        while read -r dep; do
            dest=$(dirname "$f")/$(basename "$dep")
            [ -e "$dest" ] && continue
            case "$dep" in
                "$build"/*) ;;
                *) _pack_matches "$(basename "$dep")" "$PACK_BUNDLE_LIBS" || continue ;;
            esac
            cp -L "$dep" "$dest"
            queue+=("$dest|$dep")
        done < <(ldd "$orig" 2>/dev/null | awk '$2 == "=>" && $3 ~ /^\// {print $3}')
        for e in $(readelf -d "$orig" | sed -n 's/.*R\(UN\)\{0,1\}PATH.*\[\(.*\)\]/\2/p' | tr ':' ' '); do
            case "$e" in '$ORIGIN'*|"$build"*|"$HOME"*|'') ;; *) keep="$keep:$e" ;; esac
        done
        patchelf --set-rpath "\$ORIGIN$keep" "$f"
    done
}

pack_strip() {
    local f
    while IFS= read -r -d '' f; do
        if pack_is_elf "$f"; then strip --strip-unneeded "$f"; fi
    done < <(find "$1" -type f -print0)
}

pack_build_rpaths() {
    local f
    find "$1" -type f -print0 | while IFS= read -r -d '' f; do
        pack_is_elf "$f" && readelf -d "$f" | grep -E 'R(UN)?PATH' | grep -E "$HOME|$2" | sed "s|^|$f: |"
    done || true
}

pack_missing_libs() {
    local f lib miss
    while IFS= read -r -d '' f; do
        pack_is_elf "$f" || continue
        miss=""
        for lib in $(ROCR_VISIBLE_DEVICES=-1 ldd "$f" 2>/dev/null | awk '$2 == "=>" && $3 == "not" {print $1}' | sort -u); do
            _pack_matches "$lib" "$PACK_EXTERNAL_LIBS" || miss="$miss $lib"
        done
        [ -z "$miss" ] || echo "${f#"$1"/}:$miss"
    done < <(find "$1" -type f -print0)
}
