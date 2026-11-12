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
#   pack_normalise_base_modes <tree> <keeplist>
#                                   the BASE layer's modes: dirs 0755, executables 0755, other files 0644, keeping the
#                                   paths in <keeplist> (engine#1474). Prints what it changed, by class
#   pack_writable_base_files <tree> <keeplist>
#                                   the reporter for the same thing: group/other-writable paths outside the keep-list
#   pack_build_rpaths <dir> <build> prints each ELF under <dir> whose RPATH/RUNPATH still points
#                                   into <build> or $HOME
#   pack_missing_libs <dir>         prints "<file>: <lib> ..." for each ELF under <dir> whose
#                                   libraries ldd cannot find (ROCR_VISIBLE_DEVICES=-1: nothing
#                                   touches a GPU), except PACK_EXTERNAL_LIBS, the TheRock runtime
#                                   the package documents as a requirement (HIP, hipBLAS(Lt),
#                                   rocBLAS, HSA: the HIP llama-server links them)
#   pack_localize_links <dir>       every symlink under <dir> that is absolute or resolves outside
#                                   <dir> becomes a copy of what it points to; links inside the tree
#                                   (libggml.so -> libggml.so.0) stay links. Lemonade's build links
#                                   its web app's KaTeX fonts to the build machine's fonts-katex
#                                   (/usr/share/fonts/truetype/katex), which a package or 1bit OS
#                                   does not have (2026-10-21 rehearsal). Fails on a dangling one
#   pack_outside_links <dir>        prints "<link> -> <target>" for each symlink under <dir> that is
#                                   absolute, resolves outside <dir>, or dangles

PACK_BUNDLE_LIBS=${PACK_BUNDLE_LIBS:-'libomp.so libomp.so.*'}
PACK_EXTERNAL_LIBS=${PACK_EXTERNAL_LIBS:-'libamdhip64.so* libhipblas.so* libhipblaslt.so* librocblas.so* libhsa-runtime64.so* librocm_smi64.so* librocsolver.so* libroctx64.so*'}

pack_is_elf() { [ -f "$1" ] && [ "$(head -c 4 "$1" | od -An -c | tr -d ' ')" = '177ELF' ]; }
# only an ELF with a .dynamic section can carry an RPATH/RUNPATH; a static ELF (an AIE kernel
# object, an intermediate link) must be left alone rather than fed to patchelf
pack_is_dynelf() { pack_is_elf "$1" && readelf -S "$1" 2>/dev/null | grep -c '\.dynamic' | grep -qv '^0$'; }

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
        pack_is_dynelf "$f" && queue+=("$f|$build/${f#"$stage"/}")
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

# A program or library ships only if every library it NEEDs resolves the way it will on a user's machine:
# plain ldd, with **no** LD_LIBRARY_PATH, so only the ELF's own RUNPATH and the system are consulted. A
# library path inherited from the build shell resolves against the build tree and hides exactly the defect
# this check exists for. Prints "<file>: <lib>" for each NEEDED that cannot be found.
pack_unresolved_needed() {
    local stage=$1 f
    while IFS= read -r -d '' f; do
        pack_is_dynelf "$f" || continue
        env -u LD_LIBRARY_PATH ROCR_VISIBLE_DEVICES=-1 ldd "$f" 2>/dev/null |
            awk -v f="$f" '$2 == "=>" && $3 == "not" {print f": "$1}'
    done < <(find "$stage" -type f -print0)
}

# Everything in the package has to come from the build being packaged. A binary copied in from another tree
# (a hardware-gate cache, a drop) carries that tree's RUNPATH and its build's missing parts: in the 2026-10-23
# rc, `llama-perplexity` came from a hardware-gate build and pointed at that build's directory. Compares each
# staged file with the same relative path under the build and reports what has no counterpart there.
# The programs `put` was asked for are the ones the release promises: if one of them cannot resolve, that is a
# build problem (the tool came from another tree, or its implementation library was not built) and the package
# has to fail rather than quietly lose the tool. Call once per directory group: pack_require_tools <stage>
# <rel dir> <program ...>.
pack_require_tools() {
    local stage=$1 rel=$2 f bad=0
    shift 2
    for f in "$@"; do
        if [ ! -f "$stage/$rel/$f" ]; then
            echo "  missing from the package: $rel/$f"
            bad=1
            continue
        fi
        while IFS= read -r line; do
            echo "  $rel/$f cannot load: ${line##*: }"
            bad=1
        done < <(pack_unresolved_needed "$stage" | grep -F "$stage/$rel/$f:")
    done
    return $bad
}

# Everything in the package has to come from the build being packaged. A binary copied in from another tree
# (a hardware-gate cache, a drop) carries that tree's RUNPATH and its build's missing parts: in the 2026-10-23
# rc, `llama-perplexity` came from a hardware-gate build and pointed at that build's directory. Compares each
# staged file with the same relative path under the build and reports what has no counterpart there.
pack_require_provenance() {  # <stage> <build>
    local stage=$1 build=$2 f rel bad=0
    while IFS= read -r -d '' f; do
        pack_is_elf "$f" || continue
        rel=${f#"$stage"/}
        if [ ! -f "$build/$rel" ]; then
            echo "  $rel is not in the build being packaged"
            bad=1
        elif ! cmp -s "$f" "$build/$rel"; then
            echo "  $rel differs from the build being packaged"
            bad=1
        fi
    done < <(find "$stage" -type f -print0)
    return $bad
}

# The programs `put` was asked for are the ones the release promises: if one of them cannot resolve, that is a
# build problem (the tool came from another tree, or its implementation library was not built) and the package
# has to fail rather than quietly lose the tool. Prints the offending programs.
# A thin binary whose implementation library was not built has to leave the package rather than ship broken
# (the 2026-10-23 rc shipped llama-perplexity without libllama-perplexity-impl.so). Removing the binary is
# the safe direction: the tool is then absent instead of failing at first use.
pack_prune_unloadable() {
    local stage=$1 bad dropped=0
    while IFS= read -r line; do
        bad=${line%%:*}
        [ -f "$bad" ] || continue
        echo "  drop $(echo "$bad" | sed "s|^$stage/||"): needs a library the package does not carry"
        rm -f "$bad"
        dropped=$((dropped+1))
    done < <(pack_unresolved_needed "$stage")
    [ "$dropped" = 0 ] || echo "  $dropped program(s) dropped as unloadable"
}

pack_strip() {
    local f
    while IFS= read -r -d '' f; do
        if pack_is_elf "$f"; then
            strip --strip-unneeded "$f"
            # .comment names the compiler and its version; it is not needed to load or link
            objcopy --remove-section=.comment --remove-section=.note.gnu.gold-version "$f" 2>/dev/null || true
        fi
    done < <(find "$1" -type f -print0)
}

# Everything a shipped program and library may not carry: a compiler identification section, a path
# into a build machine, or an absolute RUNPATH. Prints one line per violation, empty when clean.
pack_artifact_hygiene() {
    local f out=""
    while IFS= read -r -d '' f; do
        pack_is_elf "$f" || continue
        readelf -p .comment "$f" 2>/dev/null | grep -qE "clang version|GCC:" && out="$out
$f: .comment names the compiler"
        strings -a "$f" 2>/dev/null | grep -qE "^/(home|Users|root)/[A-Za-z0-9_./-]+" &&
            out="$out
$f: embedded build-machine path"
        # An absolute RUNPATH is only acceptable under the toolchain root: some shipped programs link the
        # SDK's own libraries (whose closure is far larger than the package), and the release needs that
        # toolchain to run anyway.
        readelf -d "$f" 2>/dev/null | grep -E 'R(UN)?PATH' |
            sed -n 's/.*Library \(runpath\|rpath\): \[\(.*\)\]/\2/p' | tr ':' '\n' |
            grep -E "^/" | grep -vE "^(${ONEBIT_TOOLCHAIN_ROOT:-/opt/rocm-therock})/" | grep -q . &&
            out="$out\n$f: absolute RUNPATH outside the toolchain root"
    done < <(find "$1" -type f -print0)
    # every shipped program and library must resolve what it links against
    while IFS= read -r line; do out="$out\n$line: a NEEDED library is not in the package or on the machine"; done \
        < <(pack_unresolved_needed "$1")
    [ -z "$out" ] || printf "%b\n" "$out"
}

# Every regular file in the package is group- or other-writable (prints "<path> <mode>", empty when clean).
#
# c31 and c32 both shipped 134 of the 195 files under opt/1bit-package writable this way (engine#1352): the packer used
# `cp -a`, so every mode came from the build tree or the builder's umask.  Nothing was exploitable -- every file is
# root:root and no appliance user is in group root -- but a 0775 root-owned binary in a public tarball reads as
# deliberate to anyone auditing it.
#
# A reporter, not a fixer, on purpose: a function that normalises and then checks its own work can never fail, so the
# regression it is supposed to catch would have to be invisible.  `pack_normalise_modes` below fixes; this one is what
# the package asserts with afterwards, and what can be pointed at an unpacked tarball by anything that did not pack it.
# Directories are included -- a world-writable directory in an installed tree is the worse of the two.
pack_writable_files() {
    local tree=$1 f rel
    while IFS= read -r -d '' f; do
        rel=${f#"$tree"/}
        [ "$rel" = "$f" ] && rel=.     # the tree root itself, which find yields without a trailing slash
        printf '%s %s\n' "$rel" "$(stat -c %a "$f")"
    done < <(find "$tree" \( -type f -o -type d \) \( -perm -0020 -o -perm -0002 \) -print0)
}

# Give every staged file a mode the packer decided, rather than the one it was built with.  The executable bit as
# staged decides the class, so behaviour cannot change: what was executable stays executable (0755) and everything else
# is data (0644).  It also clears setuid/setgid/sticky, which `cp -a` was faithfully carrying in from the build tree.
# Symlinks are skipped on purpose: their mode is not used by the kernel, and chmod would follow the link to whatever it
# points at -- and `install -m` cannot be used for this at all, because it dereferences and would replace the package's
# `lib*.so.0 -> lib*.so.0.x.y` sets with a second copy of each library.
pack_normalise_modes() {
    local stage=$1
    find "$stage" -type d -exec chmod 0755 {} + 2>/dev/null || true
    find "$stage" -type f -perm -u+x -exec chmod 0755 {} + 2>/dev/null || true
    find "$stage" -type f ! -perm -u+x -exec chmod 0644 {} + 2>/dev/null || true
}

# ── the BASE layer's modes (engine#1474) ─────────────────────────────────────────────────────────────────────────────
# The payload's modes are settled (pack_normalise_modes above): c33 ships 0 group/other-writable files under
# opt/1bit-package. The layer UNDER it had 2,176 such paths measured on the c33 base SFS, 1,814 of them regular files --
# every one of them inherited from woof-CE's build with the builder's umask, carried faithfully by `cp -a`, and none of
# them relied on by anything: the appliance runs as root, and root does not need group-write. Eight classes of path DO
# need it, and they are the ones in os/v3/puppy/base-modes-keep.
#
# These two functions are the fix and its reporter, in the same split as pack_normalise_modes/pack_writable_files: a
# function that normalises and then checks its own work can never fail.

# _pack_keep_list <keeplist file>: the keep-list as a newline-joined string, comments and whitespace gone.
# Read ONCE per run: the first version of the matcher re-read the file for every path in the tree, which is 60,000 file
# reads on a real base layer and took longer than the rest of the cycle put together.
_pack_keep_list() {
    sed -e 's/#.*//' -e 's/[[:space:]]//g' "$1" 2> /dev/null | grep -v '^$' || true
}

# _pack_keep_match <relative path> <keep-list string>: 0 when the path is a keep-list entry or below one.
# An entry covers the path ITSELF as well as everything under it -- `tmp/` keeps /tmp's mode 1777, not just `tmp/...` --
# and matching is per COMPONENT (`var/log` does not keep `var/logrotate`).
_pack_keep_match() {
    local rel=$1 pfx
    while IFS= read -r pfx; do
        [ -n "$pfx" ] || continue
        pfx=${pfx%/}
        case $rel in "$pfx"|"$pfx"/*) return 0 ;; esac
    done <<< "$2"
    return 1
}

# _pack_chmod_batch <mode> <paths...>: chmod in chunks, because a tree this size overruns the argument list.
_pack_chmod_batch() {
    local mode=$1 batch=() n=0 p
    shift
    for p in "$@"; do
        batch+=("$p")
        n=$((n + 1))
        if [ $n -ge 400 ]; then chmod "$mode" "${batch[@]}" 2> /dev/null; batch=(); n=0; fi
    done
    [ $n -gt 0 ] && chmod "$mode" "${batch[@]}" 2> /dev/null
    return 0
}

# pack_normalise_base_modes <tree> <keeplist>: drop the GROUP AND OTHER WRITE BITS on every path in <tree> outside the
# keep-list, and change nothing else.
#
# The first version set "dirs 0755, executables 0755, other files 0644" the way the payload's normaliser does, and that
# was wrong for a whole operating system: it TAKES BITS AWAY FROM NOBODY but GRANTS them -- /etc/shadow and /etc/gshadow
# (0640) become 0644, every 0600 private key becomes readable, and a 0700 directory becomes traversable. The payload has no
# secrets, which is why the blunt rule is fine there; the base layer has all of them. `go-w` is the property engine#1474
# is actually about (2,176 group/other-writable paths), it can only narrow, and the reporter below checks exactly it.
#
# Symlinks are skipped: their mode is not used by the kernel, and chmod would follow the link.
pack_normalise_base_modes() {
    local tree=$1 list=$2 keep m rel
    [ -d "$tree" ] || { echo "pack_normalise_base_modes: no tree $tree" >&2; return 2; }
    [ -f "$list" ] || { echo "pack_normalise_base_modes: no keep-list $list" >&2; return 2; }
    keep=$(_pack_keep_list "$list")
    local -a dirs_=() execs_=() files_=()
    local kept=0
    # three passes, each one find + a printf: nothing is stat'ed or forked per path
    while IFS=$'\t' read -r rel; do
        [ -n "$rel" ] || continue
        if _pack_keep_match "$rel" "$keep"; then kept=$((kept + 1)); continue; fi
        dirs_+=("$tree/$rel")
    done < <(find "$tree" -type d \( -perm -0020 -o -perm -0002 \) -printf '%P\n')
    while IFS=$'\t' read -r rel; do
        [ -n "$rel" ] || continue
        if _pack_keep_match "$rel" "$keep"; then kept=$((kept + 1)); continue; fi
        execs_+=("$tree/$rel")
    done < <(find "$tree" -type f -perm -u+x \( -perm -0020 -o -perm -0002 \) -printf '%P\n')
    while IFS=$'\t' read -r rel; do
        [ -n "$rel" ] || continue
        if _pack_keep_match "$rel" "$keep"; then kept=$((kept + 1)); continue; fi
        files_+=("$tree/$rel")
    done < <(find "$tree" -type f ! -perm -u+x \( -perm -0020 -o -perm -0002 \) -printf '%P\n')
    # go-w on the three classes -- BOTH group and other write: `g-w` alone left a 757 (o+w) behind, which the reporter
    # then caught (measured). The counts are what the cycle log shows.
    [ ${#dirs_[@]} -gt 0 ] && _pack_chmod_batch go-w "${dirs_[@]}"
    [ ${#execs_[@]} -gt 0 ] && _pack_chmod_batch go-w "${execs_[@]}"
    [ ${#files_[@]} -gt 0 ] && _pack_chmod_batch go-w "${files_[@]}"
    printf 'base modes: go-w on %s directories, %s executables, %s other files (%s path(s) kept by the keep-list); no other bit changed\n' \
        "${#dirs_[@]}" "${#execs_[@]}" "${#files_[@]}" "$kept"
}

# pack_writable_base_files <tree> <keeplist>: the reporter the release row uses -- group/other-writable paths in the base
# layer OUTSIDE the keep-list, which is what has to be zero. Prints "<path> <mode>" and nothing when clean.
pack_writable_base_files() {
    local tree=$1 list=$2 keep m rel
    keep=$(_pack_keep_list "$list")
    while IFS=$'\t' read -r m rel; do
        [ -n "$rel" ] || continue
        _pack_keep_match "$rel" "$keep" && continue
        printf '%s %s\n' "$rel" "$m"
    done < <(find "$tree" \( -type f -o -type d \) \( -perm -0020 -o -perm -0002 \) -printf '%m\t%P\n')
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

# a symlink is "inside" when its target is relative and resolves to an existing path under the
# tree's top (followed through any chain of links)
_pack_link_inside() {  # <top (physical)> <link>
    local t
    case "$(readlink -- "$2")" in /*) return 1 ;; esac
    t=$(readlink -e -- "$2") || return 1
    case "$t" in "$1"/*) return 0 ;; *) return 1 ;; esac
}

pack_localize_links() {
    local top l tmp bad=0
    top=$(cd "$1" && pwd -P)
    while IFS= read -r -d '' l; do
        _pack_link_inside "$top" "$l" && continue
        if [ ! -e "$l" ]; then echo "pack_localize_links: dangling ${l#"$1"/} -> $(readlink -- "$l")" >&2; bad=1; continue; fi
        tmp="$l.pack-tmp"
        cp -rL --preserve=mode,timestamps -- "$l" "$tmp" && rm -f -- "$l" && mv -- "$tmp" "$l" || bad=1
    done < <(find "$1" -type l -print0)
    return $bad
}

pack_outside_links() {
    local top l
    top=$(cd "$1" && pwd -P)
    while IFS= read -r -d '' l; do
        _pack_link_inside "$top" "$l" || echo "${l#"$1"/} -> $(readlink -- "$l")"
    done < <(find "$1" -type l -print0)
}
