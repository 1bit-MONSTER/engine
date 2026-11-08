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
# pack-check.sh: the release rows that look at a PACKED tree rather than a build tree. Sourced by
# scripts/weekly.sh beside scripts/pack-elf.sh, and by tests/pack_check_test.sh, which is how these rows run
# in CI without a package build:
#
#   build_path_strings    <unpacked tree>  no packed ELF names a build-machine path          (engine #1461)
#   runtime_selfcontained <unpacked tree>  nothing resolves outside the payload, and the HSA runtime set
#                                          that libhsa opens at run time ships beside it     (engine #1411)

# What the engine tarball must and must not hold (docs/releases.md, "Private components"):
# It is packed by scripts/package-linux.sh and folded into the Lemonade + onebit package as 1bit/;
#   runtime-deps  no program or library needs or names an external accelerator-runtime library
#             (EXT_RUNTIME_LIBS below: readelf NEEDED, ldd, strings), and no such library is packed;
#             the NPU runtime is compiled into 1bit
#   kernels   the HRX ggml library carries the private HIP kernels (it exports the add-on's
#             ggml_hrx_hip_addon_register), compiled in as code objects
#   no source no source file (.hip .c .cc .cpp .cxx .h .hpp .inc .cu .cmake) and no file named
#             like one of the private checkouts' files -- except the three GGUF-on-NPU helper scripts
#             (scripts/repack_gguf.py, scripts/check_repack_config.py, tools/gguf_to_q4nx.py), which
#             are the engine's own Apache-2.0 source, published in this repository, and without
#             which a packed install cannot serve a GGUF on the NPU
#   licensed  the closed binaries' LICENSE-BINARY and NOTICE-BINARY are packed beside LICENSE and NOTICE
#   npu       (with WEEKLY_NPU_KERNELS) every file under npu/kernels but layout.json is sealed, never a
#             plaintext ELF; every ELF in the tarball is for the host (x86-64); the packed 1bit has no
#             symbol table and its strings name no private source file and no build-machine home
# Checks the build's 1bit and every file in the tarball.
# engine #1461: no packed ELF may name a build-machine path. One row, used for BOTH packages - the engine
# tarball and the Lemonade package that carries it - over every ELF rather than the packed `1bit` alone: that
# narrow scope is why 146 build paths shipped in c32's payload (lemond 126, lemonade 13, libloomc.so.0 7)
# while the row printed "no build-machine paths", exactly as #1460's `/home/`-only predicate had been too
# narrow a pattern. OCCURRENCES are counted, not lines: a strings run glues paths to their neighbours, so one
# of lemonade's 13 does not begin a line and `grep -c` reports 12 for a file carrying 13 (measured
# 2026-11-08 on c32's own bytes). /home/ covers any builder's home; the source tree, the workdir and the
# build tree cover a build that ran elsewhere on this machine.
# A GENERATED_TEXT file is compiler output that records a path, and the set is bounded and explained because the files in
# it do NOT all have the same remedy - that distinction is the reason this comment exists:
#
#   *.s, *.mod.c, Module.symvers, modules.order   what a PREFIX MAP can fix. `gcc -S` writes `.file "<abs path>"`, and the
#                                                 map rewrites those. Measured 2026-11-09 (engine#1487): two builds of
#                                                 linux-headers-7.4.0-rc1-1bit produced devicetable-offsets.s with the SAME
#                                                 1,182 lines, one 62,895 B and one 49,425 B - the 13,470 B difference is
#                                                 exactly 449 lines x 30 B of absolute paths. Reproducible given the path.
#   *.cmd, *.d                                    what a prefix map CANNOT fix, by design: a .cmd records the compiler's
#                                                 argv as executed and a .d records dependency paths, both absolute on purpose.
#                                                 A red here means a BUILD TREE was packaged, whose remedy is to stop packaging
#                                                 it - not to map it. Written down because the day this guard meets a real
#                                                 kernel build tree it will red on files the map cannot touch, and the next
#                                                 reader must not conclude the guard is wrong.
#
# NOT every text file: a licence, a NOTICE or a man page may legitimately name a path, and a guard that reds those is a
# guard people turn off.
gen_text() {
    case ${1##*/} in
        *.s|*.cmd|*.mod.c|Module.symvers|modules.order|*.d) return 0 ;;
    esac
    return 1
}
build_path_strings() {  # build_path_strings <unpacked tree>  -> one FAIL line per offending file, rc 1
    local t=$1 f n pats rc=0 elfs=0 text=0
    [ -d "$t" ] || { printf 'FAIL no such tree: %s\n' "${t:-<empty>}"; return 1; }
    pats=$(printf '%s\n' /home/ "${SRC:-/nonexistent-build-src}" "${W:-/nonexistent-build-w}" \
                         "${BUILD:-/nonexistent-build-build}")
    while IFS= read -r -d '' f; do
        if pack_is_elf "$f"; then
            elfs=$((elfs + 1))
            n=$(strings -a "$f" | grep -o -F -f <(printf '%s\n' "$pats") | wc -l || true)
        elif gen_text "$f"; then
            # text, so grep reads it directly: `strings` would add nothing and costs a pipe per file
            text=$((text + 1))
            n=$(grep -o -F -f <(printf '%s\n' "$pats") "$f" 2> /dev/null | wc -l || true)
        else
            continue
        fi
        [ "$n" = 0 ] || {
            # the sample shows the PATH for a text file and the matching run for a binary: on a .s the pattern match alone
            # reads "/home/ /home/", which names nothing and looks like a bug in the guard rather than in the artifact
            if pack_is_elf "$f"; then
                sample=$(strings -a "$f" 2> /dev/null | grep -o -F -f <(printf '%s\n' "$pats") | head -2 | tr '\n' ' ')
            else
                sample=$(grep -m2 -F -f <(printf '%s\n' "$pats") "$f" 2> /dev/null | sed 's/^[[:space:]]*//' | cut -c1-110 | tr '\n' ' ')
            fi
            printf 'FAIL %s names a build-machine path (%s occurrences): %s\n' "$(basename "$f")" "$n" "$sample"
            rc=1
        }
    done < <(find "$t" -type f -print0)
    [ "$rc" = 0 ] && echo "ok   no packed ELF or generated text file names a build-machine path (${elfs} ELF, ${text} generated text; /home/, the source tree, the workdir, the build tree)"
    return $rc
}

# engine #1411: a packed ELF must resolve its libraries inside the payload or from the system loader, and it
# must carry no absolute RUNPATH at all. ldd alone does not decide this: the HSA runtime is dlopened by
# name, so a payload can look clean to ldd while the running process maps a builder's toolchain - measured
# on the c22 install, where /proc/<llama-server>/maps held 64 entries under /work-c22 that arrive through
# RUNPATH entries naming a toolchain nobody would keep. Both halves are checked: the statically declared
# paths (readelf) and what the loader actually does (ldd), plus the runtime set the dlopen needs present.
runtime_selfcontained() {  # runtime_selfcontained <unpacked tree>  -> FAIL lines, rc 1
    local t=$1 tcanon f n abs rc=0
    [ -d "$t" ] || { printf 'FAIL no such tree: %s\n' "${t:-<empty>}"; return 1; }
    tcanon=$(readlink -f "$t" 2>/dev/null || printf '%s' "$t")
    while IFS= read -r -d '' f; do
        pack_is_elf "$f" || continue
        abs=$(readelf -d "$f" 2>/dev/null | grep -E '\((RUNPATH|RPATH)\)' | tr ':[]' '\n\n\n' |
                  grep -E '^/' | head -3 | tr '\n' ' ')
        [ -z "$abs" ] || {
            printf 'FAIL %s declares an absolute RUNPATH/RPATH entry: %s\n' "$(basename "$f")" "$abs"
            rc=1
        }
        # what the loader resolves, with nothing inherited: each resolved path is either in the payload or a
        # system directory. A library it cannot find at all is fine here (a package may depend on the distro
        # for libc etc.); one it finds outside both is the leak.
        while IFS= read -r so; do
            # ldd prints the path it was given, unnormalised ("<tree>/bin/../../outside/libx.so"), so the
            # prefix test below has to run on the real path or a library that leaves the payload through a
            # relative rpath reads as "inside" it (found by tests/pack_check_test.sh's escaping arm)
            so=$(readlink -f "$so" 2>/dev/null || printf '%s' "$so")
            case $so in
                /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*|"$tcanon"/*) ;;
                *) printf 'FAIL %s resolves %s, which is neither in the payload nor a system directory\n' \
                          "$(basename "$f")" "$so"; rc=1 ;;
            esac
        done < <(LD_LIBRARY_PATH='' ldd "$f" 2>/dev/null | awk '/=> \// {print $3}' | sort -u)
    done < <(find "$t" -type f -print0)
    # and the set the dlopen needs has to be there, not merely resolvable from somewhere else
    local hsa; hsa=$(find "$t" -type d -path '*/hrx/llama/bin' | head -1)
    if [ -n "$hsa" ]; then
        for n in libhsa-runtime64.so.1 libhsa-amd-aqlprofile64.so.1 librocprofiler-register.so.0; do
            [ -f "$hsa/$n" ] || { printf 'FAIL the HSA runtime set ships without %s (libhsa opens it at run time; engine #1411)\n' "$n"; rc=1; }
        done
        n=$(find "$t" -path '*/hrx/llama/bin/librocm_sysdeps_*.so*' -type f | wc -l)
        [ "$n" -ge 8 ] || { printf 'FAIL the HSA runtime set ships %s rocm_sysdeps entries, fewer than the 8 the runtime needs\n' "$n"; rc=1; }
    fi
    [ "$rc" = 0 ] && echo "ok   every packed ELF resolves inside the payload or the system loader, declares no absolute RUNPATH, and the HSA runtime set it dlopens ships beside it"
    return $rc
}

# engine #1543: the FOURTH leak class - a build-machine path delivered as the filesystem LAYOUT rather than as
# a string inside a file. The rows above read packed ELFs, so neither a directory named after the builder nor
# a symlink pointing into their home is visible to them; the mini image shipped
# /bin/1bit -> /home/<user>/.../build/1bit and a whole /home/<user> tree carrying a 4.7 MB binary.
layout_build_paths() {  # layout_build_paths <tree> [pattern ...]   (default pattern: /home/)
    local t=$1 rc=0 f p l pat
    [ -d "$t" ] || { printf 'FAIL no such tree: %s\n' "${t:-<empty>}"; return 1; }
    shift
    [ $# -gt 0 ] || set -- /home/
    while IFS= read -r -d '' f; do
        p=${f#"$t"}
        for pat in "$@"; do
            case $p in *"$pat"*) printf 'FAIL %s is a path naming a build machine\n' "$p"; rc=1 ;; esac
        done
        # a symlink's TARGET is not file content: it has to be read, not searched
        if [ -L "$f" ]; then
            l=$(readlink "$f")
            for pat in "$@"; do
                case $l in *"$pat"*) printf 'FAIL %s -> %s: the target names a build machine\n' "$p" "$l"; rc=1 ;; esac
            done
        fi
    done < <(find "$t" -print0)
    [ "$rc" = 0 ] && echo "ok   no path in the layout (name or symlink target) names a build machine ($*)"
    return $rc
}
