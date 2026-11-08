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
# pack_check_test.sh: the two release rows that look at a PACKED tree (scripts/pack-check.sh), exercised on
# synthetic trees so CI runs them without a package build. Each arm asserts the row's own message, not just
# its exit status, so a row that fails for the wrong reason is caught:
#
#   build_path_strings     #1461  a packed ELF naming a build-machine path fails the release
#   runtime_selfcontained  #1411  a packed ELF that declares an absolute RUNPATH fails it, one that resolves a
#                                 library outside the payload fails it, and a payload that ships the HSA
#                                 runtime without the aqlprofile agent libhsa opens fails it
#
# The clean tree is the control: both rows must pass it, or the arms below prove nothing.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=scripts/pack-elf.sh
. "$here/../scripts/pack-elf.sh"
# shellcheck source=scripts/pack-check.sh
. "$here/../scripts/pack-check.sh"
# the CI build job compiles with gcc-16 and has no bare `cc`, so the compiler is probed rather than assumed;
# a tree with none at all cannot build this repository either, so that is a failure and not a skip
cc=${CC:-}
if [ -z "$cc" ]; then
    for cand in cc gcc gcc-16 gcc-15 clang; do
        command -v "$cand" >/dev/null 2>&1 && { cc=$cand; break; }
    done
fi
[ -n "$cc" ] || { echo "pack_check_test: no C compiler found (tried \$CC, cc, gcc, gcc-16, gcc-15, clang)"; exit 2; }
echo "pack_check_test: building the synthetic trees with $cc"

w=$(mktemp -d); trap 'rm -rf "$w"' EXIT
fail=0
printf 'int f(void) { return 0; }\n' > "$w/x.c"

# a tree shaped like the engine payload: bin/ holds the programs and the runtime set beside them
clean_tree() {  # clean_tree <dir>
    local t=$1 bin=$1/hrx/llama/bin n
    mkdir -p "$bin"
    "$cc" -shared -fPIC -o "$bin/libhrx.so.0" "$w/x.c" -Wl,-rpath,'$ORIGIN' || return 1
    for n in libhsa-runtime64.so.1 libhsa-amd-aqlprofile64.so.1 librocprofiler-register.so.0 \
             librocm_sysdeps_bz2.so librocm_sysdeps_drm.so.2 librocm_sysdeps_drm_amdgpu.so.1 \
             librocm_sysdeps_elf.so.1 librocm_sysdeps_liblzma.so.5 librocm_sysdeps_numa.so.1 \
             librocm_sysdeps_z.so.1 librocm_sysdeps_zstd.so.1; do
        cp "$bin/libhrx.so.0" "$bin/$n"
    done
}

# <label> <expected: pass|fail> <expected message or ""> <tree>
arm() {
    local label=$1 want=$2 phrase=$3 t=$4 out out1 out2 rc=0 rc1=0 rc2=0
    # each row's status separately: a group's status is only its LAST command's, so a failing
    # build_path_strings followed by a passing runtime_selfcontained would read as a pass
    out1=$(build_path_strings "$t" 2>&1) || rc1=1
    out2=$(runtime_selfcontained "$t" 2>&1) || rc2=1
    out="$out1
$out2"; rc=$((rc1 + rc2))
    if [ "$want" = pass ]; then
        if [ "$rc" = 0 ] && ! printf '%s' "$out" | grep -q '^FAIL'; then echo "PASS  $label"
        else echo "FAIL  $label: expected both rows to pass, got rc=$rc"; printf '%s\n' "$out" | head -5 | sed 's/^/      /'; fail=$((fail+1)); fi
    else
        if [ "$rc" != 0 ] && printf '%s' "$out" | grep -q -- "$phrase"; then echo "PASS  $label"
        else echo "FAIL  $label: expected a FAIL naming '$phrase', got rc=$rc"; printf '%s\n' "$out" | head -5 | sed 's/^/      /'; fail=$((fail+1)); fi
    fi
}

# 1. control: a tree that is clean passes both rows
clean_tree "$w/clean" || { echo "pack_check_test: could not build the control tree"; exit 2; }
arm "a clean payload passes both rows" pass "" "$w/clean"

# 2. engine #1411: the HSA runtime set ships without the aqlprofile agent libhsa opens at run time
cp -a "$w/clean" "$w/no-aqlprofile"
rm -f "$w/no-aqlprofile/hrx/llama/bin/libhsa-amd-aqlprofile64.so.1"
arm "#1411: a runtime set without aqlprofile fails" fail "libhsa-amd-aqlprofile64" "$w/no-aqlprofile"

# 3. engine #1411: an absolute RUNPATH entry, which is how c22's install reached a builder's toolchain
cp -a "$w/clean" "$w/abs-runpath"
"$cc" -shared -fPIC -o "$w/abs-runpath/hrx/llama/bin/libabs.so.0" "$w/x.c" -Wl,-rpath,"$w/outside" || exit 2
arm "#1411: an absolute RUNPATH fails" fail "declares an absolute RUNPATH" "$w/abs-runpath"

# 4. engine #1411: a RELATIVE RUNPATH that resolves outside the payload - the case an absolute-RUNPATH row
#    alone would miss, and the one ldd has to answer for
mkdir -p "$w/outside" "$w/rel-runpath"
"$cc" -shared -fPIC -o "$w/outside/libescaped.so.0" "$w/x.c" || exit 2
cp -a "$w/clean/hrx" "$w/rel-runpath/"
# a library in the payload whose NEEDED resolves into a directory outside it, through a RELATIVE rpath:
# the entry itself looks portable, and only what the loader does shows where it lands
"$cc" -shared -fPIC -o "$w/rel-runpath/hrx/llama/bin/libesc.so.0" "$w/x.c" \
      -Wl,--no-as-needed -L"$w/outside" -l:libescaped.so.0 -Wl,-rpath,'$ORIGIN/../../../../outside' || exit 2
readelf -d "$w/rel-runpath/hrx/llama/bin/libesc.so.0" | grep -E 'NEEDED|RUNPATH' | head -3
arm "#1411: a library resolved outside the payload fails" fail "neither in the payload nor a system directory" "$w/rel-runpath"

# 5. engine #1461: a packed ELF naming a build-machine path
cp -a "$w/clean" "$w/buildpath"
printf 'const char *p = "/home/builder/wt/roll/src/app/serve.cpp";\nint g(void) { return p != 0; }\n' > "$w/bp.c"
"$cc" -shared -fPIC -o "$w/buildpath/hrx/llama/bin/libbp.so.0" "$w/bp.c" || exit 2
arm "#1461: a build-machine path in a packed ELF fails" fail "names a build-machine path" "$w/buildpath"

if [ "$fail" = 0 ]; then echo "pack_check: all rows PASS"; else echo "pack_check: $fail row(s) FAILED"; fi
[ "$fail" = 0 ]
