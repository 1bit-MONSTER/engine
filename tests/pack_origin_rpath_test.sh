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
# pack_origin_rpath_test.sh - scripts/pack-elf.sh's pack_origin_rpath (engine#3287). TheRock's HSA runtime set is
# linked with RPATHs into its own pip tree ($ORIGIN/../../_rocm_sdk_core/lib/llvm/lib, ...); staged as built, the
# payload failed the release check's payload-runpath row on all 11 files. Asserts, on compiled fixtures:
#   1. a library with a TheRock-shaped RPATH, and one with a RUNPATH, end up with $ORIGIN alone, and still load
#      their sibling through it;
#   2. a library already at $ORIGIN, and one with no RPATH at all, are left byte-for-byte unchanged;
#   3. symlinks and non-ELF files are not touched, and nothing below the directory is visited;
#   4. package-linux.sh calls it on the staged set, after the set is staged and before the set's ldd check.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../scripts/pack-elf.sh
. "$here/../scripts/pack-elf.sh"
cc=$(command -v cc || command -v gcc || command -v clang || true)
[ -n "$cc" ] || { echo "pack_origin_rpath_test: skip (no C compiler)"; exit 77; }
for t in patchelf readelf; do command -v "$t" > /dev/null || { echo "pack_origin_rpath_test: skip (no $t)"; exit 77; }; done
w=$(mktemp -d); trap 'rm -rf "$w"' EXIT
fails=0
ok() { echo "  ok   $*"; }
no() { echo "  FAIL $*"; fails=$((fails + 1)); }
rpath_of() { readelf -d "$1" 2>/dev/null | sed -n 's/.*(R\(UN\)\{0,1\}PATH).*path: \[\(.*\)\]/\2/p' | head -1; }

d=$w/bin; mkdir -p "$d/sub"
echo 'int dep(void){return 7;}' > "$w/dep.c"
echo 'int dep(void); int top(void){return dep();}' > "$w/top.c"
"$cc" -shared -fPIC -o "$d/libdep.so.1" "$w/dep.c" -Wl,-soname,libdep.so.1 || exit 2
# TheRock's shape: a DT_RPATH (not RUNPATH) of $ORIGIN-relative dirs that do not exist in the payload
"$cc" -shared -fPIC -o "$d/libtherock.so.1" "$w/top.c" -L"$d" -l:libdep.so.1 \
    -Wl,--disable-new-dtags,-rpath,'$ORIGIN/../../_rocm_sdk_core/lib/llvm/lib:$ORIGIN/rocm_sysdeps/lib' || exit 2
"$cc" -shared -fPIC -o "$d/librunpath.so.1" "$w/top.c" -L"$d" -l:libdep.so.1 -Wl,--enable-new-dtags,-rpath,/opt/elsewhere || exit 2
"$cc" -shared -fPIC -o "$d/liborigin.so.1" "$w/top.c" -L"$d" -l:libdep.so.1 -Wl,--enable-new-dtags,-rpath,'$ORIGIN' || exit 2
cp "$d/libtherock.so.1" "$d/sub/libdeep.so.1"
ln -s libtherock.so.1 "$d/libtherock.so"
printf 'not an elf\n' > "$d/README"
before_origin=$(sha256sum "$d/liborigin.so.1" | cut -c1-64)
before_dep=$(sha256sum "$d/libdep.so.1" | cut -c1-64)
before_deep=$(sha256sum "$d/sub/libdeep.so.1" | cut -c1-64)

n=$(pack_origin_rpath "$d") || no "pack_origin_rpath returned non-zero"
[ "$n" = 2 ] && ok "two libraries rewritten" || no "rewrote $n libraries, want 2 (libtherock, librunpath)"
for f in libtherock.so.1 librunpath.so.1; do
    r=$(rpath_of "$d/$f"); [ "$r" = '$ORIGIN' ] && ok "$f: RPATH is \$ORIGIN" || no "$f: RPATH is '$r'"
    # loads its sibling through $ORIGIN alone, with nothing else on the search path
    if env -u LD_LIBRARY_PATH ldd "$d/$f" 2>/dev/null | grep -q "libdep.so.1 => $d/libdep.so.1"; then ok "$f resolves libdep beside itself"
    else no "$f does not resolve libdep.so.1 beside itself"; fi
done
[ "$(sha256sum "$d/liborigin.so.1" | cut -c1-64)" = "$before_origin" ] && ok "a library already at \$ORIGIN is untouched" ||
    no "a library already at \$ORIGIN was rewritten"
[ "$(sha256sum "$d/libdep.so.1" | cut -c1-64)" = "$before_dep" ] && ok "a library without an RPATH is untouched" ||
    no "a library without an RPATH was rewritten"
[ "$(sha256sum "$d/sub/libdeep.so.1" | cut -c1-64)" = "$before_deep" ] && ok "a subdirectory is not visited" ||
    no "a library in a subdirectory was rewritten"
[ -L "$d/libtherock.so" ] && ok "the symlink is still a symlink" || no "the symlink was replaced"
[ "$(cat "$d/README")" = "not an elf" ] && ok "a non-ELF file is untouched" || no "a non-ELF file changed"
n2=$(pack_origin_rpath "$d"); [ "$n2" = 0 ] && ok "a second run changes nothing" || no "a second run rewrote $n2"

# package-linux.sh: called on the staged set, between staging it and the set's own ldd check
pl=$here/../scripts/package-linux.sh
a=$(grep -n 'install -m 0644 "$so" "$stage/hrx/llama/bin' "$pl" | tail -1 | cut -d: -f1)
b=$(grep -n 'pack_origin_rpath "$stage/hrx/llama/bin"' "$pl" | head -1 | cut -d: -f1)
c=$(grep -n '^ *missing_set=' "$pl" | head -1 | cut -d: -f1)
if [ -n "$a" ] && [ -n "$b" ] && [ -n "$c" ] && [ "$a" -lt "$b" ] && [ "$b" -lt "$c" ]; then
    ok "package-linux.sh sets the staged HSA set's RUNPATH after staging it and before checking it"
else no "package-linux.sh does not call pack_origin_rpath between staging the HSA set and its ldd check (lines $a/$b/$c)"; fi

[ "$fails" = 0 ] && echo "pack_origin_rpath_test: PASS" || echo "pack_origin_rpath_test: $fails failure(s)"
[ "$fails" = 0 ]
