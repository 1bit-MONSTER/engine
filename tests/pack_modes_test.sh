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
# pack_modes_test.sh: the package's file modes (scripts/pack-elf.sh, pack_normalise_modes -- engine#1352).
#
# c31 and c32 both shipped 134 of 195 files under opt/1bit-package group- or other-writable, because the packer used
# `cp -a` and therefore inherited whatever mode the build tree had.  This asserts the two properties that fix has to
# have, on a fixture stage tree, with no build and no tar:
#
#   1. every regular file ends up 0755 (was executable) or 0644 (was not), and no directory is group/other-writable;
#   2. a file that appears group/other-writable AFTER normalisation is REPORTED, so a staging line added below the
#      normalisation cannot regress the property silently.
#
# The fixture deliberately includes the shapes the real package has: a 0775 launcher, 0775 libraries with a symlink
# set (libx.so.0 -> libx.so.0.1.0), 0664 licence files, a 0666 file, a setgid file and a 0775 directory.  Symlinks are
# checked to still BE symlinks afterwards: `chmod -R` or `install -m` would have replaced or followed them, which is
# why the normaliser only touches `-type f` and `-type d`.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../scripts/pack-elf.sh
. "$here/../scripts/pack-elf.sh"

fails=0
check() { if [ "$2" = "$3" ]; then printf '  ok   %s = %s\n' "$1" "$3"; else printf '  FAIL %s = %s (want %s)\n' "$1" "$2" "$3"; fails=$((fails+1)); fi; }

stage=$(mktemp -d)/pkg
mkdir -p "$stage/hrx/llama/bin" "$stage/lemonade"
printf 'launcher\n' > "$stage/1bit"
printf 'lib\n' > "$stage/hrx/llama/bin/libx.so.0.1.0"
ln -s libx.so.0.1.0 "$stage/hrx/llama/bin/libx.so.0"
printf 'tool\n' > "$stage/hrx/llama/bin/llama-server"
printf 'lic\n' > "$stage/LICENSE"
printf 'notice\n' > "$stage/NOTICE"
printf 'conf\n' > "$stage/lemonade/1bit-hrx.conf"
printf 'odd\n' > "$stage/odd-0666"
printf 'sgid\n' > "$stage/odd-setgid"

# the modes the release actually shipped: 0775 executables and libraries, 0664 licences, and worse
chmod 0775 "$stage/1bit" "$stage/hrx/llama/bin/llama-server" "$stage/hrx/llama/bin/libx.so.0.1.0"
chmod 0664 "$stage/LICENSE" "$stage/NOTICE" "$stage/lemonade/1bit-hrx.conf"
chmod 0666 "$stage/odd-0666"
chmod 2664 "$stage/odd-setgid"   # setgid on a DATA file, so the class is unchanged
chmod 0775 "$stage/hrx" "$stage/hrx/llama" "$stage/lemonade"

echo "before: $(find "$stage" -type f \( -perm -0020 -o -perm -0002 \) | wc -l) group/other-writable, $(find "$stage" -type d -perm -0020 | wc -l) writable dirs"

pack_normalise_modes "$stage"

echo "after:"
check "no writable file or directory is left" "$(pack_writable_files "$stage")" ""
check "launcher" "$(stat -c %a "$stage/1bit")" "755"
check "shipped tool" "$(stat -c %a "$stage/hrx/llama/bin/llama-server")" "755"
check "shipped library" "$(stat -c %a "$stage/hrx/llama/bin/libx.so.0.1.0")" "755"
check "licence" "$(stat -c %a "$stage/LICENSE")" "644"
check "notice" "$(stat -c %a "$stage/NOTICE")" "644"
check "data file" "$(stat -c %a "$stage/lemonade/1bit-hrx.conf")" "644"
check "a group/other-writable data file is closed" "$(stat -c %a "$stage/odd-0666")" "644"
check "setgid is cleared" "$(stat -c %a "$stage/odd-setgid")" "644"
check "directory" "$(stat -c %a "$stage/hrx")" "755"
check "the symlink set is still links" "$(test -L "$stage/hrx/llama/bin/libx.so.0" && echo link || echo file)" "link"
check "and still points where it did" "$(readlink "$stage/hrx/llama/bin/libx.so.0")" "libx.so.0.1.0"
check "no executable bit was invented" "$(find "$stage" -type f -perm -u+x | wc -l)" "3"

# The check is a REPORTER, so it must still fire on a tree nobody normalised -- that is the property that makes it
# worth having. Two shapes: a fixture as the packer receives it, and a file staged after the normalisation ran.
pristine=$stage/../pristine
mkdir -p "$pristine"; chmod 0755 "$pristine"   # so the fixture reports its files, not its own umask
printf 'x\n' > "$pristine/a.json"; chmod 0664 "$pristine/a.json"
printf 'y\n' > "$pristine/b"; chmod 0775 "$pristine/b"
check "an unnormalised tree reports both files" "$(pack_writable_files "$pristine" | cut -d' ' -f1 | sort | tr '\n' ',')" "a.json,b,"

printf 'late\n' > "$stage/staged-after.json"; chmod 0664 "$stage/staged-after.json"
chmod 0777 "$pristine"
check "a writable tree root is reported as '.'" "$(pack_writable_files "$pristine" | grep -c '^\. 777')" "1"
chmod 0755 "$pristine"

check "a file staged below the normalisation is caught" "$(pack_writable_files "$stage" | cut -d' ' -f1)" "staged-after.json"
check "and reports the mode a user would see" "$(pack_writable_files "$stage" | cut -d' ' -f2)" "664"

# The rule has to hold at EVERY packing site, not at the one being fixed. This is the check whose absence let the gap
# reopen: #1361 fixed package-linux.sh, #1360 fixed weekly.sh's lemonade fold, and make-binary-drop.sh -- a third
# script that sources this same library and writes a PUBLISHED tarball -- was missed by both until it was looked for.
# Enumerated from the tree rather than listed here, so a fourth site cannot be forgotten either.
#
# Two tiers, because the sites are not the same job:
#   * every site that stages files must call pack_normalise_modes -- no exceptions, it cannot change behaviour;
#   * the sites that pack a FLAT payload must also assert with pack_writable_files.
# pack-distro.sh is the one exemption and it is documented in that script (lines 63-74): it stages on the
# formatter's own copy, so the chmod cannot fail there, and it decided the reporter would be "decoration". This test
# records that decision instead of silently widening the rule; if a second site wants the exemption, someone has to
# edit the list below, which is the point.
flat="package-linux.sh weekly.sh make-binary-drop.sh"
echo "packing sites:"
for f in "$here"/../scripts/*.sh; do
    grep -q 'pack-elf.sh' "$f" || continue      # only scripts that stage files with this library
    grep -qE '\btar -C' "$f" || continue        # and actually write an archive
    b=$(basename "$f")
    printf '  %-28s ' "$b"
    # A real invocation, not a mention: the first version of this check matched a COMMENTED-OUT call and passed on a
    # site with the guard disabled -- the same "a check that cannot fail" defect the fixture half already caught once.
    if ! grep -qE '^[[:space:]]*pack_normalise_modes' "$f"; then
        echo "MISSING pack_normalise_modes"; fails=$((fails+1)); continue
    fi
    case " $flat " in
        *" $b "*)
            if grep -qE '^[[:space:]]*[a-z_]+=\$\(pack_writable_files' "$f"; then
                echo "normalises and asserts"
            else
                echo "MISSING pack_writable_files (a flat payload must assert)"; fails=$((fails+1))
            fi ;;
        *)  echo "normalises (flat-payload assertion not required: see its own comment)" ;;
    esac
done
# A floor, not an exact count: a fifth packing site that does carry the guard is a good change, and failing on it
# would teach the next person to edit the test instead of the packer. The per-site loop above is the real assertion.
sites=$(for f in "$here"/../scripts/*.sh; do grep -q 'pack-elf.sh' "$f" && grep -qE '\btar -C' "$f" && echo x; done | wc -l)
check "the enumeration found every known packing site (>= 3)" "$([ "$sites" -ge 3 ] && echo yes || echo no)" "yes"

rm -rf "$(dirname "$stage")"
if [ "$fails" -ne 0 ]; then printf '%d check(s) failed\n' "$fails"; exit 1; fi
echo "pack_modes: ok"
