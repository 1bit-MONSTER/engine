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
# patch-sfs-selftest.sh - the manifest mode of patch-sfs.sh, proved on a two-megabyte SFS instead of on a 1.5 GB one.
#
# `patch-sfs.sh` is the tool every surgical cycle cut goes through, and its manifest mode is what lets one cycle carry a
# payload tree, a new file and a handful of scripts in ONE repack. That mode is new code, so it gets a fixture: a small
# tree packed with mksquashfs, patched, unpacked again and inspected - never the real image, which is 10 minutes a run.
#
# Covered: a single-file cut (mode inherited, a setuid file elsewhere untouched), a manifest cut (a new path, a changed
# file and a whole directory with one file removed and one added), and the three refusals that matter - bytes already in
# the SFS, a manifest row with no source, and an SFS that is not an SFS. NOT covered: the undeclared-difference refusal
# (an input cannot produce it - every difference this tool makes comes from a manifest row) and the uid >= 1000 refusal
# (the tool chowns what it copies, so it needs a hand-built SFS to trip; the real cut checks it on the real image).
#
# Run as root (the tool refuses to run otherwise, and setuid bits are part of what it proves).
# Exit: 0 every case behaved, 1 one did not.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
tool=$here/patch-sfs.sh
[ "$(id -u)" = 0 ] || { echo "selftest: run as root" >&2; exit 1; }
for t in mksquashfs unsquashfs sha256sum; do command -v "$t" >/dev/null || { echo "selftest: no $t" >&2; exit 1; }; done

# /var/tmp and not /tmp (tmpfs,nodev on this box) nor $HOME (which is /root under sudo): the fixtures are unpacked
# squashfs trees and the tool refuses a tmpfs work dir for the same reason.
base=${SFS_WORK:-/var/tmp/patch-sfs-selftest}
mkdir -p "$base"
work=$(mktemp -d "$base/run.XXXXXX")
case $(df --output=fstype "$work" | tail -1) in tmpfs|ramfs) echo "selftest: $work is tmpfs; set SFS_WORK" >&2; exit 1;; esac
trap 'rm -rf "$work"' EXIT
pass=0; fail=0
ok()   { pass=$((pass+1)); echo "PASS $*"; }
bad()  { fail=$((fail+1)); echo "FAIL $*"; }
is()   { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: got [$2] want [$3]"; fi; }

# --- the fixture: a tree shaped like the parts of the image a cycle touches -------------------------------------------
src=$work/src
mkdir -p "$src/etc/init.d" "$src/usr/local/bin" "$src/opt/1bit-package/bin" "$src/bin"
printf 'old release check\n' > "$src/usr/local/bin/1bit-release-check"
printf 'old lemonade service\n' > "$src/etc/init.d/10lemonade"; chmod 0750 "$src/etc/init.d/10lemonade"
printf 'old installer\n' > "$src/usr/local/bin/1bit-install"
printf 'lemond v1\n' > "$src/opt/1bit-package/bin/lemond"
printf 'kept\n' > "$src/opt/1bit-package/bin/retired-tomorrow"
printf '#!/bin/sh\n' > "$src/bin/su"; chmod 04755 "$src/bin/su"
mksquashfs "$src" "$work/base.sfs" -noappend -no-progress -comp gzip >/dev/null 2>&1 || { echo "selftest: fixture pack failed" >&2; exit 1; }
base_unpack=$work/base-unpack
unsquashfs -q -n -d "$base_unpack" "$work/base.sfs" >/dev/null 2>&1 || { echo "selftest: fixture unpack failed" >&2; exit 1; }

# --- case 1: single file, the c28 shape (mode inherited from the shipped file, setuid elsewhere untouched) -----------
printf 'new release check\n' > "$work/new-rc"
"$tool" "$work/base.sfs" usr/local/bin/1bit-release-check "$work/new-rc" "$work/one.sfs" >"$work/one.log" 2>&1
is "single file: rc" "$?" "0"
if [ -s "$work/one.sfs" ]; then
    u=$work/one-unpack; unsquashfs -q -n -d "$u" "$work/one.sfs" >/dev/null 2>&1
    is "single file: bytes" "$(cat "$u/usr/local/bin/1bit-release-check")" "new release check"
    is "single file: mode" "$(stat -c%a "$u/usr/local/bin/1bit-release-check")" "644"
    is "single file: setuid elsewhere untouched" "$(stat -c%a "$u/bin/su")" "4755"
    is "single file: untouched file" "$(cat "$u/usr/local/bin/1bit-install")" "old installer"
else
    bad "single file: no output"; sed -n 1,10p "$work/one.log"
fi

# --- case 2: the manifest shape - new path, changed file, and a directory that gains and loses a file ----------------
repl=$work/repl; mkdir -p "$repl/payload/bin"
printf 'new selftest\n' > "$work/new-selftest"
printf 'new lemonade service\n' > "$work/new-lemonade"
printf 'lemond v2\n' > "$repl/payload/bin/lemond"
printf 'brand new backend\n' > "$repl/payload/bin/new-backend"
printf 'usr/local/bin/1bit-selftest\t%s\t0755\nopt/1bit-package\t%s\netc/init.d/10lemonade\t%s\n' \
    "$work/new-selftest" "$repl/payload" "$work/new-lemonade" > "$work/manifest"
"$tool" --manifest "$work/manifest" "$work/base.sfs" "$work/two.sfs" >"$work/two.log" 2>&1
is "manifest: rc" "$?" "0"
if [ -s "$work/two.sfs" ]; then
    u=$work/two-unpack; unsquashfs -q -n -d "$u" "$work/two.sfs" >/dev/null 2>&1
    is "manifest: new path added" "$(cat "$u/usr/local/bin/1bit-selftest" 2>/dev/null)" "new selftest"
    is "manifest: new path owner" "$(stat -c%u:%g "$u/usr/local/bin/1bit-selftest" 2>/dev/null)" "0:0"
    # 0755 from the manifest, NOT the source file's mode: the repo keeps selftest.sh 0775 and the build installs it 0755,
    # so an addition that inherited the source mode shipped a group-writable executable and no log said so.
    is "manifest: added file mode" "$(stat -c%a "$u/usr/local/bin/1bit-selftest" 2>/dev/null)" "755"
    is "manifest: changed file" "$(cat "$u/etc/init.d/10lemonade")" "new lemonade service"
    is "manifest: changed file keeps its mode" "$(stat -c%a "$u/etc/init.d/10lemonade")" "750"
    is "manifest: directory replaced" "$(cat "$u/opt/1bit-package/bin/lemond")" "lemond v2"
    is "manifest: directory gained a file" "$(cat "$u/opt/1bit-package/bin/new-backend" 2>/dev/null)" "brand new backend"
    is "manifest: directory lost a file" "$(ls "$u/opt/1bit-package/bin/retired-tomorrow" 2>/dev/null)" ""
    is "manifest: untouched by manifest" "$(cat "$u/usr/local/bin/1bit-install")" "old installer"
    is "manifest: setuid elsewhere untouched" "$(stat -c%a "$u/bin/su")" "4755"
    is "manifest: no uid >= 1000 in the pack" \
        "$(unsquashfs -lln "$work/two.sfs" 2>/dev/null | awk '{ split($2,a,"/"); if (a[1]+0>=1000 || a[2]+0>=1000) print }' | wc -l)" "0"
else
    bad "manifest: no output"; sed -n 1,10p "$work/two.log"
fi

# --- case 3: refusals. Each must fail AND leave no output file ------------------------------------------------------
refuse() { # <label> <expected-text> <args...>
    local label=$1 want=$2; shift 2
    local log=$work/refuse.log
    rm -f "$log"
    "$@" >"$log" 2>&1
    local rc=$?
    if [ "$rc" = 0 ]; then bad "$label: exited 0"; return; fi
    if ! grep -qF -- "$want" "$log"; then bad "$label: refused without saying '$want'"; sed -n 1,6p "$log"; return; fi
    ok "$label (rc=$rc, said '$want')"
}
# the same bytes the SFS already carries: a manifest row that would ship nothing
"$tool" "$work/base.sfs" usr/local/bin/1bit-install "$base_unpack/usr/local/bin/1bit-install" "$work/dup.sfs" >"$work/dup.log" 2>&1
if [ $? = 0 ]; then bad "identical bytes: exited 0"; else ok "identical bytes: refused"; fi
[ -e "$work/dup.sfs" ] && bad "identical bytes: left an output file behind" || ok "identical bytes: no output file"
# a manifest row with no source
printf 'usr/local/bin/1bit-selftest\n' > "$work/badman"
refuse "manifest row with no source" "no source" "$tool" --manifest "$work/badman" "$work/base.sfs" "$work/bad.sfs"
# a manifest mode that is not octal
printf 'usr/local/bin/1bit-selftest\t%s\t07x5\n' "$work/new-selftest" > "$work/badmode"
refuse "manifest mode is not octal" "is not octal" "$tool" --manifest "$work/badmode" "$work/base.sfs" "$work/bad3.sfs"
# not an SFS
printf 'not a filesystem\n' > "$work/bogus.sfs"
refuse "source is not an SFS" "unsquashfs failed" "$tool" --manifest "$work/manifest" "$work/bogus.sfs" "$work/bad2.sfs"

echo "selftest: $pass passed, $fail failed"
[ "$fail" = 0 ]
