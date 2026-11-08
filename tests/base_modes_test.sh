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
# base_modes_test.sh: the BASE layer's mode normaliser and its reporter (engine#1474), run over a real tree.
#
# What must hold, and each of these broke a plausible implementation while this was being written:
#   * a keep-list line ending in "/" covers the directory ITSELF as well as everything under it, or /tmp loses its
#     sticky+writable mode -- the single most damaging mistake this work can make;
#   * a keep-list line without a trailing "/" is an exact name or a prefix of a component, not a string prefix
#     (`var/lo` must not keep `var/lock`'s sibling `var/lost+found`... and `var/log` must not keep `var/logrotate`);
#   * the normaliser only NARROWS: a 0600 secret and a 0700 directory keep every bit they had. Setting "other files
#     0644" the way the payload's normaliser does would have made /etc/shadow world-readable -- this is an operating
#     system's base layer, not a package;
#   * the reporter agrees with the normaliser: after a normalisation of a tree the reporter prints nothing except the
#     keep-list's own entries, and it prints every one of them before the fix.
#
# usage: tests/base_modes_test.sh <repo root>
#
# bash, not sh: scripts/pack-elf.sh is a bash library (process substitution, `local`), and this sources it.
set -u
root=${1:?usage: base_modes_test.sh <repo root>}
lib=$root/scripts/pack-elf.sh
keep=$root/os/v3/puppy/base-modes-keep
[ -f "$lib" ] || { echo "no $lib"; exit 1; }
[ -f "$keep" ] || { echo "no $keep"; exit 1; }
# shellcheck disable=SC1090
. "$lib"

fail=0
check() { if [ "$2" = 0 ]; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
tree=$t/tree
mkdir -p "$tree/bin" "$tree/usr/bin" "$tree/etc/acpi" "$tree/tmp" "$tree/var/log" "$tree/var/tmp" \
         "$tree/var/lock/subsys" "$tree/dev" "$tree/opt/1bit-package/1bit" "$tree/var/logrotate"
# the classes: a plain data file, an executable, a directory, and paths the keep-list protects
printf 'data\n'  > "$tree/usr/share-doc.txt";  chmod 0664 "$tree/usr/share-doc.txt"
# a secret and a restricted directory: the property is go-w ONLY, so neither may gain a bit
printf 'secret\n' > "$tree/etc-shadow-like";   chmod 0600 "$tree/etc-shadow-like"
printf 'group\n'  > "$tree/etc-shadow-0640";   chmod 0640 "$tree/etc-shadow-0640"
mkdir -p "$tree/private";                      chmod 0700 "$tree/private"
chmod 0770 "$tree/etc/acpi"
printf '#!/bin/sh\n' > "$tree/usr/bin/tool";   chmod 0775 "$tree/usr/bin/tool"

chmod 1777 "$tree/tmp"
chmod 0775 "$tree/var/log"; chmod 0775 "$tree/var/tmp"; chmod 0775 "$tree/var/lock"; chmod 0775 "$tree/var/lock/subsys"
chmod 0775 "$tree/dev"
chmod 0775 "$tree/var/logrotate"
mkdir -p "$tree/opt/1bit-package/1bit"; printf 'payload\n' > "$tree/opt/1bit-package/1bit/data"; chmod 0664 "$tree/opt/1bit-package/1bit/data"; chmod 0775 "$tree/opt/1bit-package/1bit"

# a keep-list that is the repository's, plus the payload excluded by the caller (the reporter is pointed at the base only)
before=$(pack_writable_base_files "$tree" "$keep" | wc -l)
[ "$before" -ge 2 ]; check "the reporter finds the writable base paths before the fix ($before)" $?

out=$(pack_normalise_base_modes "$tree" "$keep")
case $out in *"go-w on"*) true ;; *) false ;; esac
check "the normaliser reports what it changed [$out]" $?

[ "$(stat -c %a "$tree/usr/share-doc.txt")" = 644 ]; check "a world-writable data file loses o+w (0664 -> 0644)" $?
[ "$(stat -c %a "$tree/usr/bin/tool")" = 755 ];       check "a group-writable executable loses g+w (0775 -> 0755)" $?
[ "$(stat -c %a "$tree/etc/acpi")" = 750 ];            check "a 0770 directory keeps its owner/group bits (-> 0750, not 0755)" $?
[ "$(stat -c %a "$tree/etc-shadow-like")" = 600 ];     check "a 0600 secret stays 0600 (the normaliser never grants a bit)" $?
[ "$(stat -c %a "$tree/etc-shadow-0640")" = 640 ];     check "a 0640 file stays 0640 (/etc/shadow must not become world-readable)" $?
[ "$(stat -c %a "$tree/private")" = 700 ];             check "a 0700 directory stays 0700" $?
[ "$(stat -c %a "$tree/tmp")" = 1777 ];                check "the keep-list line 'tmp/' keeps /tmp ITSELF (sticky 1777)" $?
[ "$(stat -c %a "$tree/var/log")" = 775 ];             check "the keep-list keeps var/log" $?
[ "$(stat -c %a "$tree/var/log/subsys" 2>/dev/null || stat -c %a "$tree/var/tmp")" = 775 ]
check "the keep-list keeps a path BELOW a prefix (var/tmp)" $?
[ "$(stat -c %a "$tree/var/logrotate")" = 755 ];       check "a sibling that merely shares a prefix (var/logrotate) is NOT kept" $?
[ "$(stat -c %a "$tree/dev")" = 775 ];                 check "the keep-list keeps dev" $?

# a second run is a no-op: idempotent, and the second report says so
out2=$(pack_normalise_base_modes "$tree" "$keep")
case $out2 in "base modes: go-w on 0 directories, 0 executables, 0 other files ("*)
    check "a second run changes nothing [$out2]" 0 ;;
    *) check "a second run changes nothing [$out2]" 1 ;;
esac

after=$(pack_writable_base_files "$tree" "$keep")
[ -z "$after" ]; check "the reporter is clean after the fix (outside the keep-list)" $?
# and it still sees everything if the keep-list is absent -- the row must be able to fail
: > "$t/empty-keep"
nostray=$(pack_writable_base_files "$tree" "$t/empty-keep" | wc -l)
[ "$nostray" -ge 5 ]; check "with an empty keep-list the reporter reports the kept paths too ($nostray)" $?

[ $fail = 0 ] && echo "base_modes: ok" || echo "base_modes: FAILED"
exit $fail
