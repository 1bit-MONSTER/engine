#!/bin/bash
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
# prune-kernel-headers-deb.sh IN.deb OUT.deb - drop the one build artefact the headers package ships that nothing
# reads, so a published .deb stops carrying the builder's home path, the build tree name and source line numbers
# (engine#1487).
#
# What it removes, and why it is only that: `scripts/mod/devicetable-offsets.s` is a `gcc -S` by-product of building
# `devicetable-offsets.h`, and the kernel's own rule regenerates it from the shipped `.c` when `scripts` is rebuilt
# (scripts/mod/Makefile: `$(obj)/devicetable-offsets.s: $(obj)/devicetable-offsets.c FORCE`), so dropping it cannot
# break a later `make modules_prepare`. It carried 449 `/home/...` occurrences in linux-headers-7.4.0-rc1-1bit.
#
# What it deliberately does NOT touch, because removing it would break module builds:
#   tools/objtool/objtool, tools/bpf/resolve_btfids/resolve_btfids  - needed (ORC unwinder, BTF ids); they embed our
#   build paths too, and the fix for those is at build time (-ffile-prefix-map / KBUILD_HOSTCFLAGS), not deletion.
#   scripts/rust_is_available_test.py  - the path in it (/home/jd/Documents/dev/kernel-module-flake/...) is upstream's
#   own example text, not ours.
#
# The repack keeps the control members and toolchain as they are and regenerates DEBIAN/md5sums, so the result differs
# from the input by exactly the removed files and their checksum lines.
set -uo pipefail
IN=${1:?usage: prune-kernel-headers-deb.sh IN.deb OUT.deb}; OUT=${2:?usage: prune-kernel-headers-deb.sh IN.deb OUT.deb}
[ -f "$IN" ] || { echo "no such deb: $IN" >&2; exit 2; }
work=$(mktemp -d) || exit 2
trap 'rm -rf "$work"' EXIT
dpkg-deb -R "$IN" "$work/pkg" || { echo "cannot unpack $IN" >&2; exit 3; }
hdrs=$(find "$work/pkg/usr/src" -maxdepth 1 -type d -name 'linux-headers-*' | head -1)
[ -n "$hdrs" ] || { echo "$IN carries no /usr/src/linux-headers-* tree" >&2; exit 4; }
before=$(grep -rlE '/home/[a-z]+/' "$hdrs" 2>/dev/null | wc -l)
removed=0
while IFS= read -r f; do rm -f "$f"; removed=$((removed+1)); done < <(find "$hdrs/scripts/mod" -maxdepth 1 -name '*.s' 2>/dev/null)
[ "$removed" -gt 0 ] || { echo "nothing to prune (no scripts/mod/*.s) - refusing to rewrite the package" >&2; exit 5; }
rm -f "$work/pkg/DEBIAN/md5sums"
( cd "$work/pkg" && find . -path ./DEBIAN -prune -o -type f -print0 | xargs -0 md5sum > DEBIAN/md5sums )
after=$(grep -rlE '/home/[a-z]+/' "$hdrs" 2>/dev/null | wc -l)
dpkg-deb --root-owner-group -b "$work/pkg" "$OUT" >/dev/null || { echo "repack failed" >&2; exit 6; }
echo "pruned $removed file(s) from $(basename "$hdrs")"
echo "files carrying a build-machine path: $before -> $after"
grep -rlE '/home/[a-z]+/' "$hdrs" 2>/dev/null | sed "s|$hdrs/|  still: |"
echo "wrote $OUT ($(stat -c %s "$OUT") B vs $(stat -c %s "$IN") B)"
