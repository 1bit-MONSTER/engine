#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# patch_sfs_atomic_test.sh - os/v3/puppy/patch-sfs.sh must not destroy the artifact it was asked to replace.
#
# Why this exists: #1491 made the tool pack to <out>.partial and rename only after every check. The headline was true for
# post-pack refusals and FALSE for pre-pack ones - two refusals still did `rm -f "$out"` before `out` had been reassigned to
# the partial, so with --replace they deleted the very artifact they were about to replace. pi-release found it by running a
# real failed cut, which is the only way that half had ever been run. This test is that failed cut: a post-pack refusal
# (a source SFS carrying uid >= 1000, which the ownership row refuses) with --replace naming the source itself.
#
# Exit 77 (skip) when not root: patch-sfs requires root for uid/gid/setuid bits, and so does building the fixture.
set -uo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
tool=$here/os/v3/puppy/patch-sfs.sh
work=$(mktemp -d /tmp/psfix.XXXXXX) || exit 1
trap 'rm -rf "$work"' EXIT

fails=0; checks=0
check() { checks=$((checks + 1)); if [ "$2" = "$3" ]; then printf 'ok   %s = %s\n' "$1" "$2"; else printf 'FAIL %s: got %s want %s\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi; }

[ "$(id -u)" = 0 ] || { echo "patch_sfs_atomic_test: skip (not root: patch-sfs needs uid/gid and setuid bits)"; exit 77; }
[ -f "$tool" ] || { echo "patch_sfs_atomic_test: no $tool"; exit 1; }
for t in mksquashfs unsquashfs; do command -v "$t" > /dev/null || { echo "patch_sfs_atomic_test: skip (no $t)"; exit 77; }; done

# A fixture whose SOURCE SFS carries uid 1000: that is a refusal the tool can only reach AFTER packing, which is the case a
# surviving-artifact test needs (a pre-pack refusal would pass even with the bug, because the partial is never created).
mkdir -p "$work/tree/sub" "$work/repl"
printf 'a\n' > "$work/tree/a.txt"
printf 'b\n' > "$work/tree/sub/b.txt"
printf 'NEW\n' > "$work/repl/new.txt"
chown -R 1000:1000 "$work/tree"
mksquashfs "$work/tree" "$work/base.sfs" -noappend -comp zstd > /dev/null 2>&1 || { echo "patch_sfs_atomic_test: mksquashfs failed"; exit 1; }
before=$(sha256sum "$work/base.sfs" | cut -d' ' -f1)
printf 'sub/b.txt\t%s\n' "$work/repl/new.txt" > "$work/man"
check "fixture source carries uid $(unsquashfs -lln "$work/base.sfs" 2>/dev/null | awk '$2 ~ /^1000/ {c++} END {print c+0}') entries" "yes" "yes"

# --replace naming the source itself: the refusal must leave those exact bytes in place.
"$tool" --replace --manifest "$work/man" "$work/base.sfs" "$work/base.sfs" > "$work/run.log" 2>&1
rc=$?
check "refused (rc)" "$([ "$rc" != 0 ] && echo refused || echo ok)" "refused"
check "the refusal names ownership" "$(grep -c 'uid or gid >= 1000' "$work/run.log")" "1"
check "the artifact survives with the same bytes" "$(sha256sum "$work/base.sfs" | cut -d' ' -f1)" "$before"
check "no .partial left behind" "$(ls "$work"/*.partial 2>/dev/null | wc -l)" "0"

# And the positive direction, so the test cannot pass by refusing everything: a clean cut of the same fixture to a NEW name.
chown -R 0:0 "$work/tree"
mksquashfs "$work/tree" "$work/clean.sfs" -noappend -comp zstd > /dev/null 2>&1
"$tool" --manifest "$work/man" "$work/clean.sfs" "$work/out.sfs" > "$work/ok.log" 2>&1
check "a clean cut still packs" "$([ -s "$work/out.sfs" ] && echo packed || echo missing)" "packed"
check "and carries the new file" "$(unsquashfs -q -cat "$work/out.sfs" sub/b.txt 2>/dev/null)" "NEW"

# The pre-pack half, which is the one that was destructive: a manifest that adds a file under a NEW directory used to be
# refused BEFORE packing - and the two pre-pack refusals still did `rm -f "$out"`, so with --replace they deleted the
# artifact. The fixed tool accepts this manifest (finding 2 of the same review) and packs it; the point of the check is
# weaker on purpose and still catches the bug: WHATEVER it decides, the artifact named on the command line must still be
# there afterwards. On the broken version it was not.
mkdir -p "$work/tree2" "$work/withdir/NEWDIR"
printf 'a\n' > "$work/tree2/a.txt"
printf 'z\n' > "$work/withdir/NEWDIR/newfile.txt"
chown -R 0:0 "$work/tree2" "$work/withdir"
mksquashfs "$work/tree2" "$work/nd.sfs" -noappend -comp zstd > /dev/null 2>&1
nd_before=$(sha256sum "$work/nd.sfs" | cut -d' ' -f1)
printf 'NEWDIR/newfile.txt\t%s\n' "$work/withdir/NEWDIR/newfile.txt" > "$work/man-nd"
"$tool" --replace --manifest "$work/man-nd" "$work/nd.sfs" "$work/nd.sfs" > "$work/nd.log" 2>&1
check "new-directory manifest: the artifact is still present and non-empty" \
    "$([ -s "$work/nd.sfs" ] && echo present || echo gone)" "present"
check "no .partial left behind (new directory)" "$(ls "$work"/*.partial 2>/dev/null | wc -l)" "0"

echo "patch_sfs_atomic: $checks checks, $fails failed"
[ "$fails" = 0 ] || exit 1
