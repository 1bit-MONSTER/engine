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
# image-probe.sh <image.img> (--payload <dir> | --engine-sha <sha256>) [--tag <tag>]
#
# The pre-write probe (engine#3287): before a stick is written, open the IMAGE'S OWN filesystem and prove it
# carries the payload that was just built and tested - never a staged copy that a re-roll forgot to refresh.
# On the night of c34 an image carried the previous engine because the probe printed "cannot read the image's
# SFS - skipped" and the write went ahead. Here every path that cannot look is a refusal, and a refusal is a
# non-zero exit, so a write gated on this script cannot proceed on silence.
#
# What it reads (no root, no mount, no loop device: debugfs reads the ext4 partition at its byte offset, and
# unsquashfs reads the SFS as a file, so "could not mount" is not a way to skip):
#   1. the GPT, for the partition named 1BIT-DATA (mkimage.sh's layout);
#   2. /1bit on that partition: SHA256SUMS and every file it names (vmlinuz, initrd.gz, the four SFS) - copied
#      out and checked with sha256sum -c, so the frugal set on the image is internally consistent;
#   3. inside puppy_*.sfs: /opt/1bit-package (the payload) and /etc/1bit/release-identity.
# What it proves:
#   --payload <dir>     the image's /opt/1bit-package is the SAME TREE as <dir>: every name, every byte, every
#                       symlink target (diff -r --no-dereference), and the identity's engine_sha256 is <dir>/1bit/1bit's
#   --engine-sha <sha>  the image's 1bit/1bit and the identity's engine_sha256 both equal <sha> (when no payload
#                       tree is at hand, e.g. re-checking an archived image against the release table)
#   --tag <tag>         the identity's payload= label is lemonade-onebit-<tag>
#
# Exit: 0 PROBE PASS, 1 PROBE FAIL (a mismatch: do not write), 2 PROBE REFUSED (could not look: do not write).
# PROBE_WORK=<dir> is the scratch directory (default: beside the image; the copied set is ~1.3 GB, which is why it
# is not /tmp - a RAM tmpfs on the build host). It is removed on exit.
set -uo pipefail
prog=$(basename "$0")
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0" | sed -n '/^image-probe.sh /,$p'; }
refuse() { echo "$prog: PROBE REFUSED: $*" >&2; exit 2; }
fail=0
bad() { echo "$prog: MISMATCH: $*" >&2; fail=1; }
ok() { echo "  ok   $*"; }

img="" payload="" esha="" tag=""
while [ $# -gt 0 ]; do
    case "$1" in
        --payload) payload=${2:-}; shift 2 ;;
        --engine-sha) esha=${2:-}; shift 2 ;;
        --tag) tag=${2:-}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; exit 2 ;;
        *) [ -z "$img" ] || { usage >&2; exit 2; }; img=$1; shift ;;
    esac
done
[ -n "$img" ] || { usage >&2; exit 2; }
[ -n "$payload" ] || [ -n "$esha" ] || refuse "name what the image must carry: --payload <dir> or --engine-sha <sha256>"
[ -r "$img" ] || refuse "cannot read $img"
for t in sfdisk debugfs unsquashfs sha256sum python3 diff; do
    command -v "$t" > /dev/null || refuse "no $t on this host"
done
if [ -n "$payload" ]; then
    [ -f "$payload/1bit/1bit" ] || refuse "$payload has no 1bit/1bit: not a payload tree"
    want=$(sha256sum "$payload/1bit/1bit" | cut -c1-64)
    [ -z "$esha" ] || [ "$esha" = "$want" ] || refuse "--engine-sha $esha disagrees with $payload/1bit/1bit ($want)"
    esha=$want
fi
[[ "$esha" =~ ^[0-9a-f]{64}$ ]] || refuse "engine sha256 '$esha' is not 64 hex digits"

img=$(cd "$(dirname "$img")" && pwd)/$(basename "$img")
work=${PROBE_WORK:-$(dirname "$img")/.image-probe.$$}
mkdir -p "$work" || refuse "cannot create the scratch directory $work"
trap 'rm -rf "$work"' EXIT
echo "== $prog $img"
echo "   image sha256 $(sha256sum "$img" | cut -c1-64)  ($(stat -c %s "$img") B)"

# 1. the data partition's byte offset, from the GPT
start=$(sfdisk -J "$img" 2> /dev/null | python3 -c '
import json, sys
try:
    parts = json.load(sys.stdin)["partitiontable"]["partitions"]
except Exception:
    sys.exit(1)
s = [p["start"] for p in parts if p.get("name") == "1BIT-DATA"]
print(s[0] if len(s) == 1 else "")') || refuse "sfdisk cannot read a partition table in $img"
[ -n "$start" ] || refuse "no single partition named 1BIT-DATA in $img"
dev="$img?offset=$((start * 512))"
debugfs -R 'stat /1bit' "$dev" > /dev/null 2>&1 || refuse "debugfs cannot open the 1BIT-DATA ext4 at sector $start"
ok "1BIT-DATA at sector $start opens"

# 2. the frugal set and its own SHA256SUMS
mkdir -p "$work/1bit"
debugfs -R "dump /1bit/SHA256SUMS $work/1bit/SHA256SUMS" "$dev" > /dev/null 2>&1
[ -s "$work/1bit/SHA256SUMS" ] || refuse "the image has no /1bit/SHA256SUMS"
names=$(sed -E 's/^[0-9a-f]{64} +\*?(\.\/)?//' "$work/1bit/SHA256SUMS")
[ -n "$names" ] || refuse "/1bit/SHA256SUMS names no file"
for n in $names; do
    case "$n" in */*|..|.) refuse "/1bit/SHA256SUMS names a path outside /1bit: $n" ;; esac
    debugfs -R "dump /1bit/$n $work/1bit/$n" "$dev" > /dev/null 2>&1
    [ -f "$work/1bit/$n" ] || refuse "/1bit/$n is named in SHA256SUMS and is not on the image"
done
if (cd "$work/1bit" && sha256sum --quiet -c SHA256SUMS > "$work/sums.out" 2>&1); then
    ok "/1bit: $(echo "$names" | wc -w) files match the image's own SHA256SUMS"
else
    bad "/1bit on the image does not match its own SHA256SUMS: $(tr '\n' ' ' < "$work/sums.out")"
fi
sfs=$(ls "$work"/1bit/puppy_*.sfs 2> /dev/null | head -1)
[ -n "$sfs" ] || refuse "no puppy_*.sfs in the image's /1bit"

# 3. the payload and the identity inside the SFS
unsquashfs -q -n -d "$work/root" "$sfs" opt/1bit-package etc/1bit/release-identity > "$work/unsq.out" 2>&1 ||
    refuse "unsquashfs cannot read opt/1bit-package from $(basename "$sfs"): $(tail -2 "$work/unsq.out" | tr '\n' ' ')"
[ -f "$work/root/opt/1bit-package/1bit/1bit" ] || refuse "$(basename "$sfs") has no opt/1bit-package/1bit/1bit"
[ -f "$work/root/etc/1bit/release-identity" ] || refuse "$(basename "$sfs") has no etc/1bit/release-identity"
got=$(sha256sum "$work/root/opt/1bit-package/1bit/1bit" | cut -c1-64)
[ "$got" = "$esha" ] && ok "opt/1bit-package/1bit/1bit is $esha" ||
    bad "the image's engine is $got, the build's is $esha (a stale payload)"
idsha=$(sed -n 's/^engine_sha256=//p' "$work/root/etc/1bit/release-identity")
[ "$idsha" = "$esha" ] && ok "release-identity engine_sha256 matches" ||
    bad "release-identity says engine_sha256=${idsha:-<none>}, the build's engine is $esha"
if [ -n "$tag" ]; then
    lab=$(sed -n 's/^payload=//p' "$work/root/etc/1bit/release-identity")
    [ "$lab" = "lemonade-onebit-$tag" ] && ok "release-identity payload=$lab" ||
        bad "release-identity says payload=${lab:-<none>}, this cut is lemonade-onebit-$tag"
fi
if [ -n "$payload" ]; then
    if diff -r --no-dereference -q "$payload" "$work/root/opt/1bit-package" > "$work/diff.out" 2>&1; then
        ok "opt/1bit-package is the payload tree byte for byte ($(find "$payload" -type f | wc -l) files)"
    else
        bad "opt/1bit-package differs from $payload in $(wc -l < "$work/diff.out") place(s):"
        head -20 "$work/diff.out" | sed 's/^/       /' >&2
    fi
fi

if [ "$fail" = 0 ]; then echo "PROBE PASS: $img carries engine $esha"; exit 0; fi
echo "PROBE FAIL: do not write $img" >&2
exit 1
