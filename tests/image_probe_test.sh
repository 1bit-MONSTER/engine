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
# image_probe_test.sh - os/v3/puppy/image-probe.sh (engine#3287) passes only the image that carries the payload,
# fails a stale one, and REFUSES (non-zero) whenever it cannot look. Builds a miniature appliance image the way
# mkimage.sh lays it out (GPT: EFI + 1BIT-DATA ext4 with /1bit/{SHA256SUMS,vmlinuz,initrd.gz,puppy_*.sfs}), as an
# ordinary user: mksquashfs, mkfs.ext4 -d and sfdisk on a file need no root, and neither does the probe.
# archive-image.sh is checked on the same image: the archived copy must hash to the source and carry its .sha256.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
probe=$root/os/v3/puppy/image-probe.sh
archive=$root/os/v3/puppy/archive-image.sh
for t in mksquashfs unsquashfs mkfs.ext4 sfdisk debugfs python3; do
    command -v "$t" > /dev/null || { echo "image_probe_test: skip (no $t)"; exit 77; }
done
w=$(mktemp -d); trap 'rm -rf "$w"' EXIT
pass=0 failn=0
ok() { echo "ok   $*"; pass=$((pass + 1)); }
no() { echo "FAIL $*"; failn=$((failn + 1)); }
expect() {  # expect <want exit> <label> <cmd...>
    local want=$1 label=$2; shift 2
    "$@" > "$w/out" 2>&1; local got=$?
    [ "$got" = "$want" ] && ok "$label (exit $got)" || { no "$label: exit $got, want $want"; sed 's/^/     /' "$w/out"; }
}

# the payload, and an SFS that carries it + an identity naming its engine
p=$w/payload; mkdir -p "$p/1bit" "$p/bin"
printf 'engine-bytes\n' > "$p/1bit/1bit"; printf 'lemond\n' > "$p/bin/lemond"; ln -s 1bit/1bit "$p/onebit-link"
mk_image() {  # mk_image <payload> <identity engine sha> <out.img>
    local pl=$1 idsha=$2 out=$3 r=$w/rootfs d=$w/data
    rm -rf "$r" "$d"; mkdir -p "$r/opt" "$r/etc/1bit" "$d/1bit"
    cp -a "$pl" "$r/opt/1bit-package"
    printf 'payload=lemonade-onebit-t1\nengine_sha256=%s\n' "$idsha" > "$r/etc/1bit/release-identity"
    mksquashfs "$r" "$d/1bit/puppy_test.sfs" -all-root -noappend -quiet > /dev/null 2>&1 || return 1
    printf 'k\n' > "$d/1bit/vmlinuz"; printf 'i\n' > "$d/1bit/initrd.gz"
    (cd "$d/1bit" && sha256sum vmlinuz initrd.gz ./*.sfs > SHA256SUMS)
    rm -f "$w/data.part"; mkfs.ext4 -q -F -L 1BIT-DATA -d "$d" "$w/data.part" 16M > /dev/null 2>&1 || return 1
    truncate -s $(( (1 + 1 + 16 + 1) * 1048576 )) "$out"
    printf 'label: gpt\nstart=2048, size=2048, type=U, name="EFI"\ntype=L, name="1BIT-DATA"\n' | sfdisk -q "$out" 2> /dev/null || return 1
    dd if="$w/data.part" of="$out" bs=1M seek=2 conv=notrunc status=none
}
esha=$(sha256sum "$p/1bit/1bit" | cut -c1-64)
mk_image "$p" "$esha" "$w/good.img" || { echo "image_probe_test: could not build the test image"; exit 1; }
export PROBE_WORK=$w/probe

expect 0 "the image carrying the payload passes" "$probe" "$w/good.img" --payload "$p" --tag t1
expect 0 "--engine-sha alone passes" "$probe" "$w/good.img" --engine-sha "$esha"
expect 1 "a different tag fails" "$probe" "$w/good.img" --payload "$p" --tag t2
# a re-roll whose payload changed after the image was cut: the image is stale
q=$w/payload2; cp -a "$p" "$q"; printf 'new-engine\n' > "$q/1bit/1bit"
expect 1 "a stale image (the build moved on) fails" "$probe" "$w/good.img" --payload "$q"
# same engine bytes, but another file in the tree differs: the whole tree is compared, not just 1bit
r=$w/payload3; cp -a "$p" "$r"; printf 'lemond-new\n' > "$r/bin/lemond"
expect 1 "a payload differing outside 1bit/1bit fails" "$probe" "$w/good.img" --payload "$r"
rm "$r/bin/lemond"; ln -s 1bit/1bit "$r/bin/lemond"; printf 'lemond\n' > "$r/bin/lemond.real"
expect 1 "a payload with an extra file fails" "$probe" "$w/good.img" --payload "$r"
# the identity disagrees with the bytes it ships
mk_image "$p" "$(printf '%064d' 0)" "$w/badid.img" || exit 1
expect 1 "an identity naming another engine fails" "$probe" "$w/badid.img" --payload "$p"
# could not look -> refusal (2), never a pass
head -c 4096 "$w/good.img" > "$w/trunc.img"
expect 2 "a truncated image is refused" "$probe" "$w/trunc.img" --payload "$p"
expect 2 "a missing image is refused" "$probe" "$w/nope.img" --payload "$p"
expect 2 "no expectation is refused" "$probe" "$w/good.img"
expect 2 "a malformed --engine-sha is refused" "$probe" "$w/good.img" --engine-sha abc
# the frugal set no longer matches its own SHA256SUMS (bytes on the data partition changed after mkimage)
cp "$w/good.img" "$w/sums.img"
off=$(( 2 * 1048576 )); n=$(debugfs -R 'bmap /1bit/vmlinuz 0' "$w/sums.img?offset=$off" 2> /dev/null | tail -1)
if [[ "$n" =~ ^[0-9]+$ ]]; then
    bs=$(debugfs -R stats "$w/sums.img?offset=$off" 2> /dev/null | sed -n 's/^Block size: *//p')
    printf 'X' | dd of="$w/sums.img" bs=1 seek=$(( off + n * bs )) conv=notrunc status=none
    expect 1 "a frugal set that disagrees with its SHA256SUMS fails" "$probe" "$w/sums.img" --payload "$p"
else no "could not locate vmlinuz's block to corrupt (debugfs bmap said '$n')"; fi

# archive-image.sh: the archived copy is the same bytes, with its hash beside it
if [ -x "$archive" ]; then
    expect 0 "archive-image.sh archives the image" "$archive" "$w/good.img" t1 "$w/archive"
    a=$w/archive/t1/1bit-os-puppy-t1.img
    [ -f "$a" ] && cmp -s "$a" "$w/good.img" && ok "the archived image is byte-identical" || no "the archived image differs or is missing"
    (cd "$w/archive/t1" && sha256sum --quiet -c 1bit-os-puppy-t1.img.sha256) && ok "the archive's .sha256 verifies" ||
        no "the archive's .sha256 does not verify"
    expect 1 "archive-image.sh will not overwrite an archived tag" "$archive" "$w/badid.img" t1 "$w/archive"
else no "no executable $archive"; fi

echo "image_probe_test: $pass passed, $failn failed"
[ "$failn" = 0 ]
