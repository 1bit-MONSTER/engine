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
# mk-huge-kernel.sh <kernel-stage> <out-dir> [--firmware "amdgpu amdnpu rtl_nic"] [--kver <kver>]
#
# The 1bit OS kernel as a Puppy kernel package: woof-CE's `kernel-kit/output/huge-<kver>.tar.*`,
# which `3builddistro` unpacks into `vmlinuz` and the `zdrv_<prefix>_<version>.sfs` it boots. This is
# how the Puppy build carries OUR kernel (7.3.0-rc4-1bit-perfopt+ and its out-of-tree modules), with no
# distribution kernel anywhere in the image - docs/os-v3-live-puppy-eval.md, and the owner's requirement
# that the Puppy appliance boots our kernel, our modules and our firmware.
#
# <kernel-stage> is the release kernel as it sits on the build machine, the same shape os/v2/mkdebs.sh
# takes: <dir>/boot/vmlinuz-<kver>, <dir>/boot/config-<kver> and <dir>/lib/modules/<kver>/ (the whole
# tree, including /lib/modules/<kver>/updates/ with amdxdna.ko and ec_su_axb35.ko).
#
# What it writes into <out-dir>:
#   huge-<kver>.tar.zst                        the kernel package woof-CE consumes
#   vmlinuz-<kver>                             the kernel image inside it
#   kernel-modules-zdrv-<kver>.sfs             the zdrv: /lib/modules/<kver> + the firmware + a marker
#
# The zdrv is an SFS that Puppy mounts and unions over / at boot, so the firmware sits in it at its real
# path (/lib/firmware/<vendor>) and module loading works with no extra step. --firmware picks the
# vendor directories: a whole linux-firmware is ~1 GB and the appliance needs three (measured on the
# reference box: amdgpu 28 MB, amdnpu 556 KB, rtl_nic 180 KB).
#
# A kernel bump: build/obtain the kernel .deb tree, run this again with the new <kver>, drop the new
# tarball in kernel-kit/output, rebuild. No other file changes, which is what makes the bump a script.
#
# Exit: 0 ok, 1 usage, 2 refused (a module for another kernel, or an unsigned one), 3 a tool is missing.
set -euo pipefail
prog=mk-huge-kernel
die() { printf '%s: REFUSED: %s\n' "$prog" "$*" >&2; exit 2; }
usage() { sed -n '/^# mk-huge-kernel.sh/,/^# Exit:/p' "$0" | sed 's/^# \{0,1\}//'; }

[ $# -ge 2 ] || { usage >&2; exit 1; }
stage=$1 out=$2
shift 2
firmware="amdgpu amdnpu rtl_nic"
kver=""
while [ $# -gt 0 ]; do
    case $1 in
    --firmware) firmware=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    --kver) kver=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    --help | -h) usage; exit 0 ;;
    *) printf '%s: unknown option %s\n' "$prog" "$1" >&2; exit 1 ;;
    esac
done

for t in mksquashfs depmod file tar zstd find; do
    command -v "$t" >/dev/null 2>&1 || { printf '%s: %s is missing\n' "$prog" "$t" >&2; exit 3; }
done

stage=$(cd "$stage" && pwd)
[ -d "$stage/lib/modules" ] || die "$stage has no lib/modules"
if [ -z "$kver" ]; then
    # one tree in the stage, or the caller says which: a stage that carries two is ambiguous, not a guess
    mapfile -t trees < <(find "$stage/lib/modules" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
    [ ${#trees[@]} -eq 1 ] || die "$stage has ${#trees[@]} module trees; name one with --kver"
    kver=${trees[0]}
fi
[ -d "$stage/lib/modules/$kver" ] || die "$stage has no lib/modules/$kver"
[ -f "$stage/boot/vmlinuz-$kver" ] || die "$stage has no boot/vmlinuz-$kver"

# the image is the kernel the modules were built for: the release string sits uncompressed in the setup
# header, which is what `file` reads (os/mini/mkimage.sh makes the same check for the same reason)
imgkver=$(file -b "$stage/boot/vmlinuz-$kver" | sed -n 's/.*version \([^ ]*\) .*/\1/p' | head -n 1 || true)
[ -z "$imgkver" ] || [ "$imgkver" = "$kver" ] ||
    die "$stage/boot/vmlinuz-$kver is $imgkver, not $kver: the image and the modules are different kernels"

# every out-of-tree module must be this kernel's and signed, or the appliance boots without the NPU and
# without its fans (the os/mini/mkimage.sh lesson, asserted here for the Puppy zdrv)
updates=$stage/lib/modules/$kver/updates
unsigned=""
mismatch=""
if [ -d "$updates" ]; then
    for ko in "$updates"/*.ko "$updates"/*.ko.zst; do
        [ -e "$ko" ] || continue
        magic=$(modinfo -F vermagic "$ko" 2>/dev/null | awk '{print $1}')
        [ "$magic" = "$kver" ] || mismatch="$mismatch $(basename "$ko"):$magic"
        signer=$(modinfo -F signer "$ko" 2>/dev/null)
        [ -n "$signer" ] || unsigned="$unsigned $(basename "$ko")"
    done
fi
[ -z "$mismatch" ] || die "modules for another kernel:$mismatch"
[ -z "$unsigned" ] || die "unsigned modules:$unsigned (Secure Boot refuses them at load)"

mkdir -p "$out"
out=$(cd "$out" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
tree=$work/zdrv
# WHERE the tree goes matters more than it looks. Ubuntu 26.04 is usrmerged and the built puppy SFS carries
# `/lib -> usr/lib` as a symlink (measured with unsquashfs -ll on cycle 4's puppet SFS), so a zdrv that puts
# modules at /lib/modules/<kver> stacks under a symlink and is invisible in the running system: measured in
# a booted appliance image, `ls /lib/modules/<kver>` said "No such file or directory" and `modprobe
# virtio_net`/`e1000` said "not found in directory", so no module at all could load - no NIC, no amdgpu, no
# NPU. Usrmerge layout plus a `lib -> usr/lib` link of our own is what makes both paths resolve.
libroot=${ZDRV_LIBROOT:-usr/lib}
mkdir -p "$tree/$libroot/modules" "$tree/$libroot/firmware" "$tree/etc/1bit"
ln -sfn usr/lib "$tree/lib"   # for depmod and anything that still asks for /lib directly

echo "== the modules ($kver)"
# the tree without build/ and source/: those are symlinks to a headers tree the appliance does not carry,
# and on a box where the headers have been removed they dangle and stop the copy
mkdir -p "$tree/$libroot/modules/$kver"
tar -C "$stage/lib/modules/$kver" --exclude=./build --exclude=./source -cf - . |
    tar -C "$tree/$libroot/modules/$kver" -xf -
# modules.dep for this tree from this tree, so the depmod that Puppy's boot would otherwise need is done
# once, at build time, and a missing dependency is a build error instead of a boot error
if [ -f "$stage/boot/config-$kver" ]; then
    mkdir -p "$work/$libroot/modules"
    depmod -b "$tree" "$kver" 2>/dev/null || depmod -b "$tree" 2>/dev/null || true
fi
# Ubuntu ships its modules compressed (this kernel: .ko.xz, 3819 of them, measured); ours in updates/ are
# not. The kernel's own module decompressor handles both, so the count and the checks take '*.ko*'.
mods=$(find "$tree/$libroot/modules/$kver" -name '*.ko*' | wc -l)
[ "$mods" -gt 1000 ] || printf '%s: note: only %s modules in the zdrv (a kernel package normally carries more)\n' "$prog" "$mods"

echo "== the firmware ($firmware)"
for d in $firmware; do
    if [ -d "/lib/firmware/$d" ]; then
        mkdir -p "$tree/$libroot/firmware/$d"
        cp -aL "/lib/firmware/$d/." "$tree/$libroot/firmware/$d/"
        printf '   %s/firmware/%s  %s\n' "$libroot" "$d" "$(du -sh "$tree/$libroot/firmware/$d" | cut -f1)"
    else
        printf '%s: note: no /lib/firmware/%s on this build machine\n' "$prog" "$d" >&2
    fi
done
[ -n "$(ls -A "$tree/$libroot/firmware")" ] || printf '%s: note: the zdrv carries no firmware\n' "$prog" >&2

# what the machine is: release-check's `kernel` row and a human read this
{
    echo "# Written by os/v3/puppy/mk-huge-kernel.sh"
    echo "kver=$kver"
    echo "image=$(sha256sum "$stage/boot/vmlinuz-$kver" | cut -d' ' -f1)"
    echo "modules=$mods"
    echo "firmware=$firmware"
    for ko in "$updates"/*.ko "$updates"/*.ko.zst; do
        [ -e "$ko" ] || continue
        printf 'module=%s signer=%s\n' "$(basename "$ko")" "$(modinfo -F signer "$ko" 2>/dev/null)"
    done
} >"$tree/etc/1bit/kernel"

cp "$stage/boot/vmlinuz-$kver" "$out/vmlinuz-$kver"
echo "== the zdrv"
rm -f "$out/kernel-modules-zdrv-$kver.sfs"
# xz like the distribution's own zdrv (DISTRO_SPECS SFS compression), noappend so a re-run replaces it
mksquashfs "$tree" "$out/kernel-modules-zdrv-$kver.sfs" -comp xz -noappend -no-progress >/dev/null
echo "== the kernel package"
rm -f "$out/huge-$kver.tar.zst"
tar -C "$out" --zstd -cf "$out/huge-$kver.tar.zst" "vmlinuz-$kver" "kernel-modules-zdrv-$kver.sfs"
printf '%s: vmlinuz-%s %s, zdrv %s, huge-%s.tar.zst %s (%s modules, firmware: %s)\n' \
    "$prog" "$kver" "$(du -h "$out/vmlinuz-$kver" | cut -f1)" \
    "$(du -h "$out/kernel-modules-zdrv-$kver.sfs" | cut -f1)" "$kver" \
    "$(du -h "$out/huge-$kver.tar.zst" | cut -f1)" "$mods" "$firmware"
printf '%s: drop it in woof-out_*/kernel-kit/output/ before 3builddistro\n' "$prog"
