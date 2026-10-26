#!/bin/sh
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
# 1bit-install /dev/nvme0n1 --yes
#
# Installs 1bit OS from this live boot onto ONE internal disk, in the layout an installed box has:
#
#   partition 1  512 MB  EFI system partition (1BIT-OS)  - shim, our MOK-signed grub, and a grub.cfg
#                                                          that boots the disk's own /boot
#   partition 2  ext4   1BIT-ROOT  - the live squashfs unpacked, plus /boot (kernel + initramfs) and an
#                                    fstab that mounts / and /data by label
#   partition 3  ext4   1BIT-DATA  - the stick's /data (models, Lemonade's backends and config, SSH keys)
#
# 1BIT-ROOT is min(64 GB, max(24 GB, 25 % of the disk)); 1BIT-DATA is the rest.
#
# It writes a disk, so every refusal is checked before anything is touched: no --yes, a partition
# instead of a whole disk, a removable device, the disk this live boot runs from, a device with
# anything mounted, or a missing GPT/filesystem tool. Then it prints the disk and waits 10 s.
#
# Exit: 0 installed, 1 usage, 2 refused (nothing written), 3 failed while writing.
set -u

ESP_MB=512
ESP_LABEL=1BIT-OS
ROOT_LABEL=1BIT-ROOT
DATA_LABEL=1BIT-DATA
# by absolute path, and checked before anything is written - the rule #539 learned the hard way on the
# BusyBox image (its shell ran its own applet for a name it knew, before PATH). This base has no busybox,
# but the installer names its tools the same way so the two images cannot drift.
SFDISK=/usr/sbin/sfdisk
MKE2FS=/usr/sbin/mke2fs
MKFS_VFAT=/usr/sbin/mkfs.vfat
UNSQUASHFS=/usr/bin/unsquashfs
EFIBOOTMGR=/usr/bin/efibootmgr
# the live squashfs is a partition of the live medium (PARTLABEL, not a filesystem label: squashfs has no
# label of its own), mounted read-only as the overlay's lower layer by the live initramfs
LIVE_SQ=/dev/disk/by-partlabel/1BIT-LIVE

new_part() {   # <disk> <1|2|3> -> the kernel's device name for that partition
    case "${1##*/}" in
        *[0-9]) echo "$1p$2" ;;   # nvme0n1, mmcblk0: a digit glued to a number needs the p
        *)      echo "$1$2" ;;    # sda, vda, hda
    esac
}

usage() {
    sed -n '/^# 1bit-install \/dev/,/^# Exit:/p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//'
}

die() { printf '1bit-install: REFUSED: %s\n' "$*" >&2; exit 2; }

DEV=""
YES=0
for a in "$@"; do
    case "$a" in
        --yes|-y) YES=1 ;;
        --help|-h) usage; exit 0 ;;
        -*) printf '1bit-install: unknown option %s\n' "$a" >&2; exit 1 ;;
        *) if [ -z "$DEV" ]; then DEV=$a; else printf '1bit-install: one target disk, not two\n' >&2; exit 1; fi ;;
    esac
done

if [ -z "$DEV" ]; then
    printf '1bit-install: name the disk to install onto, and pass --yes when you mean it.\n\n' >&2
    usage >&2
    exit 1
fi

# --- refusals, before anything is written ------------------------------------------------------------
[ -b "$DEV" ] || die "$DEV is not a block device"
name=${DEV#/dev/}; name=${name##*/}
sys=/sys/class/block/$name
[ -d "$sys" ] || die "$DEV is not a disk this kernel knows"
[ -e "$sys/partition" ] && die "$DEV is a partition; name the whole disk (e.g. /dev/nvme0n1, not /dev/nvme0n1p1)"
[ "$YES" = 1 ] || die "refusing to write $DEV without --yes (this destroys its contents)"
[ "$(cat "$sys/removable" 2>/dev/null || echo 0)" = 1 ] &&
    die "$DEV is a removable device; 1bit OS installs onto an internal disk, and the live stick is left as it is"

# The disk this live boot is running from: /data is the stick's 1BIT-DATA partition, mounted by the live
# system. Ask /proc/mounts, never findfs: once a machine has been installed there are two 1BIT-DATA
# partitions and two 1BIT-OS ones (the stick's and the installed disk's) and findfs may answer with the
# wrong one - #539 measured exactly that, and it made a reinstall impossible.
live=$(awk '$2 == "/data" { print $1; exit }' /proc/mounts)
live_esp=""
if [ -n "$live" ]; then
    lname=${live#/dev/}; lname=${lname##*/}
    ldisk=$(basename "$(readlink -f "/sys/class/block/$lname/.." 2>/dev/null)" 2>/dev/null || true)
    [ "$ldisk" = "$name" ] && die "$DEV carries this live boot's 1BIT-DATA partition; install onto a different disk"
    [ -n "$ldisk" ] && live_esp=$(new_part "/dev/$ldisk" 1)
fi

# anything mounted from this disk (any partition of it) means it is in use
der=$(awk -v d="/dev/$name" 'index($1, d) == 1 { print $1; exit }' /proc/mounts 2>/dev/null)
[ -n "${der:-}" ] && die "$DEV is in use: $der is mounted"

for t in "$SFDISK" "$MKE2FS" "$MKFS_VFAT" "$UNSQUASHFS"; do
    [ -x "$t" ] || die "$(basename "$t") is missing from this image; cannot lay out or fill a disk"
done
[ -b "$LIVE_SQ" ] || die "this boot has no live squashfs at $LIVE_SQ; it cannot install itself"

# --- what it is about to destroy ---------------------------------------------------------------------
model=$(cat "$sys/device/model" 2>/dev/null | tr -s ' ' | sed 's/^ //;s/ $//')
[ -n "$model" ] || model="(no model reported)"
bytes=$(blockdev --getsize64 "$DEV" 2>/dev/null || echo 0)
gb=$((bytes / 1000000000))
total_mb=$((bytes / 1000000))
root_mb=$((total_mb / 4))
[ "$root_mb" -lt 24576 ] && root_mb=24576
[ "$root_mb" -gt 65536 ] && root_mb=65536
[ "$total_mb" -gt $((ESP_MB + root_mb + 1024)) ] ||
    die "$DEV is ${gb} GB: too small for a ${ESP_MB} MB ESP, ${root_mb} MB root and a data partition"

echo "1bit OS install"
echo "  target : $DEV  ($model, ${gb} GB)"
echo "  layout : GPT - ${ESP_MB} MB EFI system partition ($ESP_LABEL), ${root_mb} MB ext4 ($ROOT_LABEL), the rest ext4 ($DATA_LABEL)"
echo "  source : this live boot (its squashfs, its ESP, and its /data)"
echo "  current contents of $DEV, about to be destroyed:"
$SFDISK -l "$DEV" 2>/dev/null | sed 's/^/    /'
echo
echo "  Nothing is written for 10 seconds."
i=10
while [ "$i" -gt 0 ]; do
    printf '\r  starting in %2d s ' "$i"
    sleep 1
    i=$((i - 1))
done
printf '\r  writing now      \n'

# --- partition --------------------------------------------------------------------------------------
esp=$(new_part "$DEV" 1)
root=$(new_part "$DEV" 2)
data=$(new_part "$DEV" 3)

printf 'label: gpt\nstart=2048, size=%s, type=U, name="EFI"\nsize=%s, type=L, name="%s"\ntype=L, name="%s"\n' \
    $((ESP_MB * 2048)) $((root_mb * 2048)) "$ROOT_LABEL" "$DATA_LABEL" |
    $SFDISK -q "$DEV" || { echo "1bit-install: sfdisk failed" >&2; exit 3; }

i=0
while [ ! -b "$esp" ] || [ ! -b "$root" ] || [ ! -b "$data" ]; do
    i=$((i + 1))
    [ "$i" -gt 20 ] && { echo "1bit-install: the kernel never published $esp / $root / $data" >&2; exit 3; }
    sleep 1
done
sleep 1
$SFDISK -l "$DEV" 2>/dev/null | sed 's/^/  /'

$MKFS_VFAT -F 32 -n "$ESP_LABEL" "$esp" >/dev/null || { echo "1bit-install: mkfs.vfat failed" >&2; exit 3; }
$MKE2FS -q -t ext4 -L "$ROOT_LABEL" "$root" || { echo "1bit-install: mke2fs failed (root)" >&2; exit 3; }
$MKE2FS -q -t ext4 -L "$DATA_LABEL" "$data" || { echo "1bit-install: mke2fs failed (data)" >&2; exit 3; }

T=/tmp/install
mkdir -p "$T/esp" "$T/root" "$T/data" "$T/live-esp"
mount -t vfat "$esp" "$T/esp" || { echo "1bit-install: cannot mount the new ESP" >&2; exit 3; }
mount -t ext4 "$root" "$T/root" || { echo "1bit-install: cannot mount the new root" >&2; exit 3; }
mount -t ext4 "$data" "$T/data" || { echo "1bit-install: cannot mount the new data partition" >&2; exit 3; }

# --- the root filesystem: the live squashfs, unpacked ------------------------------------------------
echo "1bit-install: unpacking the live system onto $root"
$UNSQUASHFS -f -d "$T/root" "$LIVE_SQ" >/dev/null ||
    { echo "1bit-install: unsquashfs failed" >&2; exit 3; }

# an installed machine is not a live one: the marker the live image ships is what release-check reads,
# and 1bit-first-boot.service writes /etc/1bit/installed after it has made this machine's own SSH host
# keys. Writing that marker here would skip the first boot and leave the live image's public key as this
# machine's identity - so this removes one and writes the other, and never the other way round.
rm -f "$T/root/etc/1bit/live"
mkdir -p "$T/root/etc/1bit"
{
    echo "# Written by 1bit-install. Mounted by label, so the same fstab works on any disk."
    echo "LABEL=$ROOT_LABEL  /      ext4  defaults,noatime  0 1"
    echo "LABEL=$DATA_LABEL  /data  ext4  defaults,nofail   0 2"
} > "$T/root/etc/fstab"

# what the box updates from: the repository and the key are packages, not files written here. Only
# assert that the source is the production one, so an install cannot silently come out with a test source.
[ -f "$T/root/etc/apt/sources.list.d/1bit.list" ] ||
    echo "1bit-install: WARNING - /etc/apt/sources.list.d/1bit.list is not in the live system" >&2

# --- the ESP: the live boot's own signed chain, plus a grub.cfg for the disk -------------------------
if [ -n "$live_esp" ] && mount -t vfat -o ro "$live_esp" "$T/live-esp" 2>/dev/null; then
    cp -a "$T/live-esp"/. "$T/esp"/ || { echo "1bit-install: copying the ESP failed" >&2; exit 3; }
    umount "$T/live-esp"
else
    echo "1bit-install: the live ESP is not mountable; the installed disk would not boot" >&2
    exit 3
fi
# shim hands over to grubx64.efi beside it, which reads this file. The kernel and initramfs are on the
# root partition now (that is what a package-managed /boot means), so grub searches for the label and
# reads /boot from it - and the live boot's own copies of them, stale the moment apt upgrades the
# kernel, come off the ESP. One entry per installed kernel: that is the rollback apt kept installed.
rm -f "$T/esp/vmlinuz" "$T/esp/initrd.img"
{
    echo "# Written by 1bit-install: boot the installed 1BIT-ROOT, one entry per kernel in /boot."
    echo "set default=0"
    echo "set timeout=3"
    emit() {   # <kver>
        echo "menuentry '1bit OS, Lemonade Server appliance (kernel $1)' {"
        echo "    search --no-floppy --label $ROOT_LABEL --set=root"
        echo "    linux /boot/vmlinuz-$1 root=LABEL=$ROOT_LABEL ro quiet"
        echo "    initrd /boot/initrd.img-$1"
        echo "}"
    }
    for v in $(ls "$T/root"/boot/vmlinuz-* 2>/dev/null | sed 's|.*/vmlinuz-||' | sort -V -r); do
        emit "$v"
    done
} > "$T/esp/EFI/BOOT/grub.cfg"
sync

# --- the data partition: the stick's own /data -------------------------------------------------------
if grep -q " /data " /proc/mounts; then
    echo "1bit-install: copying /data ($(du -sh /data 2>/dev/null | cut -f1)) onto the disk"
    cp -a /data/. "$T/data/" || { echo "1bit-install: copying /data failed" >&2; exit 3; }
else
    echo "1bit-install: no /data on this boot; the installed disk gets an empty data partition"
    mkdir -p "$T/data/models" "$T/data/lemonade/config" "$T/data/lemonade/cache" "$T/data/cache/hrx-jit"
fi
sync

umount "$T/esp" && umount "$T/root" && umount "$T/data" ||
    { echo "1bit-install: cannot flush the new partitions" >&2; exit 3; }

# --- the firmware entry ------------------------------------------------------------------------------
# The removable-media path (\\EFI\\BOOT\\BOOTX64.EFI) already boots this ESP; an explicit entry keeps
# the boot order stable after other media has been present. Not fatal: a VM's firmware may refuse it.
if [ -x "$EFIBOOTMGR" ] && [ -d /sys/firmware/efi ]; then
    $EFIBOOTMGR -c -d "$DEV" -p 1 -L "1bit OS" -l '\EFI\BOOT\BOOTX64.EFI' >/dev/null 2>&1 &&
        echo "1bit-install: firmware boot entry '1bit OS' created" ||
        echo "1bit-install: WARNING - efibootmgr could not create an entry; the removable-media path still boots"
fi

echo
echo "1bit OS is installed on $DEV."
echo "  partition 1: $esp  ${ESP_MB} MB, EFI system partition ($ESP_LABEL) - shim, grub, grub.cfg"
echo "  partition 2: $root  ${root_mb} MB, ext4 ($ROOT_LABEL) - the system, its packages and /boot"
echo "  partition 3: $data  ext4 ($DATA_LABEL) - the models, backends, configuration and keys"
echo
echo "  Next: power off, remove the stick, and boot the machine from this disk."
echo "  The first boot generates this machine's own SSH host keys and writes /etc/1bit/installed."
echo "  Secure Boot needs no change: shim, grub and the kernel on this disk are the MOK-signed files the stick booted."
echo "  Updates: apt update && apt upgrade. The previous kernel stays installed; apt can go back to it."
