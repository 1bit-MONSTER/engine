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
# Installs 1bit OS from this live boot onto ONE internal disk, laid out exactly as the live stick is:
# an EFI system partition (shim + signed grub + vmlinuz/initramfs + grub.cfg, the same files this boot
# came from) and a 1BIT-DATA ext4 partition filling the rest, carrying the stick's models, backends,
# configuration and SSH keys. The installed system boots the same init; it detects /data/.installed and
# says INSTALLED instead of LIVE.
#
# It writes a disk, so every refusal is checked before anything is touched: no --yes, a partition
# instead of a whole disk, a removable device, the disk this live boot is running from, a device with
# anything mounted, or a missing GPT tool. Then it prints the disk and waits 10 s.
#
# Exit: 0 installed, 1 usage, 2 refused (nothing written), 3 failed while writing.
set -u

ESP_MB=512
DATA_LABEL=1BIT-DATA
ESP_LABEL=1BIT-OS
# by absolute path, because this is a busybox shell: with CONFIG_FEATURE_SH_STANDALONE it runs its OWN
# applet for a name it knows before it looks at PATH, so `mke2fs` reached busybox's ext2-only applet
# instead of the e2fsprogs binary the image ships ("invalid option -- 't'", measured in the VM). The
# tools this installer depends on are therefore named, and checked, where mkimage.sh puts them.
SFDISK=/bin/sfdisk
MKE2FS=/bin/mke2fs

new_part() {   # <disk> <1|2> -> the kernel's device name for that partition
    # by name, not by asking the kernel whether the node exists yet: right after sfdisk it does not, and
    # "does /dev/nvme0n1p1 exist?" answered "no" for a disk whose partitions were simply not published
    # yet, which is how this wrote /dev/nvme0n11 (measured in the VM).
    case "${1##*/}" in
        *[0-9]) echo "$1p$2" ;;   # nvme0n1, mmcblk0: a digit glued to a number needs the p
        *)      echo "$1$2" ;;    # sda, vda, hda
    esac
}

usage() {
    # by pattern, not by line number: this file's header moved once (the SPDX line) and the numbered
    # range then printed half a sentence and the code below it
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

# The disk this live boot is running from: /data is mounted from the stick's 1BIT-DATA partition, and its
# ESP (partition 1 of that same disk) holds the signed chain to copy. Ask /proc/mounts, not findfs: once a
# machine has been installed there are TWO 1BIT-DATA partitions and TWO 1BIT-OS ones, the stick's and the
# installed disk's, and findfs may answer with the installed one. In the VM that made the installer refuse
# the target disk as "this live boot's disk" (the disk it had just labelled), which would block every
# reinstall, and it would have copied the wrong ESP.
live=$(awk '$2 == "/data" { print $1; exit }' /proc/mounts)
live_esp=""
if [ -n "$live" ]; then
    lname=${live#/dev/}; lname=${lname##*/}
    # /sys/class/block/sda2/.. resolves to the disk's directory, so its basename is the disk
    ldisk=$(basename "$(readlink -f "/sys/class/block/$lname/.." 2>/dev/null)" 2>/dev/null || true)
    [ "$ldisk" = "$name" ] && die "$DEV carries this live boot's 1BIT-DATA partition; install onto a different disk"
    [ -n "$ldisk" ] && live_esp=$(new_part "/dev/$ldisk" 1)
fi

# anything mounted from this disk (any partition of it) means it is in use
der=$(awk -v d="/dev/$name" 'index($1, d) == 1 { print $1; exit }' /proc/mounts 2>/dev/null)
[ -n "${der:-}" ] && die "$DEV is in use: $der is mounted"

[ -x "$SFDISK" ] || die "sfdisk is missing from this image; cannot lay out a GPT disk"
[ -x "$MKE2FS" ] || die "mke2fs is missing from this image; cannot make the data partition"

# --- what it is about to destroy ---------------------------------------------------------------------
model=$(cat "$sys/device/model" 2>/dev/null | tr -s ' ' | sed 's/^ //;s/ $//')
[ -n "$model" ] || model="(no model reported)"
bytes=$(blockdev --getsize64 "$DEV" 2>/dev/null || echo 0)
gb=$((bytes / 1000000000))
echo "1bit OS install"
echo "  target : $DEV  ($model, ${gb} GB)"
echo "  layout : GPT - partition 1 = ${ESP_MB} MB EFI system partition ($ESP_LABEL), the rest = ext4 ($DATA_LABEL)"
echo "  source : this live boot (the ESP it booted from, and its /data)"
echo "  current contents of $DEV, about to be destroyed:"
# sfdisk, not busybox fdisk: the image carries the real one for the layout, and it is the one tool here
# that can read a GPT back (busybox fdisk is MBR-only and calls a GPT disk unrecognised)
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

# --- write -------------------------------------------------------------------------------------------
esp=$(new_part "$DEV" 1)
data=$(new_part "$DEV" 2)
esp_mb=$ESP_MB

# the same script mkimage.sh uses to build the live stick, so an installed disk and a stick are the
# same layout: 2048-sector start, EFI system partition, then the data partition filling the disk
printf 'label: gpt\nstart=2048, size=%s, type=U, name="EFI"\ntype=L, name="%s"\n' \
    $((esp_mb * 2048)) "$DATA_LABEL" | $SFDISK -q "$DEV" || { echo "1bit-install: sfdisk failed" >&2; exit 3; }

# the kernel needs a moment to publish the new partitions
i=0
while [ ! -b "$esp" ] || [ ! -b "$data" ]; do
    i=$((i + 1))
    [ "$i" -gt 20 ] && { echo "1bit-install: the kernel never published $esp / $data" >&2; exit 3; }
    sleep 1
done
sleep 1

mkdosfs -F 32 -n "$ESP_LABEL" "$esp" >/dev/null || { echo "1bit-install: mkdosfs failed" >&2; exit 3; }
$MKE2FS -q -t ext4 -L "$DATA_LABEL" "$data" || { echo "1bit-install: mke2fs failed" >&2; exit 3; }

mkdir -p /tmp/install/esp /tmp/install/data /tmp/install/live-esp
[ -n "$live_esp" ] || live_esp=$(findfs LABEL=$ESP_LABEL 2>/dev/null || true)   # fallback: no /data mount
mount -t vfat "$esp" /tmp/install/esp || { echo "1bit-install: cannot mount the new ESP" >&2; exit 3; }

if [ -n "$live_esp" ] && mount -t vfat -o ro "$live_esp" /tmp/install/live-esp 2>/dev/null; then
    # the files this boot came from: shim, grub, vmlinuz, initramfs, grub.cfg - the signed chain, byte
    # for byte the same, which is why the MOK enrolment on the machine needs no change
    cp -a /tmp/install/live-esp/. /tmp/install/esp/ || { echo "1bit-install: copying the ESP failed" >&2; exit 3; }
    umount /tmp/install/live-esp
else
    echo "1bit-install: the live ESP is not mountable; the installed disk would not boot" >&2
    umount /tmp/install/esp
    exit 3
fi
sync
umount /tmp/install/esp || { echo "1bit-install: cannot flush the ESP" >&2; exit 3; }

mount -t ext4 "$data" /tmp/install/data || { echo "1bit-install: cannot mount the new data partition" >&2; exit 3; }
if grep -q " /data " /proc/mounts; then
    # models, Lemonade's backends/config/cache, the SSH keys, the NPU kernel sets: the appliance that
    # the machine already knows, moved to the disk
    cp -a /data/. /tmp/install/data/ || { echo "1bit-install: copying /data failed" >&2; exit 3; }
    echo "1bit-install: copied /data ($(du -sh /data 2>/dev/null | cut -f1)) onto the disk"
else
    echo "1bit-install: no /data on this boot; the installed disk gets an empty data partition"
fi
# this is what makes the next boot say INSTALLED instead of LIVE (os/mini/init.sh)
: > /tmp/install/data/.installed
sync
umount /tmp/install/data || { echo "1bit-install: cannot flush the data partition" >&2; exit 3; }

echo
echo "1bit OS is installed on $DEV."
echo "  partition 1: $esp  ${esp_mb} MB, EFI system partition ($ESP_LABEL) - shim, grub, vmlinuz, initramfs, grub.cfg"
echo "  partition 2: $data  ext4, $DATA_LABEL - the models, backends, configuration and keys"
echo
echo "  Next: power off, remove the stick, and boot the machine from this disk."
echo "  If the firmware asks, its boot entry for this disk should point at \\\\EFI\\\\BOOT\\\\BOOTX64.EFI."
echo "  Secure Boot needs no change: the kernel and initramfs are the same MOK-signed files the stick booted."
echo "  To get back to the live stick, boot the stick again - the installed disk is not touched by it."
