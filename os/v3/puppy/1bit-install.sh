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
# 1bit-install /dev/sdX --yes
#
# Installs this booted 1bit OS (Puppy appliance) onto ONE disk, laid out exactly as the stick is: an EFI
# system partition (the stick's own shim + grub + grub.cfg) and a 1BIT-DATA ext4 partition carrying the
# frugal install (/1bit: vmlinuz, initrd, the four SFS) plus the data contract (/data: models, lemonade,
# npu, ssh, cache). One filesystem for both, because Puppy's psubdir=1bit puts the save layer on the same
# partition the engine reads /data from - the same reason the stick is built that way.
#
# The one value that MUST differ from the stick: the boot line and grub's root. On the stick they name the
# LABEL 1BIT-DATA, and two 1BIT-DATA partitions exist the moment a machine is installed (engine#539/#711),
# so the installed grub.cfg searches this disk's data UUID and passes both pupsfs=UUID=<uuid> (Puppy's init,
# which blkid-matches it) and 1bit.data=UUID=<uuid> (/etc/init.d/05onboard-data).
#
# Nothing is written before every refusal is checked, and the disk this boot came from is refused by
# device, not by guesswork. The firmware entry (U6, #709) is written with efibootmgr and put first.
#
# Exit: 0 installed, 1 usage, 2 refused (nothing written), 3 failed while writing.
set -u

ESP_MB=512
ESP_LABEL=1BIT-OS
DATA_LABEL=1BIT-DATA
ESP_MOUNT=/tmp/1bit-install/esp
DATA_MOUNT=/tmp/1bit-install/data
SRC_ESP_MOUNT=/tmp/1bit-install/src-esp
LOG=/tmp/1bit-install.log

usage() {
    sed -n '/^# 1bit-install \/dev/,/^# Exit:/p' "$0" 2> /dev/null | sed 's/^# \{0,1\}//'
}
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
die() { say "1bit-install: REFUSED: $*" >&2; exit 2; }
fail() { say "1bit-install: FAILED: $*" >&2; exit 3; }

dev=
yes=
for a in "$@"; do
    case $a in
        --yes|-y) yes=1 ;;
        -h|--help) usage; exit 0 ;;
        /dev/*) dev=$a ;;
        *) die "unknown argument '$a' (usage: 1bit-install /dev/sdX --yes)" ;;
    esac
done
[ -n "$dev" ] || { usage >&2; exit 1; }
[ -n "$yes" ] || die "no --yes: this writes a whole disk"

: > "$LOG" 2> /dev/null || LOG=/dev/null

# ── what is this boot, and what must not be touched ──────────────────────────────────────────────
# The disk this boot came from: the partition holding /data (05onboard-data mounts the data partition) or,
# failing that, Puppy's own mount of it. Both are the same filesystem on a healthy boot.
bootpart=$(awk '$2 == "/data" { print $1; exit }' /proc/mounts 2> /dev/null)
[ -n "$bootpart" ] || bootpart=$(awk '$2 == "/initrd/mnt/dev_save" { print $1; exit }' /proc/mounts 2> /dev/null)
[ -n "$bootpart" ] || die "this boot has no /data mount and no Puppy dev_save mount: cannot tell which disk it is running from, and installing over it would destroy the running system"
bootdisk=$(basename "$(readlink -f "/sys/class/block/${bootpart##*/}/.." 2> /dev/null)" 2> /dev/null)
[ -n "$bootdisk" ] || die "cannot resolve the disk behind $bootpart"

target=$(basename "$dev")
[ -b "$dev" ] || die "$dev is not a block device"
[ -d "/sys/block/$target" ] || die "$dev is not a whole disk (a partition cannot be an install target)"
[ "$target" != "$bootdisk" ] || die "$dev is the disk this system booted from ($bootpart is on it)"
for p in /sys/block/"$target"/"$target"*; do
    [ -e "$p" ] || continue
    n=$(basename "$p")
    if grep -q "^/dev/$n " /proc/mounts 2> /dev/null; then
        die "$dev has mounted partitions (/dev/$n): unmount it first"
    fi
done
if grep -q "^$dev " /proc/mounts 2> /dev/null; then die "$dev itself is mounted"; fi

for t in sfdisk mkfs.vfat mkfs.ext4 efibootmgr blkid; do
    command -v "$t" > /dev/null 2> /dev/null || die "$t is missing from this image, so the install cannot complete"
done
[ -d /sys/firmware/efi ] || die "not booted under UEFI: this installer writes an ESP and a firmware entry, and neither can be written or tested without EFI"
[ -d /sys/firmware/efi/efivars ] || die "efivarfs is not mounted, so efibootmgr cannot write the firmware entry (the appliance mounts it in /etc/init.d/03onboard-efivars)"

src_esp=$(findfs LABEL="$ESP_LABEL" 2> /dev/null)
[ -n "$src_esp" ] || die "no $ESP_LABEL partition on this boot: the stick's own shim + grub are what the installed disk has to chain to"
src_data=
if grep -q " /data " /proc/mounts; then src_data=/data; else src_data=/initrd/mnt/dev_save; fi
[ -d "$src_data/1bit" ] || die "no frugal install at $src_data/1bit: this boot cannot install itself"

# ── say what is about to be destroyed, then wait ─────────────────────────────────────────────────
model=$(sed -n 's/^MODEL=//p' "/sys/block/$target/device/model" 2> /dev/null)
[ -n "$model" ] || model=$(cat "/sys/block/$target/device/model" 2> /dev/null)
gb=$(awk -v s="$(cat "/sys/block/$target/size" 2> /dev/null || echo 0)" 'BEGIN { printf "%.1f", s * 512 / 1073741824 }')
say "1bit OS install"
say "  target : $dev  (${model:-unknown model}, ${gb} GB)"
say "  booted from: /dev/$bootdisk ($bootpart) - refused as a target"
say "  source : ESP $src_esp, frugal install + data $src_data"
say "  anything on $dev is about to be destroyed:"
sfdisk -l "$dev" 2> /dev/null | sed 's/^/    /' >> "$LOG"
sfdisk -l "$dev" 2> /dev/null | sed 's/^/    /'
say ""
say "  writing in 10 s (Ctrl-C now and nothing has been touched)"
i=10
while [ "$i" -gt 0 ]; do printf '\r  starting in %2d s ' "$i"; sleep 1; i=$((i - 1)); done
printf '\r  writing now      \n'

# ── write ───────────────────────────────────────────────────────────────────────────────────────
esp=${dev}1
data=${dev}2
case ${dev##*/} in
    *[0-9]) esp=${dev}p1; data=${dev}p2 ;;   # nvme0n1, mmcblk0
esac
datauuid=$(cat /proc/sys/kernel/random/uuid)

say "== 1/7 partition table: ${ESP_MB} MB ESP + $DATA_LABEL"
printf 'label: gpt\nstart=2048, size=%s, type=U, name="EFI"\ntype=L, name="1BIT-DATA"\n' $((ESP_MB * 2048)) \
    | sfdisk -q "$dev" >> "$LOG" 2>&1 || fail "sfdisk could not write the partition table on $dev"
i=0
while [ ! -b "$esp" ] || [ ! -b "$data" ]; do
    i=$((i + 1)); [ "$i" -gt 20 ] && fail "the kernel never published $esp / $data"
    sleep 1
done
sleep 1

say "== 2/7 formats ($ESP_LABEL vfat, $DATA_LABEL ext4)"
mkfs.vfat -F 32 -n "$ESP_LABEL" "$esp" >> "$LOG" 2>&1 || fail "mkfs.vfat on $esp"
mkfs.ext4 -q -F -L "$DATA_LABEL" -U "$datauuid" "$data" >> "$LOG" 2>&1 || fail "mkfs.ext4 on $data"

say "== 3/7 boot chain from this stick's own ESP ($src_esp)"
rm -rf "$ESP_MOUNT" "$DATA_MOUNT" "$SRC_ESP_MOUNT"
mkdir -p "$ESP_MOUNT" "$DATA_MOUNT" "$SRC_ESP_MOUNT"
mount -t vfat "$esp" "$ESP_MOUNT" >> "$LOG" 2>&1 || fail "cannot mount the new ESP"
mount -t vfat -o ro "$src_esp" "$SRC_ESP_MOUNT" >> "$LOG" 2>&1 || fail "cannot mount the source ESP $src_esp"
cp -a "$SRC_ESP_MOUNT/." "$ESP_MOUNT/" >> "$LOG" 2>&1 || fail "copying the ESP tree failed"
umount "$SRC_ESP_MOUNT"

say "== 4/7 frugal install + data contract onto $data"
mount -t ext4 "$data" "$DATA_MOUNT" >> "$LOG" 2>&1 || fail "cannot mount the new data partition"
# the whole data partition, not only /data: the frugal install is /1bit on this same filesystem (psubdir),
# so copying /data carries the kernel, the initrd, the four SFS, the models and the keys in one go
cp -a "$src_data/." "$DATA_MOUNT/" >> "$LOG" 2>&1 || fail "copying $src_data onto the new data partition failed"
: > "$DATA_MOUNT/.installed"
cp -f "$LOG" "$DATA_MOUNT/install.log" 2> /dev/null || true

say "== 5/7 this disk's own identity in grub.cfg (UUID=$datauuid)"
gcfg=$ESP_MOUNT/EFI/BOOT/grub.cfg
[ -f "$gcfg" ] || fail "the copied ESP has no EFI/BOOT/grub.cfg"
sed -i \
    -e "s|search --no-floppy --label 1BIT-DATA --set=root|search --no-floppy --fs-uuid $datauuid --set=root|" \
    -e "s|pupsfs=1BIT-DATA|pupsfs=UUID=$datauuid 1bit.data=UUID=$datauuid|g" \
    "$gcfg" >> "$LOG" 2>&1 || fail "could not point grub.cfg at this disk"
grep -q "search --no-floppy --fs-uuid $datauuid --set=root" "$gcfg" || fail "grub.cfg does not search this disk's UUID"
n=$(grep -c "pupsfs=UUID=$datauuid 1bit.data=UUID=$datauuid" "$gcfg")
[ "$n" -ge 1 ] || fail "no grub entry pins this disk's data partition (pupsfs=UUID=...)"
say "   $n entr(y/ies) pinned to UUID=$datauuid, root searched by uuid"

say "== 6/7 read back"
sync
for f in EFI/BOOT/BOOTX64.EFI EFI/BOOT/grubx64.efi EFI/BOOT/grub.cfg; do
    [ -e "$ESP_MOUNT/$f" ] || fail "the new ESP is missing $f"
done
a=$(sha256sum "$src_data/1bit/vmlinuz" 2> /dev/null | cut -d' ' -f1)
b=$(sha256sum "$DATA_MOUNT/1bit/vmlinuz" 2> /dev/null | cut -d' ' -f1)
[ -n "$a" ] && [ "$a" = "$b" ] || fail "the kernel on the new disk does not match the source (${a:-missing} vs ${b:-missing})"
for f in initrd.gz puppy_*.sfs; do
    ls "$DATA_MOUNT"/1bit/$f > /dev/null 2>&1 || fail "the new data partition has no 1bit/$f"
done
say "   kernel $b, initrd and the SFS present, .installed written"
umount "$ESP_MOUNT" || fail "cannot flush the new ESP"
umount "$DATA_MOUNT" || fail "cannot flush the new data partition"

say "== 7/7 firmware boot entry (#709)"
# engine#866: `efibootmgr -c` prints the WHOLE updated list, so the first Boot#### in it is whatever already existed -
# on the measured machine Boot0000 BootManagerMenuApp - and BootNext/-Order then pointed at the UEFI menu, which on a
# headless box waits for input for ever. So the entry is created first, then found again by the two things that
# identify THIS one: the label it was created with and this disk's ESP partition GUID, which is what the firmware
# prints in `efibootmgr -v` as HD(1,GPT,<partuuid>,...).
esppart=$(blkid -s PARTUUID -o value "${dev}1" 2> /dev/null | head -1)
[ -n "$esppart" ] || fail "cannot read the ESP's PARTUUID on ${dev}1, so the created entry cannot be identified"
efibootmgr -c -d "$dev" -p 1 -L "1bit OS" -l '\EFI\BOOT\BOOTX64.EFI' >> "$LOG" 2>&1
fwm=$(efibootmgr -v 2> /dev/null)
new=$(printf '%s\n' "$fwm" | sed -n "s/^Boot\([0-9A-Fa-f]\{4\}\)\* 1bit OS.*${esppart}.*/\1/p" | head -1)
if [ -z "$new" ]; then
    # No entry carries this disk's GUID. Either the firmware does not print HD(1,GPT,...) in -v, or the creation did
    # not land - and the two must not be confused, because the wrong answer is the bug this step exists to avoid. A
    # label-only fallback is taken ONLY when the firmware shows no PARTUUID anywhere; a fresh entry is appended, so
    # its id is the last label match.
    if ! printf '%s\n' "$fwm" | grep -q 'HD([0-9]*,GPT,'; then
        new=$(printf '%s\n' "$fwm" | sed -n 's/^Boot\([0-9A-Fa-f]\{4\}\)\* 1bit OS.*/\1/p' | tail -1)
        [ -n "$new" ] && say "   (this firmware prints no partition GUID in efibootmgr -v; matched the label, last entry)"
    fi
fi
[ -n "$new" ] || {
    efibootmgr >> "$LOG" 2>&1; printf '%s\n' "$fwm" >> "$LOG"
    fail "no firmware entry for 1bit OS on $dev (ESP PARTUUID $esppart) - the list is in $LOG"; }
# and it must still be the right one: the id comes from the firmware, not from our own arithmetic
printf '%s\n' "$fwm" | grep -q "^Boot$new\* 1bit OS" ||
    fail "Boot$new is not the 1bit OS entry (list in $LOG)"
order=$(efibootmgr 2> /dev/null | sed -n 's/^BootOrder: //p' | head -1)
rest=$(printf '%s' "$order" | tr ',' '\n' | grep -v "^$new$" | tr '\n' ',' | sed 's/,$//')
efibootmgr -o "$new${rest:+,$rest}" >> "$LOG" 2>&1 || fail "efibootmgr could not set BootOrder"
efibootmgr -n "$new" >> "$LOG" 2>&1 || true
say "   Boot$new = 1bit OS on $dev, first in BootOrder, BootNext"

say ""
say "INSTALLED: 1bit OS on $dev (ESP $ESP_LABEL, data $DATA_LABEL, UUID=$datauuid)"
say "  reboot and remove the stick: the firmware boots Boot$new first."
say "  then: 1bit-release-check   (it says INSTALLED and runs the same rows)"
exit 0
