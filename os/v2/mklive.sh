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
# mklive.sh <rootfs-dir> <data-seed-dir> <out.img> [mok-key.priv mok-cert.pem]
#
# Writes the live media: one disk image to dd onto a stick, in the same three partitions an installed
# box has, which is what makes 1bit-install a copy rather than a different layout.
#
#   partition 1  512 MB  EFI system partition, 1BIT-OS  - shim, our MOK-signed grub, the kernel and the
#                                                         initramfs, and a grub.cfg for the live entry
#   partition 2  squashfs, PARTLABEL=1BIT-LIVE          - the whole root filesystem, read-only
#   partition 3  ext4, 1BIT-DATA                        - models, Lemonade's backends and config, keys
#
# The live system is the squashfs with a tmpfs overlay (an initramfs-tools script, so nothing is custom
# in userspace): it runs from RAM, the stick is never written, and `apt` on a live boot cannot damage the
# medium. /data stays the stick's own partition, by label, exactly as on an installed box.
#
# Secure Boot: shim (Microsoft-signed) -> grubx64.efi and vmlinuz, both signed with the MOK that is
# enrolled on the machine, so an install copies the same signed chain and needs no enrolment change.
# Without a MOK key pair this still builds, says so loudly, and the image only boots with Secure Boot off.
set -euo pipefail
[ $# -ge 3 ] || { sed -n '/^# mklive.sh/,/^# Secure Boot/p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
if [ "$(id -u)" != 0 ]; then exec sudo -n "$0" "$@"; fi
rootfs=$(cd "$1" && pwd) seed=$(cd "$2" && pwd) out=$3 key=${4:-} cert=${5:-}
here=$(cd "$(dirname "$0")" && pwd)
ESP_MB=512
PART_LIVE=1BIT-LIVE
DATA_LABEL=1BIT-DATA
work=$(mktemp -d)
loop=
# every step of the cleanup has to succeed-or-not without changing the runner's exit status: under
# `set -e` a failing command inside an EXIT trap aborts the trap and becomes the script's status, which
# is how a *successful* image build reported failure (the ESP is already unmounted on the happy path)
trap '{ [ -n "$loop" ] && losetup -d "$loop"; umount "$work/esp"; umount "$work/root/proc"; umount "$work/root/sys"; umount "$work/root/dev"; rm -rf "$work"; } >/dev/null 2>&1 || true' EXIT

# the kernel this stick boots: our own when the rootfs carries it, otherwise the base's
kver=$(ls "$rootfs"/boot/vmlinuz-* 2>/dev/null | sed 's/.*vmlinuz-//' | sort -V | tail -1)
[ -n "$kver" ] || { echo "mklive: no kernel in $rootfs/boot"; exit 2; }
echo "== live tree (a copy: the rootfs this was built from stays as it was)"
cp -a "$rootfs" "$work/root"
rm -rf "$work/root/var/cache/apt/archives"/*.deb "$work/root/tmp"/*
# a test harness's leftovers are not part of a released medium: vmtest-v2.sh writes a test-only unit into
# the rootfs it boots, and a medium built from that rootfs must not carry it (measured: the installed
# disk ran the harness's unit instead of the install check, because the unit had been baked into rootfs6)
rm -f "$work/root"/usr/lib/1bit/vmtest*.sh \
      "$work/root"/usr/lib/systemd/system/1bit-vmtest*.service \
      "$work/root"/etc/systemd/system/1bit-vmtest*.service \
      "$work/root"/etc/systemd/system/multi-user.target.wants/1bit-vmtest*.service
# LIVE_EXTRA is the test harness's hook: a tree laid over the live system (a test-only unit), never part
# of a shipped image
[ -n "${LIVE_EXTRA:-}" ] && cp -a "$LIVE_EXTRA"/. "$work/root/"

# the installer ships in the image, and the mode marker is the live image's (1bit-install removes it -
# 1bit-first-boot.service then writes /etc/1bit/installed after making this machine's own SSH keys)
install -m 755 "$here/install.sh" "$work/root/usr/bin/1bit-install"
mkdir -p "$work/root/etc/1bit"
: > "$work/root/etc/1bit/live"
# the overlay is built in the initramfs, so its modules have to be in there
mkdir -p "$work/root/etc/initramfs-tools"
grep -qx overlay "$work/root/etc/initramfs-tools/modules" 2>/dev/null || echo overlay >> "$work/root/etc/initramfs-tools/modules"
grep -qx squashfs "$work/root/etc/initramfs-tools/modules" 2>/dev/null || echo squashfs >> "$work/root/etc/initramfs-tools/modules"

# --- the initramfs script that makes the squashfs the root ------------------------------------------
# initramfs-tools sources /scripts/init-bottom/* after root= has been mounted and before the final
# run-init, so assigning rootmnt there is what redirects the switch_root. It has to be init-bottom and
# not local-bottom: local_bottom() only runs when a local-top/local-premount script ran, and nothing in
# this image does - measured, the overlay never got built and run-init found the data partition's empty
# root instead. The "0-" prefix sorts it before udev's init-bottom script, which moves /dev into the new
# root and so has to see the overlay already in place.
mkdir -p "$work/root/etc/initramfs-tools/scripts/init-bottom"
cat > "$work/root/etc/initramfs-tools/scripts/init-bottom/0-1bit-live" <<'LIVE'
# 1bit OS live boot: the initramfs has already mounted the squashfs at $rootmnt (root= is the
# 1BIT-LIVE partition, rootfstype=squashfs), so the overlay is that squashfs with a tmpfs upper.
# Nothing here writes the stick.
if grep -q '1bit.live=1' /proc/cmdline; then
    echo "1bit live: building the overlay root"
    [ -e "$rootmnt/etc/issue" ] || panic "1bit live: root= did not mount the 1BIT-LIVE squashfs (empty $rootmnt)"
    mount -t tmpfs -o mode=0755 tmpfs /cow || panic "1bit live: no tmpfs for the overlay"
    mkdir -p /cow/upper /cow/work /live-root
    mount -t overlay -o "lowerdir=$rootmnt,upperdir=/cow/upper,workdir=/cow/work" overlay /live-root ||
        panic "1bit live: the overlay would not mount"
    # A script in here cannot hand a variable back by assigning it: initramfs-tools' ORDER runs each
    # script as a *command*, so its shell is the script's own. /conf/param.conf is the channel - ORDER
    # sources it after every script - and it has to be used, or run-init walks off to the old rootmnt
    # (measured: rootmnt=/live-root was silently ignored and run-init used /root).
    printf 'rootmnt=/live-root\n' >> /conf/param.conf
    echo "1bit live: squashfs=$([ -e "$rootmnt/etc/issue" ] && echo ok || echo empty) overlay=$([ -e /live-root/sbin/init ] && echo ok || echo empty) root=/live-root"
fi
LIVE
chmod 755 "$work/root/etc/initramfs-tools/scripts/init-bottom/0-1bit-live"
echo "== initramfs for $kver (overlay + squashfs in it)"
# initramfs-tools wants a live /proc and /dev to run inside a chroot; without them mkinitramfs fails and
# says so only on its own stderr, which is why its log is kept, not thrown away
for d in proc sys dev; do mount --bind "/$d" "$work/root/$d" || { echo "mklive: cannot bind /$d into the tree"; exit 1; }; done
if ! chroot "$work/root" update-initramfs -c -k "$kver" > "$work/initramfs.log" 2>&1; then
    for d in proc sys dev; do umount "$work/root/$d"; done
    echo "mklive: update-initramfs failed for $kver:"; tail -20 "$work/initramfs.log" | sed 's/^/  /'
    exit 1
fi
for d in proc sys dev; do umount "$work/root/$d"; done
[ -f "$work/root/boot/initrd.img-$kver" ] || { echo "mklive: no initrd for $kver"; exit 1; }

echo "== squashfs"
mksquashfs "$work/root" "$work/live.squashfs" -comp zstd -noappend -no-progress -quiet
echo "   live.squashfs $(du -h "$work/live.squashfs" | cut -f1)"

echo "== the disk image"
sq_mb=$(( $(stat -c %s "$work/live.squashfs") / 1000000 + 64 ))
data_mb=$(( $(du -sm "$seed" | cut -f1) + 256 ))
total_mb=$(( ESP_MB + sq_mb + data_mb + 8 ))
rm -f "$out"
truncate -s "${total_mb}M" "$out"
sgdisk -Z "$out" >/dev/null
sgdisk -n "1:2048:+${ESP_MB}M" -t 1:ef00 -c 1:EFI \
       -n "2:0:+${sq_mb}M" -t 2:8300 -c "2:$PART_LIVE" \
       -n "3:0:0" -t 3:8300 -c "3:$DATA_LABEL" "$out" >/dev/null
loop=$(losetup --find --show --partscan "$out")
i=0
while [ ! -b "${loop}p3" ] && [ "$i" -lt 20 ]; do sleep 1; i=$((i + 1)); done

echo "== the ESP: shim, our grub, the kernel and the initramfs"
esp=$work/esp
mkdir -p "$esp"
mkfs.vfat -F 32 -n 1BIT-OS "${loop}p1" >/dev/null
mount "${loop}p1" "$esp"
# the directories go on the *mounted* filesystem: a mkdir before the mount is hidden by it (measured)
mkdir -p "$esp/EFI/BOOT"
cleanup_esp() { umount "$esp" 2>/dev/null || true; }
install -m 644 "$work/root/boot/vmlinuz-$kver" "$esp/vmlinuz"
install -m 644 "$work/root/boot/initrd.img-$kver" "$esp/initrd.img"

# grub: one standalone binary with the modules the ESP needs; it reads grub.cfg from the ESP, which is
# what 1bit-install rewrites for the installed disk
cat > "$work/root/tmp/grub-embed.cfg" <<'EMBED'
search --no-floppy --file --set=root /EFI/BOOT/grub.cfg
configfile /EFI/BOOT/grub.cfg
EMBED
chroot "$work/root" grub-mkstandalone -O x86_64-efi -o /tmp/grubx64.efi \
    --modules='part_gpt part_msdos fat ext2 normal linux search search_fs_file search_label configfile echo ls' \
    'boot/grub/grub.cfg=/tmp/grub-embed.cfg' >/dev/null 2>&1 ||
    { echo "mklive: grub-mkstandalone failed (grub-efi-amd64-bin in the rootfs?)"; exit 1; }
install -m 644 "$work/root/tmp/grubx64.efi" "$esp/EFI/BOOT/grubx64.efi"

if [ -n "$key" ] && [ -n "$cert" ] && [ -s "$key" ] && [ -s "$cert" ]; then
    for f in "$esp/vmlinuz" "$esp/EFI/BOOT/grubx64.efi"; do
        sbsign --key "$key" --cert "$cert" --output "$f.signed" "$f" >/dev/null
        mv "$f.signed" "$f"
        sbverify --cert "$cert" "$f" >/dev/null || { echo "mklive: $f did not take the signature"; exit 1; }
    done
    echo "   signed with $(openssl x509 -in "$cert" -noout -subject 2>/dev/null | sed 's/^subject=//')"
else
    echo "mklive: WARNING - no MOK key pair: grub and the kernel are UNSIGNED, so this medium only boots" >&2
    echo "mklive: with Secure Boot disabled. Pass <mok.priv> <mok.pem> to sign them." >&2
fi

# shim first (Microsoft-signed), grub beside it: the chain Secure Boot walks. Ubuntu ships the second
# stage for MOK enrolment as plain mmx64.efi and only some versions add a .signed copy, so both are tried.
shim=/usr/lib/shim/shimx64.efi.signed
mmx=/usr/lib/shim/mmx64.efi
[ -f "$work/root/usr/lib/shim/mmx64.efi.signed" ] && mmx=/usr/lib/shim/mmx64.efi.signed
if [ -f "$work/root$shim" ] && [ -f "$work/root$mmx" ]; then
    install -m 644 "$work/root$shim" "$esp/EFI/BOOT/BOOTX64.EFI"
    install -m 644 "$work/root$mmx" "$esp/EFI/BOOT/mmx64.efi"
else
    echo "mklive: WARNING - no shim in the rootfs (shim-signed): BOOTX64.EFI is grub itself" >&2
    install -m 644 "$esp/EFI/BOOT/grubx64.efi" "$esp/EFI/BOOT/BOOTX64.EFI"
fi

cat > "$esp/EFI/BOOT/grub.cfg" <<GRUB
# Written by os/v2/mklive.sh. 1bit-install rewrites this file on the installed disk.
set default=0
set timeout=3
menuentry '1bit OS, Lemonade Server appliance (live)' {
    search --no-floppy --label 1BIT-OS --set=root
    echo "1bit live: root=\$root"
    linux /vmlinuz root=/dev/disk/by-partlabel/$PART_LIVE rootfstype=squashfs ro quiet 1bit.live=1 console=ttyS0,115200
    initrd /initrd.img
}
GRUB
sync
cleanup_esp

echo "== the squashfs, straight onto the partition"
dd if="$work/live.squashfs" of="${loop}p2" bs=4M conv=notrunc status=none
echo "   $(blkid -o value -s TYPE "${loop}p2" 2>/dev/null || echo squashfs) on ${loop}p2"

echo "== the data partition: $(du -sh "$seed" | cut -f1) of models, backends and config"
mkfs.ext4 -q -F -L "$DATA_LABEL" -d "$seed" "${loop}p3"
losetup -d "$loop"; loop=

sgdisk -p "$out" | sed 's/^/  /'
echo
echo "1bit OS v2 live image: $out  ($(du -h "$out" | cut -f1))"
echo "  write it with: dd if=$out of=/dev/sdX bs=4M conv=fsync status=progress"
echo "  then boot the machine from it and run: 1bit-install /dev/nvme0n1 --yes"
sha256sum "$out"
