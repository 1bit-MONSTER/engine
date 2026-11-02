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
# mkimage.sh <puppy-out-dir> <out-dir> [data-mb]
#
# The appliance as a **disk image**, not an ISO: a GPT with an EFI system partition carrying the shim ->
# grub chain and a `1BIT-DATA` ext4 partition holding the frugal install and everything that persists.
# That is os/mini/mkimage.sh's shape on purpose, so os/mini/prepare-stick.sh (grow 1BIT-DATA, mint the
# API key, add ssh keys, copy models) and the fleet's USB write run on it unchanged. The ISO stays a side
# artifact of build.sh: it is what boots in QEMU without a disk, and it is not what ships.
#
# <puppy-out-dir>  what build.sh wrote: the ISO plus the four SFS (puppy/zdrv/fdrv/bdrv)
# <out-dir>        where 1bit-os-puppy.img lands
# [data-mb]        the 1BIT-DATA partition's size, default 8192; prepare-stick.sh grows it to the stick
#
# On the data partition, at the root:
#   1bit/            vmlinuz, initrd.gz and the four SFS - Puppy's frugal install, and the save folder
#                    Puppy creates lives here too, so U2's persistence is this same partition
#   models/ ssh/ lemonade/{cache,config}/ npu/kernels/ cache/hrx-jit/
#                    os/mini's data contract; /etc/init.d/05onboard-data wires it to /data at boot
#
# Boot entries in grub.cfg: normal, and a "root shell on the serial port" entry (1bit.serial=1) so a
# headless machine is administrable without a monitor.
set -euo pipefail
prog=$(basename "$0")
usage() { sed -n "/^# $prog /,/^# Boot entries/p" "$0" | sed 's/^# \{0,1\}//'; }
[ $# -ge 2 ] || { usage >&2; exit 1; }
src=$(cd "$1" && pwd)
mkdir -p "$2"; out=$(cd "$2" && pwd)
data_mb=${3:-8192}
esp_mb=${ESP_MB:-512}

iso=$(ls "$src"/*.iso 2> /dev/null | head -1)
[ -n "$iso" ] || { echo "$prog: no ISO in $src (run build.sh first)" >&2; exit 1; }
for f in puppy zdrv fdrv bdrv; do
    ls "$src"/${f}_*.sfs > /dev/null 2>&1 || { echo "$prog: no ${f}_*.sfs in $src" >&2; exit 1; }
done
for t in xorriso sfdisk mkfs.ext4 mkfs.vfat mcopy dd python3; do
    command -v "$t" > /dev/null || { echo "$prog: $t is missing" >&2; exit 1; }
done
# the Secure Boot chain: shim signed by a db key, then grub. Same two files os/mini hard-codes, with the
# environment overrides it accepts, because a stick that only boots with Secure Boot off is not release.
shim=${SHIM:-/usr/lib/shim/shimx64.efi.signed}
[ -f "$shim" ] || shim=/boot/efi/EFI/ubuntu/shimx64.efi
grub=${GRUB:-/usr/lib/grub/x86_64-efi-signed/grubx64.efi.signed}
[ -f "$grub" ] || grub=/boot/efi/EFI/ubuntu/grubx64.efi
[ -f "$shim" ] || { echo "$prog: no Microsoft-signed shim (set SHIM=)" >&2; exit 1; }
[ -f "$grub" ] || { echo "$prog: no signed grub (set GRUB=)" >&2; exit 1; }

work=$out/.mkimage; rm -rf "$work"; mkdir -p "$work"
echo "== kernel and initrd out of $(basename "$iso")"
xorriso -osirrox on -indev "$iso" -extract /vmlinuz "$work/vmlinuz" -extract /initrd.gz "$work/initrd.gz" \
    > /dev/null 2>&1
[ -s "$work/vmlinuz" ] || { echo "$prog: cannot extract /vmlinuz from $iso" >&2; exit 1; }
kver=$(file -b "$work/vmlinuz" | sed -n 's/.*version \([^ ]*\).*/\1/p')

echo "== ESP ($esp_mb MB): shim + grub + grub.cfg"
rm -f "$work/efi.part"
mkfs.vfat -F 32 -n 1BIT-OS -C "$work/efi.part" $((esp_mb * 1024)) > /dev/null
esp=$work/esp; mkdir -p "$esp/EFI/BOOT"
cp -L "$shim" "$esp/EFI/BOOT/BOOTX64.EFI"
cp -L "$grub" "$esp/EFI/BOOT/grubx64.efi"
mm=${MOK_MANAGER:-/usr/lib/shim/mmx64.efi}
[ -f "$mm" ] && cp -L "$mm" "$esp/EFI/BOOT/mmx64.efi"
# The kernel and initrd are on the data partition, not on the ESP: grub reads ext4, and the frugal
# install has to sit next to its save folder. search --label is why the label matters.
cat > "$esp/EFI/BOOT/grub.cfg" <<GRUBCFG
set timeout=8
set default=0
# headless: the menu and its input on the serial port as well as the screen, so a machine in a rack can
# pick an entry and watch the boot without a monitor (and the automated VM run can select the shell entry)
serial --unit=0 --speed=115200
terminal_input serial console
terminal_output serial console
search --no-floppy --label 1BIT-DATA --set=root
menuentry "1bit OS (Puppy appliance, ${kver})" {
    linux /1bit/vmlinuz pmedia=usbhd pfix=copy console=tty0 console=ttyS0,115200
    initrd /1bit/initrd.gz
}
menuentry "1bit OS - serial root shell (1bit.serial=1)" {
    linux /1bit/vmlinuz pmedia=usbhd pfix=copy console=tty0 console=ttyS0,115200 1bit.serial=1
    initrd /1bit/initrd.gz
}
menuentry "1bit OS - no X, verbose (pfix=nox)" {
    linux /1bit/vmlinuz pmedia=usbhd pfix=nox,fsck console=tty0 console=ttyS0,115200 1bit.serial=1
    initrd /1bit/initrd.gz
}
GRUBCFG
MTOOLS_SKIP_CHECK=1 mcopy -s -i "$work/efi.part" "$esp"/* ::/ || {
    echo "$prog: copying the ESP tree into the FAT failed" >&2; exit 1; }

echo "== 1BIT-DATA (${data_mb} MB): frugal install + the data contract"
d=$work/data; mkdir -p "$d/1bit" "$d/models" "$d/ssh" "$d/lemonade/cache" "$d/lemonade/config" \
    "$d/npu/kernels" "$d/cache/hrx-jit"
cp -L "$work/vmlinuz" "$work/initrd.gz" "$d/1bit/"
for f in puppy zdrv fdrv bdrv; do cp -L "$src"/${f}_*.sfs "$d/1bit/"; done
cat > "$d/models/README" <<'TXT'
Put GGUF files here, and NPU model directories (a directory with model.q4nx) beside them.
/etc/init.d/05onboard-data registers every one of them with Lemonade under /data/models, so
/api/v1/models lists them; a .q4nx directory is registered with the NPU backend. Models Lemonade
downloads itself go to lemonade/cache.
TXT
rm -f "$work/data.part"
mkfs.ext4 -q -F -L 1BIT-DATA -d "$d" "$work/data.part" $((data_mb))M

img=$out/1bit-os-puppy.img
rm -f "$img"
truncate -s $(( (esp_mb + data_mb + 2) * 1048576 )) "$img"
printf 'label: gpt\nstart=2048, size=%s, type=U, name="EFI"\ntype=L, name="1BIT-DATA"\n' $((esp_mb * 2048)) \
    | sfdisk -q "$img"
dd if="$work/efi.part" of="$img" bs=1M seek=1 conv=notrunc status=none
dd if="$work/data.part" of="$img" bs=1M seek=$((esp_mb + 1)) conv=notrunc status=none
rm -rf "$work"

echo "== $img"
sfdisk -d "$img" 2> /dev/null | sed 's/^/   /'
ls -l "$img"
echo "   kernel: ${kver:-unknown}, ESP ${esp_mb} MB, data ${data_mb} MB"
echo "   write it: sudo dd if=$img of=/dev/sdX bs=4M conv=fsync   (then os/mini/prepare-stick.sh grows"
echo "             1BIT-DATA to fill the stick and writes the API key and ssh keys)"
