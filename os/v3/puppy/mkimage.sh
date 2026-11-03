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
# psubdir=1bit is what tells Puppy's init to look for the SFS in /1bit and to keep its save file (or save
# folder) there too - measured in the initrd: PSUBDIR is both the search directory and SAVE_DIR.
#
# On the data partition, at the root:
#   1bit/            vmlinuz, initrd.gz and the four SFS - Puppy's frugal install, and the save folder
#                    Puppy creates lives here too, so U2's persistence is this same partition
#   models/ ssh/ lemonade/{cache,config}/ npu/kernels/ cache/hrx-jit/
#                    os/mini's data contract; /etc/init.d/05onboard-data wires it to /data at boot
#
# MODELS=<dir>     optional: *.gguf and */model.q4nx copied onto the data partition, so the stick boots
#                  with a model the engine can serve. Cycle 12 ships an HRX (GPU) GGUF this way: without one
#                  the artifact carries only the NPU set, and U4's chat-hrx row has nothing that runs on the
#                  8060S (measured on the 2 Nov board: /v1/models listed one entry and it was the NPU model).
#
# OWNER (2026-11-02 17:4x): headless by DEFAULT. pfix=nox is the whole mechanism - Puppy then never runs
# xwin, /etc/inittab's tty1 entry (plogin -> getty -n -l /bin/autologin) gives a login prompt, and Lemonade,
# the engine and sshd come up exactly as before. The X packages stay in the image; the desktop is entry 1.
#
# Boot entries in grub.cfg, in this order: the DEFAULT is the headless console (pfix=nox: no X, a login on
# tty1 from Puppy's own plogin -> getty chain), then the desktop (JWM, as before), then a root shell on the
# serial port (1bit.serial=1) so a machine with no monitor is administrable.
set -euo pipefail
prog=$(basename "$0")
usage() { sed -n "/^# $prog /,/^# Boot entries/p" "$0" | sed 's/^# \{0,1\}//'; }
[ $# -ge 2 ] || { usage >&2; exit 1; }
src=$(cd "$1" && pwd)
mkdir -p "$2"; out=$(cd "$2" && pwd)
data_mb=${3:-8192}
esp_mb=${ESP_MB:-512}

# <puppy-out-dir> is what build.sh wrote, but a build whose ISO wrapper failed still has everything the
# artifact needs in its sandbox3/build, so take either: an ISO to pull vmlinuz and initrd out of, or those
# two files directly. The appliance ships the disk image; the ISO is only a source of those two files.
iso=$(ls "$src"/*.iso 2> /dev/null | head -1 || true)   # ls exits 2 with no match, and pipefail would end the script
if [ -z "$iso" ]; then
    [ -s "$src/vmlinuz" ] && [ -s "$src/initrd.gz" ] || {
        echo "$prog: $src has neither an ISO nor vmlinuz + initrd.gz (run build.sh first)" >&2; exit 1; }
fi
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
if [ -n "$iso" ]; then
    echo "== kernel and initrd out of $(basename "$iso")"
    xorriso -osirrox on -indev "$iso" -extract /vmlinuz "$work/vmlinuz" -extract /initrd.gz "$work/initrd.gz" \
        > /dev/null 2>&1
    [ -s "$work/vmlinuz" ] || { echo "$prog: cannot extract /vmlinuz from $iso" >&2; exit 1; }
else
    echo "== kernel and initrd taken from the build tree itself"
    cp -L "$src/vmlinuz" "$work/vmlinuz"
    cp -L "$src/initrd.gz" "$work/initrd.gz"
fi
[ -s "$work/initrd.gz" ] || { echo "$prog: the initrd is empty - the build had no cpio" >&2; exit 1; }
# The initrd's /init must carry the 1bit rescue guard (patch-init.sh): on the 2 Nov board boot PID 1 exited
# cleanly through init's `exec /bin/sh` failure path and the kernel panicked at 26.7 s. This is the artifact
# that ships, so it is checked here as well as in build.sh - an image whose init can exit is not shippable.
"$(dirname "$0")/patch-init.sh" --verify "$work/initrd.gz" > /dev/null || {
    echo "$prog: REFUSED: this initrd's /init can exit (PID 1 exit = kernel panic) - patch-init.sh" >&2
    exit 1; }
kver=$(file -b "$work/vmlinuz" | sed -n 's/.*version \([^ ]*\).*/\1/p')
# The image the stick boots has to be signed: with Secure Boot on, shim -> grub verifies it and refuses an
# unsigned one before init (board, 2 Nov: `error: bad shim lock signature`). mk-huge-kernel.sh refuses the
# same thing at packaging time; this is the artifact that actually ships and it costs one sbverify to say so.
command -v sbverify > /dev/null || { echo "$prog: sbverify is missing (apt install sbsigntool)" >&2; exit 1; }
ksig=$(sbverify --list "$work/vmlinuz" 2> /dev/null | sed -n 's/^ - //p' | head -n 1)
[ -n "$ksig" ] || {
    echo "$prog: REFUSED: the kernel image is unsigned - Secure Boot's GRUB refuses it (sign-vmlinuz.sh)" >&2
    exit 1; }
echo "   kernel $kver, signed by $ksig"

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
setcmd="pmedia=usbhd pupsfs=1BIT-DATA psubdir=1bit ttm.pages_limit=31457280 ttm.page_pool_size=31457280 amdgpu.cwsr_enable=0"
cat > "$esp/EFI/BOOT/grub.cfg" <<GRUBCFG
set timeout=8
set default=0
# headless: the menu and its input on the serial port as well as the screen, so a machine in a rack can
# pick an entry and watch the boot without a monitor (and the automated VM run can select the shell entry)
serial --unit=0 --speed=115200
terminal_input serial console
terminal_output serial console
search --no-floppy --label 1BIT-DATA --set=root
# The boot media is PINNED on the kernel command line: pupsfs=1BIT-DATA sets Puppy's PDRV, and with PDRV set
# the init does not scan any drive at all (measured in the initrd: LOOK_PUP stays empty). Before this, the
# 2 Nov board boot scanned the JetKVM's USB mass storage and the internal NVMe's ESP and could not settle on
# one, then dropped to its \`exec /bin/sh\` failure path, whose shell read EOF, exited 0 and panicked PID 1 at
# 26.7 s (panic-repro.sh reproduces the shape). PDRV also gets a 15 s blkid retry, which covers USB enumeration.
#
# The three TTM/amdgpu parameters are the working Ubuntu install's, measured on this box (2026-11-02):
#   ttm.pages_limit=31457280 ttm.page_pool_size=31457280   the GTT ceiling in 4 KiB pages = 120 GiB. An APU
#       serves models out of GTT, so the default ceiling is the difference between a model fitting and
#       Lemonade refusing to load it. The cycle-13 board image booted WITHOUT them while the Ubuntu install
#       it was validated against booted with them.
#   amdgpu.cwsr_enable=0   the reference install's value; the no-SVM trio puts HRX's CWSR save area in GTT
#       (engine#315), and the box that serves HRX today has CWSR off.
# They are on every entry, so a person who picks the desktop or the shell gets the same machine.
menuentry "1bit OS (headless console, ${kver})" {
    linux /1bit/vmlinuz $setcmd pfix=copy,nox console=tty0 console=ttyS0,115200
    initrd /1bit/initrd.gz
}
menuentry "1bit OS (desktop, JWM)" {
    linux /1bit/vmlinuz $setcmd pfix=copy console=tty0 console=ttyS0,115200
    initrd /1bit/initrd.gz
}
menuentry "1bit OS - install to a disk (1bit.installer=1)" {
    linux /1bit/vmlinuz $setcmd pfix=copy,nox console=tty0 console=ttyS0,115200 1bit.installer=1
    initrd /1bit/initrd.gz
}
menuentry "1bit OS - serial root shell (1bit.serial=1)" {
    linux /1bit/vmlinuz $setcmd pfix=copy,nox console=tty0 console=ttyS0,115200 1bit.serial=1
    initrd /1bit/initrd.gz
}
GRUBCFG
# The artifact is what ships, so the menu is checked here and not in the VM only: the headless entry stays the
# default (an appliance boots to its console), the TTM/amdgpu parameters every entry needs for HRX are on the
# line, and the owner's installer entry is in the menu it was asked for.
grep -q 'ttm.pages_limit=31457280 ttm.page_pool_size=31457280 amdgpu.cwsr_enable=0' "$esp/EFI/BOOT/grub.cfg" || {
    echo "$prog: REFUSED: grub.cfg has no ttm/amdgpu parameters (HRX gets no GTT headroom)" >&2; exit 1; }
grep -q 'menuentry "1bit OS - install to a disk (1bit.installer=1)"' "$esp/EFI/BOOT/grub.cfg" || {
    echo "$prog: REFUSED: grub.cfg has no install-to-a-disk entry" >&2; exit 1; }
grep -q 'set default=0' "$esp/EFI/BOOT/grub.cfg" || {
    echo "$prog: REFUSED: the headless entry is no longer the default" >&2; exit 1; }
echo "   menu: headless (default), desktop, install to a disk, serial shell; ttm/amdgpu parameters on all"
MTOOLS_SKIP_CHECK=1 mcopy -s -i "$work/efi.part" "$esp"/* ::/ || {
    echo "$prog: copying the ESP tree into the FAT failed" >&2; exit 1; }

echo "== 1BIT-DATA (${data_mb} MB): frugal install + the data contract"
d=$work/data; mkdir -p "$d/1bit" "$d/models" "$d/ssh" "$d/lemonade/cache" "$d/lemonade/config" \
    "$d/npu/kernels" "$d/cache/hrx-jit"
cp -L "$work/vmlinuz" "$work/initrd.gz" "$d/1bit/"
for f in puppy zdrv fdrv bdrv; do cp -L "$src"/${f}_*.sfs "$d/1bit/"; done
if [ -n "${MODELS:-}" ]; then
    [ -d "$MODELS" ] || { echo "$prog: MODELS=$MODELS is not a directory" >&2; exit 1; }
    n=0
    for f in "$MODELS"/*.gguf; do
        [ -f "$f" ] || continue
        cp -f "$f" "$d/models/" || { echo "$prog: copying $f onto the data partition failed" >&2; exit 1; }
        n=$((n + 1))
    done
    for m in "$MODELS"/*/model.q4nx; do
        [ -f "$m" ] || continue
        dir=$(dirname "$m")
        cp -a "$dir" "$d/models/" || { echo "$prog: copying $dir onto the data partition failed" >&2; exit 1; }
        n=$((n + 1))
    done
    [ "$n" -gt 0 ] || echo "$prog: WARNING: MODELS=$MODELS carried no *.gguf and no */model.q4nx"
    echo "== models: $n entr(y/ies) onto $d/models ($(du -sh "$d/models" | cut -f1))"
fi
cat > "$d/models/README" <<'TXT'
Put GGUF files here, and NPU model directories (a directory with model.q4nx) beside them.
/etc/init.d/05onboard-data registers every one of them with Lemonade under /data/models, so
/api/v1/models lists them; a .q4nx directory is registered with the NPU backend. Models Lemonade
downloads itself go to lemonade/cache.
TXT
need_mb=$(du -sm "$d" | cut -f1)
[ "$need_mb" -lt "$data_mb" ] || {
    echo "$prog: REFUSED: the data partition would need ${need_mb} MB of ${data_mb} MB (raise data-mb)" >&2; exit 1; }
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
# mkimage needs root for mkfs/sfdisk on most boxes; the image it leaves behind is read by qemu, dd and
# prepare-stick as the caller, so hand it over when we were called through sudo (measured: qemu as the
# invoking user got "Could not open ... Permission denied" on a root-owned 644 image).
if [ "$(id -u)" = "0" ] && [ -n "${SUDO_UID:-}" ]; then
    chown "$SUDO_UID:${SUDO_GID:-$SUDO_UID}" "$img" 2> /dev/null || true
fi
ls -l "$img"
echo "   kernel: ${kver:-unknown}, ESP ${esp_mb} MB, data ${data_mb} MB"
echo "   write it: sudo dd if=$img of=/dev/sdX bs=4M conv=fsync   (then os/mini/prepare-stick.sh grows"
echo "             1BIT-DATA to fill the stick and writes the API key and ssh keys)"
