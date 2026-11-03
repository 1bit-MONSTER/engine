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
# panic-repro.sh <1bit-os-puppy.img> <work-dir> [seconds]
#
# The 2 Nov hardware boot (cycle 10) died at 26.7 s with
#   Kernel panic - not syncing: Attempted to kill init! exitcode=0x00000000   Comm: sh
# after shim -> GRUB -> our signed kernel -> init all worked. A clean exit code on PID 1 means the init
# script EXITED - Puppy's failure path drops to `exec /bin/sh >/dev/console`, and a shell that reads EOF
# exits 0. So the failure to reproduce is "init could not find the main SFS and its rescue shell had no
# input", and the machine that has to show it is this one:
#
#   * our artifact as a USB stick (`usb-storage`), because the appliance's init refuses a virtio disk
#   * a SECOND disk carrying an ESP and an ext4 root, like the board's internal NVMe with the Ubuntu
#     install on it - a bootable-looking ESP that is not ours
#   * an EXTRA usb mass-storage device, like the JetKVM (board: 3-2:1.6, the stick was 4-1), so the init's
#     drive scan has a device to choose wrongly or to stall on
#   * the stick attached by a DELAYED HOT-PLUG (DELAY seconds after firmware starts, default 8), because
#     "the stick appeared after init's search timeout" is one of the two live hypotheses
#   * Secure Boot ON, from sb-vars.sh's store: the board's chain is the one being reproduced, not a
#     permissive one
#
# Exit: 0 the boot reached the appliance (no panic), 1 the panic reproduced / no appliance, 2 the harness
# could not run. On 1 the transcript path is printed with the init's own log tail - that is the evidence
# the board's JetKVM screen scrolled past.
set -uo pipefail
prog=$(basename "$0")
mode=boot
if [ "${1:-}" = "--no-media" ]; then mode=no-media; shift; fi
[ $# -ge 2 ] || { sed -n "/^# $prog/,/^# could not run/p" "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
img=$(cd "$(dirname "$1")" 2>/dev/null && pwd)/$(basename "$1")
work=$2
secs=${3:-180}
delay=${DELAY:-8}
[ -f "$img" ] || { echo "$prog: no such image: $img" >&2; exit 2; }
for t in qemu-system-x86_64 python3 sfdisk mkfs.ext4 mformat mcopy dd; do
    command -v "$t" > /dev/null || { echo "$prog: $t is missing" >&2; exit 2; }
done
mkdir -p "$work" || exit 2
work=$(cd "$work" && pwd)
here=$(cd "$(dirname "$0")" && pwd)

# ---- the board's other storage, built once and kept: no loop devices (this box keeps 20 busy), so the
# filesystems are made into files and dd'd in at their partition offsets.
nvme=$work/board-nvme.img
if [ ! -f "$nvme" ]; then
    echo "$prog: building the board-like second disk (ESP + ext4 root) in $nvme"
    rm -f "$nvme" "$work/esp.fat" "$work/root.ext4" "$work/esp.d" "$work/root.d"
    mkdir -p "$work/esp.d/EFI/ubuntu" "$work/esp.d/EFI/BOOT" "$work/root.d/etc" "$work/root.d/boot"
    # an ESP that, from a boot scan's point of view, is a real one: a shim+grub chain that is not ours
    printf 'not our shim' > "$work/esp.d/EFI/ubuntu/shimx64.efi"
    printf 'not our grub' > "$work/esp.d/EFI/ubuntu/grubx64.efi"
    cp "$work/esp.d/EFI/ubuntu/shimx64.efi" "$work/esp.d/EFI/BOOT/BOOTX64.EFI"
    printf 'NAME="Ubuntu"\nVERSION="26.04.1 LTS (Resolute Raccoon)"\nID=ubuntu\n' > "$work/root.d/etc/os-release"
    printf 'ubuntu kernel placeholder\n' > "$work/root.d/boot/vmlinuz-6.8.0-board"
    # ESP: 512 MiB FAT32 via mtools (no mount); root: 2 GiB ext4 with -d (no mount)
    mformat -i "$work/esp.fat" -C -T $((512 * 1024 * 2)) -F :: || exit 2
    ( cd "$work/esp.d" && mcopy -s -i "$work/esp.fat" EFI :: ) || exit 2
    truncate -s 2G "$work/root.ext4" || exit 2
    mkfs.ext4 -q -F -L ubuntu-root -d "$work/root.d" "$work/root.ext4" || exit 2
    truncate -s 3G "$nvme" || exit 2
    # p1 ESP 512 MiB at 2048, p2 root for the rest - the shape the board's nvme0n1 has
    sfdisk -q "$nvme" <<-'EOF' || exit 2
	label: gpt
	start=2048, size=1048576, type=C12A7328-F81F-11D2-BA4B-00A0C93EC93B, name="EFI"
	type=0FC63DAF-8483-4772-8E79-3D69D8477DE4, name="ubuntu-root"
	EOF
    dd if="$work/esp.fat" of="$nvme" bs=512 seek=2048 conv=notrunc status=none || exit 2
    root_start=$(sfdisk -d "$nvme" | sed -n 's/.*start= *\([0-9]*\),.*name="ubuntu-root".*/\1/p')
    [ "$root_start" ] || { echo "$prog: cannot find the root partition offset" >&2; exit 2; }
    dd if="$work/root.ext4" of="$nvme" bs=512 seek="$root_start" conv=notrunc status=none || exit 2
    rm -rf "$work/esp.d" "$work/root.d" "$work/esp.fat" "$work/root.ext4"
fi
# the JetKVM's virtual media: a small FAT disk that carries no Linux filesystem and no puppy SFS
kvm=$work/jetkvm-media.img
if [ ! -f "$kvm" ]; then
    echo "$prog: building the JetKVM-like USB media in $kvm"
    mkdir -p "$work/kvm.d"
    printf 'jetkvm virtual media\n' > "$work/kvm.d/README.txt"
    mformat -i "$kvm" -C -T $((256 * 1024 * 2)) -F :: || exit 2
    ( cd "$work/kvm.d" && mcopy -i "$kvm" README.txt :: ) || exit 2
    rm -rf "$work/kvm.d"
fi

rm -f "$work/serial.log" "$work/qemu.out" "$work/qmp.sock"
# --no-media: the board's failure ON DEMAND. A copy of the artifact whose kernel command line pins a label
# that does not exist - one drive scan that comes up empty, which is the state the board ended in - so the
# difference between an image that panics and one that logs and keeps its console is measurable instead of
# argued. The copy is booted rw (never snapshot=on here): the rescue writes /init-fail.log onto 1BIT-DATA
# and that file is read back below.
snap=snapshot=on
if [ "$mode" = no-media ]; then
    broken=$work/no-media.img
    if [ ! -f "$broken" ]; then
        echo "$prog: breaking the media pin in a copy of $(basename "$img") -> $broken"
        cp --sparse=always "$img" "$broken" || exit 2
        off=$((2048 * 512))
        mcopy -o -i "$broken@@$off" ::/EFI/BOOT/grub.cfg "$work/grub.cfg.orig" || exit 2
        sed -E 's/(pupsfs=)[^ ]+/\1NO-SUCH-LABEL/' "$work/grub.cfg.orig" > "$work/grub.cfg.broken"
        grep -q 'pupsfs=' "$work/grub.cfg.broken" ||
            sed -i 's/pmedia=usbhd/pmedia=usbhd pupsfs=NO-SUCH-LABEL/' "$work/grub.cfg.broken"
        grep -q 'pupsfs=NO-SUCH-LABEL' "$work/grub.cfg.broken" || {
            echo "$prog: cannot break the media pin (unexpected grub.cfg)" >&2; exit 2; }
        mcopy -o -i "$broken@@$off" "$work/grub.cfg.broken" ::/EFI/BOOT/grub.cfg || exit 2
    fi
    img=$broken
    snap=snapshot=off
    delay=0    # the copy has to be there when the firmware looks, or nothing boots at all
fi
"$here/sb-vars.sh" "$work/OVMF_VARS_SB.fd" > "$work/sb-vars.txt" 2>&1 || {
    cat "$work/sb-vars.txt" >&2
    echo "$prog: cannot build the Secure Boot variable store - refusing to reproduce the board's boot
      with Secure Boot off (that is the state every VM row was in when the signing defect shipped)" >&2
    exit 2; }
grep -E '^   ' "$work/sb-vars.txt"

# DELAY=0 means "the stick is there when the firmware looks", so it goes on the command line: a
# device_add even a second after start is already too late for OVMF, which then answered
# "BdsDxe: No bootable option or device was found" and never reached grub (measured here). A DELAY > 0
# is the hot-plug case: the stick appears while init is already looking.
# The stick is opened snapshot=on: this harness must never write to the artifact (another lane may be
# booting the same 9 GB image at the same time - measured: a plain rw drive made qemu refuse with
# "Failed to get write lock"), and a discarded copy-on-write layer is exactly what a boot test wants.
stickdrv=(-drive "file=$img,format=raw,if=none,id=stick,$snap")
if [ "$delay" = 0 ]; then
# shellcheck disable=SC2054  # qemu arguments carry commas inside a single word
boot=(-drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.secboot.fd
      -drive if=pflash,format=raw,file="$work/OVMF_VARS_SB.fd"
      -machine q35,smm=on -global driver=cfi.pflash01,property=secure,value=on
      -device qemu-xhci,id=xh1 -device qemu-xhci,id=xh2
      -drive file="$kvm",format=raw,if=none,id=kvm -device usb-storage,drive=kvm,bus=xh1.0
      -drive file="$nvme",format=raw,if=none,id=nvme -device nvme,drive=nvme,serial=BOARDNVME
      "${stickdrv[@]}"
      -device usb-storage,drive=stick,bus=xh2.0)
else
# shellcheck disable=SC2054  # same: commas belong to the qemu arguments
boot=(-drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.secboot.fd
      -drive if=pflash,format=raw,file="$work/OVMF_VARS_SB.fd"
      -machine q35,smm=on -global driver=cfi.pflash01,property=secure,value=on
      -device qemu-xhci,id=xh1 -device qemu-xhci,id=xh2
      -drive file="$kvm",format=raw,if=none,id=kvm -device usb-storage,drive=kvm,bus=xh1.0
      -drive file="$nvme",format=raw,if=none,id=nvme -device nvme,drive=nvme,serial=BOARDNVME
      "${stickdrv[@]}")
fi
# shellcheck disable=SC2054  # same: commas belong to the qemu arguments
net=(-netdev user,id=n0 -device virtio-net-pci,netdev=n0)
qemu-system-x86_64 -m 4096 -smp 4 "${boot[@]}" "${net[@]}" -display none -vga std -no-reboot \
    -serial "file:$work/serial.log" -qmp "unix:$work/qmp.sock,server,nowait" > "$work/qemu.out" 2>&1 &
qpid=$!
cleanup() { kill "$qpid" 2> /dev/null; wait "$qpid" 2> /dev/null; }
trap cleanup EXIT

# the delayed hot-plug: the stick arrives after firmware/init have started, which is the board's
# "appeared after the search" case. With DELAY=0 it is on the command line instead (above).
qmpid=0
if [ "$delay" != 0 ]; then
python3 - "$work/qmp.sock" "$delay" > "$work/qmp.log" 2>&1 <<'PY' &
import json, socket, sys, time
sock, delay = sys.argv[1], float(sys.argv[2])
stop = time.time() + 60
s = None
while time.time() < stop:
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock)
        break
    except OSError:
        s = None
        time.sleep(0.2)
if s is None:
    print("no qmp socket"); sys.exit(1)
f = s.makefile("rw", buffering=1)
f.readline()                                   # greeting
f.write(json.dumps({"execute": "qmp_capabilities"}) + "\n"); print(f.readline().strip())
time.sleep(delay)
f.write(json.dumps({"execute": "device_add",
                    "arguments": {"driver": "usb-storage", "drive": "stick",
                                  "id": "stickdev", "bus": "xh2.0"}}) + "\n")
print(f"device_add after {delay}s: " + f.readline().strip())
time.sleep(2)
f.write(json.dumps({"execute": "query-block"}) + "\n")
print(f.readline().strip()[:400])
PY
qmpid=$!
fi

for _ in $(seq 1 "$secs"); do kill -0 "$qpid" 2> /dev/null || break; sleep 1; done
[ "$qmpid" = 0 ] || kill "$qmpid" 2> /dev/null

log=$work/serial.log
[ -f "$log" ] || { echo "$prog: qemu wrote no serial log ($work/qemu.out)" >&2; exit 2; }
[ -s "$log" ] || {
    echo "$prog: qemu produced no serial output - it did not start:" >&2
    sed 's/^/   /' "$work/qemu.out" >&2
    exit 2; }
appliance=0
grep -q "Loading the 'puppy_.*\.sfs' main file\.\.\." "$log" && appliance=1
grep -q 'Setting up services' "$log" && appliance=1
panic=0
grep -qE 'Attempted to kill init|Kernel panic' "$log" && panic=1

echo "---- $prog: $(basename "$img") (DELAY=$delay, ${secs}s)"
if [ "$panic" = 1 ]; then
    echo "PANIC  the board's 26.7 s failure reproduced"
    grep -nE 'Attempted to kill init|Kernel panic|Comm:' "$log"
fi
[ "$appliance" = 1 ] && echo "BOOT   the appliance's userspace was reached (SFS layered / rc.sysinit)"
guard=0
grep -q '1bit: init stopped' "$log" && guard=1
grep -q '1bit rescue' "$log" && guard=1
if [ "$mode" = no-media ]; then
    [ "$guard" = 1 ] && echo "GUARD  init failed as designed and the console stayed (no PID 1 exit)"
    grep -nE '1bit: init stopped|1bit: log written|1bit rescue|pupsfs=NO-SUCH-LABEL|Finding puppy|not found' \
        "$log" | head -8
    if [ "$guard" = 1 ]; then
        # the log the rescue wrote onto the data partition - the file the next boot of a broken stick reads
        lo=$(sudo -n losetup -f --show -P "$img" 2> /dev/null)
        if [ -n "$lo" ]; then
            sudo -n mkdir -p "$work/mnt"
            if sudo -n mount "${lo}p2" "$work/mnt" 2> /dev/null; then
                if [ -s "$work/mnt/init-fail.log" ]; then
                    echo "---- $(basename "$img"):2 /init-fail.log ($(sudo -n stat -c %s "$work/mnt/init-fail.log") B)"
                    sudo -n sed -n '1,10p' "$work/mnt/init-fail.log"
                else
                    echo "NOTE   no /init-fail.log on the data partition (the rescue fell back to the console log)"
                fi
                sudo -n umount "$work/mnt"
            else
                echo "NOTE   could not mount the data partition of the copy to read /init-fail.log" >&2
            fi
            sudo -n losetup -d "$lo" 2> /dev/null
        fi
        exit 0
    fi
    if grep -qE 'Error is too critical|Dropped to initramfs shell|not found\.' "$log"; then
        echo "SHELL  the failure path was reached and left a shell; PID 1 survived only because this"
        echo "       console blocks on read - the board's ttyS0 returned EOF/EIO and the shell exited 0,"
        echo "       which is the panic. An image with the guard cannot depend on that difference."
    else
        echo "STALL  the failure was not reached at all (grub or firmware stopped earlier)"
    fi
    exit 1
fi
if [ "$panic" = 0 ] && [ "$appliance" = 0 ]; then
    echo "STALL  no panic and no appliance - a hang or a grub/firmware stop"
fi
echo "---- last 25 lines of $log"
tail -n 25 "$log" | tr -d '\000'
if [ "$delay" != 0 ] && [ "$mode" = boot ]; then
    echo "---- hint: DELAY=0 $prog $img $work runs it with the stick present from the start"
fi
[ "$panic" = 0 ] && [ "$appliance" = 1 ] && exit 0
exit 1
