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
# vmtest-install-v2.sh <rootfs-dir> <data-seed-dir> <work-dir> [mok.priv mok.pem]
#
# Builds the live medium out of <rootfs-dir> (os/v2/mklive.sh), boots it as a machine with a second
# empty disk attached and a USB stick on the bus, runs 1bit-install's refusals and then a real install,
# and finally boots the disk it installed - alone, no live medium attached - to see that machine.
#
# Both halves are judged from what the guest wrote onto the disk it owns (the stick's, then the
# installed disk's /data), not from QEMU. Prints VMTEST-V2-INSTALL PASS or FAIL.
set -uo pipefail
[ $# -ge 3 ] || { sed -n '/^# vmtest-install-v2.sh/,/^# Both halves/p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
if [ "$(id -u)" != 0 ]; then exec sudo -n "$0" "$@"; fi
rootfs=$(cd "$1" && pwd) seed=$(cd "$2" && pwd) work=$3; mkdir -p "$work"; work=$(cd "$work" && pwd)
key=${4:-} cert=${5:-}
here=$(cd "$(dirname "$0")" && pwd)

if [ ! -f /usr/share/OVMF/OVMF_CODE_4M.fd ]; then
    echo "VMTEST-V2-INSTALL FAIL (no OVMF: the live medium must boot through real firmware, not -kernel)"
    exit 1
fi

# --- the test-only unit that lives inside the live medium, and therefore also on the installed disk ---
extra=$(mktemp -d "$work/extra.XXXX"); trap 'rm -rf "$extra"' EXIT
install -d -m 755 "$extra/usr/lib/1bit" "$extra/usr/lib/systemd/system" \
    "$extra/etc/systemd/system/multi-user.target.wants"
cat > "$extra/usr/lib/1bit/vmtest-install.sh" <<'INGUEST'
#!/bin/sh
# Test-only. Two modes, decided by what this machine booted from: the live medium installs; the
# installed disk is then inspected. Everything goes to /data, which is the disk the mode is about.
if [ -e /etc/1bit/live ]; then
    exec > /data/vmtest-install.txt 2>&1
    echo "== this boot"
    grep -q '1bit.live=1' /proc/cmdline && echo "cmdline: live (squashfs + overlay)" || echo "cmdline: NOT live"
    head -2 /etc/issue
    findmnt -n -o SOURCE,FSTYPE / | head -1
    echo "== refusals"
    r() {   # <label> <command...>
        label=$1; shift
        out=$("$@" 2>&1); rc=$?
        echo "--- $label (rc=$rc)"
        echo "$out" | grep -E 'REFUSED|1bit-install:' | head -2
    }
    r "no --yes"            /usr/bin/1bit-install /dev/vdb
    r "a partition"         /usr/bin/1bit-install /dev/vda1 --yes
    # the removable check can only fire on a device the kernel calls removable, and QEMU's usb-storage
    # does not: say what the VM reports, so the row is visible rather than silently passed
    echo "usb disk: /dev/sda removable=$(cat /sys/class/block/sda/removable 2>/dev/null || echo missing)"
    r "a removable device"  /usr/bin/1bit-install /dev/sda --yes
    r "the live disk"       /usr/bin/1bit-install /dev/vda --yes
    mkfs.ext4 -q -F /dev/vdb
    mkdir -p /mnt/t
    mount /dev/vdb /mnt/t
    r "a disk in use"       /usr/bin/1bit-install /dev/vdb --yes
    umount /mnt/t
    mount --bind /dev/null /usr/sbin/sfdisk
    r "no GPT tool"         /usr/bin/1bit-install /dev/vdb --yes
    umount /usr/sbin/sfdisk
    echo "== install"
    /usr/bin/1bit-install /dev/vdb --yes 2>&1 | tail -12
    echo "== the disk it wrote"
    sfdisk -l /dev/vdb 2>&1 | tail -5
    mkdir -p /mnt/e /mnt/r
    mount -t vfat /dev/vdb1 /mnt/e && ls /mnt/e /mnt/e/EFI/BOOT 2>&1
    echo "--- grub.cfg"; sed -n '1,12p' /mnt/e/EFI/BOOT/grub.cfg 2>&1
    umount /mnt/e
    mount -t ext4 /dev/vdb2 /mnt/r && {
        echo "--- root: /boot"; ls /mnt/r/boot 2>&1
        echo "--- root: /etc/fstab"; cat /mnt/r/etc/fstab 2>&1
        echo "--- root: live marker"; [ -e /mnt/r/etc/1bit/live ] && echo "STILL LIVE" || echo "removed"
        echo "--- root: banner"; head -2 /mnt/r/etc/issue 2>&1
        echo "--- root: packages"; chroot /mnt/r dpkg-query -W -f='${Package} ' '1bit-*' 2>/dev/null
    }
    umount /mnt/r 2>/dev/null
    echo "--- data partition"; mkdir -p /mnt/d && mount -t ext4 /dev/vdb3 /mnt/d && ls /mnt/d 2>&1 && umount /mnt/d
    echo "== end"
else
    exec > /data/vmtest.txt 2>&1
    echo "== this boot"
    head -2 /etc/issue
    findmnt -n -o SOURCE,FSTYPE,LABEL /data 2>/dev/null
    echo "== first boot"
    [ -e /etc/1bit/installed ] && echo "/etc/1bit/installed: present" || echo "/etc/1bit/installed: MISSING"
    [ -e /etc/1bit/live ] && echo "/etc/1bit/live: STILL PRESENT" || echo "/etc/1bit/live: gone"
    [ -s /etc/ssh/ssh_host_ed25519_key ] && echo "ssh host key: present" || echo "ssh host key: MISSING"
    echo "== selftest"
    1bit-selftest 2>&1 | grep -E "mode      :|RESULT" | head -3
    echo "== release-check"
    1bit-release-check 2>&1 | grep -E "^(PASS|FAIL|SKIP|RESULT)" | head -20
    echo "== lemonade"
    k=$(sed -n 's/^LEMONADE_API_KEY=//p' /etc/1bit/lemonade.env 2>/dev/null)
    i=0; while [ $i -lt 30 ]; do curl -sf -H "Authorization: Bearer $k" http://127.0.0.1:13305/api/v1/health >/dev/null 2>&1 && break; i=$((i+1)); sleep 2; done
    echo "without the key: HTTP $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:13305/api/v1/models)"
    echo "models: $(curl -s -H "Authorization: Bearer $k" http://127.0.0.1:13305/api/v1/models 2>/dev/null | tr ',' '\n' | sed -n 's/.*"id": *"\([^"]*\)".*/\1/p' | tr '\n' ' ')"
    echo "chat: $(curl -s --max-time 300 -H "Authorization: Bearer $k" -H 'Content-Type: application/json' http://127.0.0.1:13305/api/v1/chat/completions -d '{"model":"Qwen3-0.6B-Q4_K_M","messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' 2>/dev/null | sed -n 's/.*"content": *"\([^"]*\)".*/\1/p' | head -1)"
    echo "== end"
fi
poweroff -f
INGUEST
chmod 755 "$extra/usr/lib/1bit/vmtest-install.sh"
cat > "$extra/usr/lib/systemd/system/1bit-vmtest-install.service" <<'UNIT'
[Unit]
Description=test-only: live install pass, then the installed-disk pass
After=multi-user.target 1bit-lemonade.service 1bit-first-boot.service
Wants=1bit-lemonade.service

[Service]
Type=oneshot
ExecStartPre=/bin/sleep 10
ExecStart=/usr/lib/1bit/vmtest-install.sh

[Install]
WantedBy=multi-user.target
UNIT
ln -sfn /etc/systemd/system/1bit-vmtest-install.service \
    "$extra/etc/systemd/system/multi-user.target.wants/1bit-vmtest-install.service"

# --- the live medium (built with the harness's unit in it) ------------------------------------------
live=$work/live.img
if [ ! -f "$live" ]; then
    echo "== building the live medium"
    LIVE_EXTRA=$extra "$here/mklive.sh" "$rootfs" "$seed" "$live" ${key:+"$key"} ${cert:+"$cert"} |
        sed 's/^/  /' || { echo "VMTEST-V2-INSTALL FAIL (mklive.sh)"; exit 1; }
fi
stick=$work/stick.img
cp --reflink=auto "$live" "$stick"
rm -f "$work/target.img" "$work/usb.img"
truncate -s 40G "$work/target.img"      # a box's disk: the design's 24 GB root must come out of this
truncate -s 2G "$work/usb.img"

copy_ovmf() { cp /usr/share/OVMF/OVMF_CODE_4M.fd "$work/OVMF_CODE.fd"; cp /usr/share/OVMF/OVMF_VARS_4M.fd "$work/vars-$1.fd"; }
copy_ovmf 1

boot() {   # <name> <args...>  -> runs qemu until it powers off, prints the serial tail on failure
    local tag=$1; shift
    qemu-system-x86_64 -enable-kvm -cpu host -m 8G -smp 8 -machine q35 \
        -drive if=pflash,format=raw,readonly=on,file="$work/OVMF_CODE.fd" \
        -drive if=pflash,format=raw,file="$work/vars-$tag.fd" \
        -display none -serial file:"$work/serial-$tag.log" -no-reboot \
        "$@" >/dev/null 2>&1 &
    local vm=$! waited=0
    while kill -0 $vm 2>/dev/null && [ $waited -lt 900 ]; do sleep 5; waited=$((waited + 5)); done
    if kill -0 $vm 2>/dev/null; then kill $vm; echo "  ($tag: the guest did not finish in 15 min)"; return 1; fi
    echo "  ($tag: finished after about ${waited}s)"
    return 0
}

echo "== pass 1: the live medium, with a target disk and a USB stick attached"
boot 1 \
    -drive file="$stick",if=virtio,format=raw,cache=writeback \
    -drive file="$work/target.img",if=virtio,format=raw,cache=writeback \
    -device qemu-xhci -drive file="$work/usb.img",if=none,id=usb0,format=raw \
    -device usb-storage,drive=usb0 \
    -nic user,model=virtio-net-pci
read_back() {   # <image> <file> <partition>
    local img=$1 f=$2 part=$3 m
    m=$(mktemp -d)
    losetup --find --show --partscan "$img" > "$m/loop"
    local lo; lo=$(cat "$m/loop")
    mkdir -p "$m/d"
    if mount -o ro "${lo}p$part" "$m/d" 2>/dev/null; then cat "$m/d/$f" 2>/dev/null; umount "$m/d"; fi
    losetup -d "$lo"
    rm -rf "$m"
}
t1=$(read_back "$stick" vmtest-install.txt 3)
printf '%s\n' "$t1" | sed 's/^/  /'

if [ -z "$t1" ]; then
    echo "== the live pass wrote no transcript: its serial log says why (not booting the target disk)"
    sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$work/serial-1.log" | grep -aE 'error:|1bit live:|not found|panic|REFUSED' | tail -12 | sed 's/^/  /'
    echo "VMTEST-V2-INSTALL FAIL"
    exit 1
fi

echo "== pass 2: the installed disk alone (no live medium attached)"
copy_ovmf 2
boot 2 -drive file="$work/target.img",if=virtio,format=raw,cache=writeback \
       -nic user,model=virtio-net-pci
t2=$(read_back "$work/target.img" vmtest.txt 3)
printf '%s\n' "$t2" | sed 's/^/  /'

# --- the verdict ------------------------------------------------------------------------------------
ok=1
have1() { printf '%s\n' "$t1" | grep -q -- "$1"; }
have2() { printf '%s\n' "$t2" | grep -q -- "$1"; }
judge() {   # <ok?> <message>
    if [ "$1" = 1 ]; then echo "$2"; else echo "NOT: $2"; ok=0; fi
}
judge "$(have1 'cmdline: live (squashfs + overlay)' && echo 1 || echo 0)" "live: the medium boots the squashfs with an overlay"
judge "$(have1 'REFUSED: refusing to write /dev/vdb without --yes' && echo 1 || echo 0)" "refused: no --yes"
judge "$(have1 'REFUSED: /dev/vda1 is a partition' && echo 1 || echo 0)" "refused: a partition instead of a disk"
if [ "$(printf '%s\n' "$t1" | sed -n 's/^usb disk: \/dev\/sda removable=//p' | head -1)" = 1 ]; then
    judge "$(have1 'REFUSED: /dev/sda is a removable device' && echo 1 || echo 0)" "refused: a removable device"
else
    echo "(the removable-device refusal is not exercised: QEMU's usb-storage reports removable=0)"
fi
judge "$(have1 'REFUSED: /dev/vda carries this live boot' && echo 1 || echo 0)" "refused: the disk the live boot runs from"
judge "$(have1 'REFUSED: /dev/vdb is in use' && echo 1 || echo 0)" "refused: a disk with something mounted"
judge "$(have1 'REFUSED: sfdisk is missing' && echo 1 || echo 0)" "refused: no GPT tool"
judge "$(have1 '1bit OS is installed on /dev/vdb' && echo 1 || echo 0)" "install: completed on the target disk"
judge "$(have1 '1BIT-ROOT' && have1 '1BIT-DATA' && echo 1 || echo 0)" "install: three partitions (ESP, 1BIT-ROOT, 1BIT-DATA)"
judge "$(have1 'BOOTX64.EFI' && echo 1 || echo 0)" "install: the ESP carries the firmware loader"
judge "$(have1 '/boot' && have1 'vmlinuz-' && echo 1 || echo 0)" "install: /boot has a kernel on the root partition"
judge "$(have1 'LABEL=1BIT-ROOT' && have1 'LABEL=1BIT-DATA' && echo 1 || echo 0)" "install: fstab mounts by label"
judge "$(have1 'removed' && echo 1 || echo 0)" "install: the live marker is gone from the installed root"
judge "$(have1 '1bit-kernel-image-7' && echo 1 || echo 0)" "install: the installed root carries our kernel package"
judge "$(have2 '1bit OS, Lemonade Server appliance' && echo 1 || echo 0)" "boot: INSTALLED machine says what it is"
judge "$(have2 '/etc/1bit/installed: present' && echo 1 || echo 0)" "boot: first-boot marker written"
judge "$(have2 '/etc/1bit/live: gone' && echo 1 || echo 0)" "boot: it does not think it is live"
judge "$(have2 'ssh host key: present' && echo 1 || echo 0)" "boot: own SSH host keys generated"
judge "$(have2 'mode      : INSTALLED' && echo 1 || echo 0)" "boot: 1bit-selftest reports INSTALLED"
judge "$(have2 'ext4' && have2 '1BIT-DATA' && echo 1 || echo 0)" "/data: on the installed disk's own partition"
judge "$(have2 'without the key: HTTP 401' && echo 1 || echo 0)" "Lemonade: refuses without the key"
judge "$(have2 'Qwen3-0.6B-Q4_K_M' && echo 1 || echo 0)" "Lemonade: lists the GGUF from /data"
case "$t2" in
    *"chat: Paris"*) echo "Lemonade: answered a chat";;
    *) echo "NOT: Lemonade answered a chat ($(printf '%s' "$t2" | grep '^chat:' | head -1))"; ok=0;;
esac

[ $ok = 1 ] && echo "VMTEST-V2-INSTALL PASS" || echo "VMTEST-V2-INSTALL FAIL"
[ $ok = 1 ] || exit 1
