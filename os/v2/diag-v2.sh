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
# diag-v2.sh <rootfs-dir> <data-seed-dir>  -- boot the v2 rootfs and ASK the guest what failed.
# Same disks and guest as vmtest-v2.sh; the only difference is what the test unit runs. Nothing is
# judged here: this is for reading, when a vmtest-v2 assertion fails and the reason is inside the guest.
set -uo pipefail
rootfs=$(cd "$1" && pwd) seed=$(cd "$2" && pwd)
if [ "$(id -u)" != 0 ]; then exec sudo -n "$0" "$@"; fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
kern=$(ls "$rootfs"/boot/vmlinuz-* 2>/dev/null | head -1)
initrd=$(ls "$rootfs"/boot/initrd.img-* 2>/dev/null | head -1)
echo "kernel: $(basename "$kern")"
install -d -m 755 "$rootfs/usr/lib/1bit"
cat > "$rootfs/usr/lib/1bit/vmtest.sh" <<'INGUEST'
#!/bin/sh
exec > /data/vmtest.txt 2>&1
echo "== failed units"
systemctl --failed --no-pager --no-legend
echo "== lemonade unit"
systemctl status 1bit-lemonade.service --no-pager --lines=0 2>&1 | head -12
journalctl -u 1bit-lemonade.service -n 40 --no-pager 2>&1
echo "== ssh unit"
systemctl is-enabled ssh.service 2>&1
systemctl show ssh.service -p ConditionResult -p ConditionTimestamp -p AssertResult -p ActiveState -p UnitFileState 2>&1
systemctl show multi-user.target -p Wants 2>&1 | tr ' ' '\n' | grep -i ssh
systemctl start ssh.service 2>&1; sleep 3
systemctl is-active ssh.service 2>&1
awk 'NR>1 {print $2}' /proc/net/tcp 2>/dev/null | grep -ci ':0016$'
journalctl -b --no-pager 2>&1 | grep -iE 'ssh.service|sshd' | tail -12
journalctl -b --no-pager 2>&1 | grep -ciE 'ordering cycle'
echo "== data"
ls -la /data /data/lemonade /data/lemonade/config /data/lemonade/cache 2>&1
cat /data/lemonade/config/config.json 2>&1
ls -l /data/lemonade/cache/bin 2>&1 | head -5
echo "== lemonade binary"
ldd /opt/lemonade/bin/lemond 2>&1 | grep -E 'not found|=>' | head -25
/opt/lemonade/bin/lemond --help 2>&1 | head -8
echo "== engine binary"
ldd /opt/1bit/1bit 2>&1 | grep 'not found' | head
/opt/1bit/1bit --help 2>&1 | head -5
/opt/1bit/hrx/llama/bin/llama-server --version 2>&1 | head -2
echo "== network"
ip -br a 2>&1
systemctl status systemd-networkd.service --no-pager --lines=0 2>&1 | head -6
echo "== end"
poweroff -f
INGUEST
chmod 755 "$rootfs/usr/lib/1bit/vmtest.sh"
cat > "$rootfs/etc/systemd/system/1bit-vmtest.service" <<'UNIT'
[Unit]
Description=test-only: diagnostics
After=multi-user.target 1bit-lemonade.service 1bit-first-boot.service
Wants=1bit-lemonade.service
[Service]
Type=oneshot
ExecStartPre=/bin/sleep 10
ExecStart=/usr/lib/1bit/vmtest.sh
[Install]
WantedBy=multi-user.target
UNIT
ln -sfn /etc/systemd/system/1bit-vmtest.service \
    "$rootfs/etc/systemd/system/multi-user.target.wants/1bit-vmtest.service"
rootimg=$work/rootfs.img; dataimg=$work/data.img
truncate -s "$(( $(du -sm "$rootfs" | cut -f1) + 2048 ))M" "$rootimg"
mkfs.ext4 -q -F -L 1BIT-ROOT -d "$rootfs" "$rootimg"
truncate -s 8G "$dataimg"
mkfs.ext4 -q -F -L 1BIT-DATA -d "$seed" "$dataimg"
qemu-system-x86_64 -enable-kvm -cpu host -m 8G -smp 8 -machine q35 \
    -kernel "$kern" -initrd "$initrd" \
    -append "root=/dev/vda rw console=ttyS0,115200 loglevel=4" \
    -drive file="$rootimg",if=virtio,format=raw,cache=writeback \
    -drive file="$dataimg",if=virtio,format=raw,cache=writeback \
    -nic user,model=virtio-net-pci \
    -display none -serial file:"$work/serial.log" -no-reboot &
vm=$!
trap 'kill $vm 2>/dev/null; rm -rf "$work"' EXIT
waited=0
while kill -0 $vm 2>/dev/null && [ $waited -lt 420 ]; do sleep 5; waited=$((waited + 5)); done
kill $vm 2>/dev/null
mkdir -p "$work/data"
mount -o ro,loop,noload "$dataimg" "$work/data" 2>/dev/null && {
    cat "$work/data/vmtest.txt" 2>/dev/null; umount "$work/data"; }
