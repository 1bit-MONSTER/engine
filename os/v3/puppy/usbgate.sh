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
# usbgate.sh <image.img> <work-dir> [timeout-seconds]
#
# The USB-medium row, in two halves that fail for different reasons.
#
# 1. BUDGET - read the initrd out of the image and assert that its partition wait can outlast what this board
#    actually needs. Cycle 26 shipped an image whose initrd waited 15 s; on the board the kernel published
#    sda1/sda2 at **45.7 s** (the stick's SCSI probe sat behind two `usb 3-2: reset high-speed USB device`
#    cycles from the JetKVM's emulated media on the same xHCI controller), so the init printed
#    `Waiting for partition [1BIT-DATA]............... failed` on a medium that was completely healthy and
#    only found it after a 60 s pause. Every cycle-26 gate was green because QEMU publishes the partition at
#    3.7 s - 12 s inside the old budget. A boot row in QEMU cannot see that class, so the class is asserted
#    where it lives: in the artifact's own wait budget.
#
# 2. BOOT - boot the image with its medium on **USB mass storage** (xhci, not virtio, not IDE) and require the
#    appliance to come up on the medium's own kernel and initrd. This catches the other half: a medium whose
#    partition is never found for a *structural* reason (missing usb-storage, an unresolvable pupsfs token, a
#    grub entry that points at the wrong partition).
#
# Rows: `FAIL budget` / `PASS budget`, then `FAIL part`, `FAIL system`, `PASS part`, `PASS system`, `PASS service`.
# exit 0 = both halves passed, 1 = a row failed, 2 = the row could not run.
set -uo pipefail
prog=$(basename "$0")
[ $# -ge 2 ] || { sed -n "/^# $prog </,/^# exit 0/p" "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
img=$(cd "$(dirname "$1")" 2>/dev/null && pwd)/$(basename "$1")
work=$2
secs=${3:-300}
# the board's own measurement is 45.7 s (engine#1159); 90 s is that with the second reset cycle's margin
need_tries=${WAIT_NEED:-90}
[ -r "$img" ] || { echo "$prog: no such image: $img" >&2; exit 2; }
for t in qemu-system-x86_64 losetup debugfs; do
    command -v "$t" > /dev/null || { echo "$prog: $t is missing" >&2; exit 2; }
done
mkdir -p "$work" || exit 2
work=$(cd "$work" && pwd)
rm -f "$work/serial.log" "$work/qemu.out"
cp -f /usr/share/OVMF/OVMF_VARS_4M.fd "$work/vars.fd" || { echo "$prog: no OVMF variable store" >&2; exit 2; }

# ---- 1. the budget, from the image's own initrd ---------------------------------------------------
rc=0
loop=$(sudo -n losetup -r -f --show -P "$img" 2>/dev/null) || loop=
if [ -n "$loop" ]; then
    sudo -n debugfs -R "dump /1bit/initrd.gz $work/initrd.gz" "${loop}p2" > /dev/null 2>&1
    sudo -n losetup -d "$loop" 2>/dev/null
fi
[ -s "$work/initrd.gz" ] || { echo "FAIL budget        cannot read /1bit/initrd.gz out of the image" >&2; exit 1; }
tmp=$(mktemp -d)
( cd "$tmp" && zcat "$work/initrd.gz" | cpio -idm --quiet init ) 2> /dev/null
if [ ! -f "$tmp/init" ]; then
    echo "FAIL budget        the initrd carries no /init"; rm -rf "$tmp"; exit 1
fi
tries=$(sed -n 's/^[[:space:]]*for NUM in \$(seq 1 \([0-9]\+\)); do$/\1/p' "$tmp/init" | head -1)
tries=${tries:-0}
if [ "$tries" -lt "$need_tries" ]; then
    echo "FAIL budget        the initrd's partition wait is $tries s; the board needed ${need_tries}s (measured 45.7 s, engine#1159)"
    rc=1
else
    echo "PASS budget        the initrd's partition wait is ${tries}s (the board needed ${need_tries}s)"
fi
grep -q "1bit: a slow medium is not a failure" "$tmp/init" && \
    echo "PASS budget        the wait says so on the console instead of looking hung" || {
    echo "FAIL budget        a slow medium waits silently (no advisory line in the wait loop)"; rc=1; }
rm -rf "$tmp"
[ "$rc" = 0 ] || exit 1

# ---- 2. boot it from USB mass storage --------------------------------------------------------------
timeout "$secs" qemu-system-x86_64 -m 4096 -smp 4 \
    -machine q35 -display none -vga std -no-reboot -net none \
    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
    -drive if=pflash,format=raw,file="$work/vars.fd" \
    -drive file="$img",format=raw,if=none,id=d0,readonly=on \
    -device qemu-xhci -device usb-storage,drive=d0 \
    -serial file:"$work/serial.log" > "$work/qemu.out" 2>&1 &
qpid=$!
verdict=2
i=0
while [ $i -lt "$secs" ]; do
    sleep 2
    i=$((i + 2))
    if grep -qE "Waiting for partition \[[^]]*\]\.* failed" "$work/serial.log" 2>/dev/null; then verdict=1; break; fi
    if grep -q "onboard-webapp: serving" "$work/serial.log" 2>/dev/null; then verdict=0; break; fi
    kill -0 "$qpid" 2> /dev/null || break
done
kill "$qpid" 2> /dev/null; wait "$qpid" 2> /dev/null
log=$work/serial.log
if [ "$verdict" = 1 ]; then
    echo "FAIL part          the init gave up before the partition appeared: $(grep -o 'Waiting for partition \[[^]]*\]' "$log" | tail -1)"
    echo "     kernel        $(grep -E '\[sda\]|sda: sda' "$log" | tail -2)"
    exit 1
fi
[ "$verdict" = 0 ] || {
    echo "FAIL system        the appliance did not come up from the USB medium in ${secs}s"
    grep -qE "usb-storage|Direct-Access" "$log" || echo "     usb           the medium never enumerated as USB mass storage"
    tail -3 "$log" 2>/dev/null
    exit 1; }
echo "PASS part          $(grep -m1 -o 'Waiting for partition \[[^]]*\]' "$log" || echo 'no wait needed') - found, then: $(grep -m1 'Attached SCSI' "$log" | cut -c1-60)"
echo "PASS system        $(grep -m1 'system in RAM after' "$log" | sed 's/^\*\*\* 1bit: //')"
echo "PASS service       $(grep -m1 'onboard-webapp: serving' "$log" | cut -c1-90)"
exit 0
