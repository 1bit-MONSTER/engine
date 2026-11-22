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
# engine#3287: os/v3/puppy/embed/usr/local/sbin/1bit-grow-data on a loop-device "stick" - a 4 GiB disk carrying a
# dd-written image's layout (a 1.5 GiB ext4 data partition last, the backup GPT at the end of the IMAGE). The partition
# is MOUNTED while it grows, as on the appliance, and a file written before must read back byte for byte. Needs root and
# loop devices, so it exits 77 (skip) without them.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
grow=$here/os/v3/puppy/embed/usr/local/sbin/1bit-grow-data
[ "$(id -u)" = 0 ] || { echo "grow_data_test: needs root (loop devices): skipped"; exit 77; }
for t in losetup sfdisk mkfs.ext4 resize2fs partx e2fsck; do command -v $t > /dev/null || { echo "no $t: skipped"; exit 77; }; done
t=$(mktemp -d); bad=0
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
img=$t/disk.img
truncate -s 2200M "$img"
printf 'label: gpt\nstart=2048, size=1048576, type=C12A7328-F81F-11D2-BA4B-00A0C93EC93B\nstart=1050624, size=3145728, type=0FC63DAF-8483-4772-8E79-3D69D8477DE4, name="1BIT-DATA"\n' | sfdisk -q "$img"
truncate -s 4G "$img"
L=$(losetup -f --show -P "$img") || { echo "losetup failed: skipped"; rm -rf "$t"; exit 77; }
mkfs.ext4 -q -L 1BIT-DATA "${L}p2"; mkdir -p "$t/m"; mount "${L}p2" "$t/m"
dd if=/dev/urandom of="$t/m/model.bin" bs=1M count=64 status=none; a=$(sha256sum "$t/m/model.bin" | cut -d' ' -f1)
before=$(df -k "$t/m" | awk 'NR == 2 {print $2}')
"$grow" "${L}p2" "$t/m" > "$t/out1"; grep -q "not removable USB" "$t/out1"; check $? "a disk that is not removable USB is not grown"
ONEBIT_GROW_ANY_DISK=1 "$grow" "${L}p2" "$t/m" > "$t/out2"
after=$(df -k "$t/m" | awk 'NR == 2 {print $2}')
[ "$after" -gt $((before + 1800000)) ]; check $? "the mounted filesystem grew to the end of the disk ($before -> $after KiB)"
[ "$(sha256sum "$t/m/model.bin" | cut -d' ' -f1)" = "$a" ]; check $? "a file written before the grow reads back byte for byte"
[ -f "$t/m/.grown" ]; check $? "the marker is written"
ONEBIT_GROW_ANY_DISK=1 "$grow" "${L}p2" "$t/m" > "$t/out3"; [ ! -s "$t/out3" ]; check $? "a second run does nothing"
umount "$t/m"; e2fsck -fn "${L}p2" > /dev/null 2>&1; check $? "the filesystem checks clean after the online grow"
losetup -d "$L"; rm -rf "$t"
echo "grow_data_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
