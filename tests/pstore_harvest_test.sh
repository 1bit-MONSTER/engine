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
# engine#3287: `1bit-logs pstore <dir>` files the kernel's persistent-store records of the previous boot into a boot
# directory and only then clears them (a pstore file's removal deletes its EFI variable). Runs on fake records in a
# temporary directory (ONEBIT_PSTORE), so it needs no root, no EFI and no reboot.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
logs=$here/os/v3/puppy/embed/usr/local/bin/1bit-logs
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
run() { ONEBIT_PSTORE=$t/ps sh "$logs" pstore "$t/boot" > "$t/out" 2>&1; }
fresh() { rm -rf "${t:?}/boot"; mkdir -p "$t/boot"; }

# 1. a clean reboot's record: the reset line is in the second part, as efi_pstore splits one dump
mkdir -p "$t/ps"; fresh
printf '<6>[  119.20] usbcore: deregistering interface driver snd-usb-audio\n' > "$t/ps/dmesg-efi_pstore-176382104901001"
printf '<6>[  121.80] r8169 0000:c2:00.0 eno1: Link is Down\n<0>[  122.10] reboot: Restarting system\n' \
    > "$t/ps/dmesg-efi_pstore-176382104901002"
run; check $? "exit 0 with records"
[ -f "$t/boot/pstore/dmesg-efi_pstore-176382104901001" ] && [ -f "$t/boot/pstore/dmesg-efi_pstore-176382104901002" ]
check $? "both parts are filed in the boot directory"
printf '<6>[  121.80] r8169 0000:c2:00.0 eno1: Link is Down\n<0>[  122.10] reboot: Restarting system\n' > "$t/want"
cmp -s "$t/boot/pstore/dmesg-efi_pstore-176382104901002" "$t/want"; check $? "a filed record reads back byte for byte"
[ -z "$(ls -A "$t/ps")" ]; check $? "the store is cleared once filed (NVRAM does not fill)"
grep -q "pstore: 2 record(s) from the previous boot, ends in the kernel's own reset line (reboot: Restarting system)" \
    "$t/boot/markers"
check $? "markers say the kernel finished its reboot"

# 2. a crash
fresh
printf '<0>[   33.00] Kernel panic - not syncing: Fatal exception\n' > "$t/ps/dmesg-efi_pstore-176382200001001"
run; grep -q 'a CRASH: Kernel panic - not syncing: Fatal exception' "$t/boot/markers"; check $? "a panic is named in markers"

# 3. no record at all
fresh
run; grep -q 'pstore: no record from the previous shutdown' "$t/boot/markers"; check $? "an empty store says so"
[ ! -d "$t/boot/pstore" ]; check $? "no empty pstore/ directory is made"

# 4. a record that cannot be copied stays in the store
fresh; mkdir -p "$t/boot/pstore"; chmod 500 "$t/boot/pstore"
printf 'x\n' > "$t/ps/dmesg-efi_pstore-1"
if [ "$(id -u)" != 0 ]; then
    run; [ -f "$t/ps/dmesg-efi_pstore-1" ]; check $? "a record whose copy fails is NOT deleted"
else
    echo "skip a record whose copy fails is NOT deleted (root ignores the mode)"
fi
chmod 700 "$t/boot/pstore"

echo "pstore_harvest_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
