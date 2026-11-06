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
# tests/os_mini_efi_boot_test.sh: device-free tests for os/mini/efi-boot.sh (engine #709).
#
# Runs the boot-entry selection and rollback rules against a fake efibootmgr and a fake /sys/class/block,
# so CI exercises them without NVRAM, a firmware, or an install. The QEMU install gate
# (os/mini/vmtest.sh) covers the end-to-end installer with OVMF.
set -eu

HERE=$(cd "$(dirname "$0")/.." && pwd)
. "$HERE/os/mini/efi-boot.sh"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT INT TERM
fail() { echo "FAIL: $*"; exit 1; }
pass() { echo "PASS: $*"; }

ESP_GUID=11111111-1111-1111-1111-111111111111
DATA_GUID=22222222-2222-2222-2222-222222222222
OTHER_GUID=33333333-3333-3333-3333-333333333333
DEAD_GUID=44444444-4444-4444-4444-444444444444

mkdir -p "$TMP/sys/block/sda1" "$TMP/sys/block/sda2" "$TMP/sys/block/sdb1"
printf 'PARTUUID=%s\n' "$ESP_GUID"   > "$TMP/sys/block/sda1/uevent"
printf 'PARTUUID=%s\n' "$DATA_GUID"  > "$TMP/sys/block/sda2/uevent"
printf 'PARTUUID=%s\n' "$OTHER_GUID" > "$TMP/sys/block/sdb1/uevent"
export EFI_SYSFS_BLOCK="$TMP/sys/block"
export EFI_STEP_LOG="$TMP/step.log"

mkdir -p "$TMP/bin" "$TMP/esp/EFI/BOOT" "$TMP/state"
mkdir -p "$TMP/esp/efi/boot"
: > "$TMP/esp/efi/boot/keep.efi"
D="$TMP/state"

cat > "$TMP/bin/efibootmgr" <<'FAKE'
#!/bin/sh
set -u
D=${EFI_FAKE_DIR:?}
entries="$D/entries"
order="$D/order"
next="$D/next"
[ -f "$entries" ] || : > "$entries"
[ -f "$order" ] || : > "$order"
if [ "${1:-}" = "-v" ]; then
    while IFS='|' read -r num label guid path; do
        [ -n "$num" ] || continue
        if [ -n "${guid:-}" ] && [ -n "${path:-}" ]; then
            printf 'Boot%s* %s\tHD(1,GPT,%s,0x800,0x100000)/%s\n' "$num" "$label" "$guid" "$path"
        else
            printf 'Boot%s  %s\tVenHw(99e275e7-75a0-4b37-a2e6-c5385e6c00cb)\n' "$num" "$label"
        fi
    done < "$entries"
    exit 0
fi
if [ "$#" -eq 0 ]; then
    echo "BootCurrent: 0001"
    echo "Timeout: 1 seconds"
    echo "BootOrder: $(cat "$order" 2>/dev/null)"
    [ -s "$next" ] && echo "BootNext: $(cat "$next")"
    while IFS='|' read -r num label guid path; do
        [ -n "$num" ] && echo "Boot$num* $label"
    done < "$entries"
    exit 0
fi
cmd=$1; shift
case "$cmd" in
    -c)
        label=; path=
        while [ "$#" -gt 0 ]; do
            case "$1" in
                -L) label=$2; shift 2 ;;
                -l) path=$2; shift 2 ;;
                *) shift ;;
            esac
        done
        max=0
        while IFS='|' read -r n l g p; do
            [ -n "$n" ] || continue
            v=$(printf '%d' "0x$n")
            [ "$v" -gt "$max" ] && max=$v
        done < "$entries"
        n=$(printf '%04X' $((max + 1)))
        printf '%s|%s|%s|%s\n' "$n" "$label" "${EFI_FAKE_CREATE_GUID:-00000000-0000-0000-0000-000000000000}" "$path" >> "$entries"
        echo "Boot$n* $label"
        ;;
    -o) [ -e "$D/fail_order" ] && exit 1; printf '%s\n' "$1" > "$order" ;;
    -n) [ -e "$D/fail_next" ] && exit 1; printf '%s\n' "$1" > "$next" ;;
    -N) rm -f "$next" ;;
    -b)
        hex=$1; shift
        if [ "${1:-}" = "-B" ]; then
            grep -v "^$hex|" "$entries" > "$entries.tmp" || true
            mv "$entries.tmp" "$entries"
            tr ',' '\n' < "$order" | grep -v -x "$hex" | tr '\n' ',' | sed 's/,$//' > "$order.tmp"
            printf '\n' >> "$order.tmp"
            mv "$order.tmp" "$order"
        fi
        ;;
    *) exit 1 ;;
esac
FAKE
chmod +x "$TMP/bin/efibootmgr"
export PATH="$TMP/bin:$PATH"
export EFI_FAKE_DIR="$D"
export EFI_FAKE_CREATE_GUID="$ESP_GUID"

reset_state() {   # <order>
    rm -rf "$D"; mkdir -p "$D"
    : > "$D/entries"
    printf '%s\n' "$1" > "$D/order"
}
add_entry() {     # <hex> <label> <guid> <path>
    printf '%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" >> "$D/entries"
}

reset_state "0002,0001,0003,0004,0005"
add_entry 0001 ours-missing "$ESP_GUID" '\EFI\BOOT\BOOTX64.EFI'
add_entry 0002 ours-present "$ESP_GUID" '\EFI\BOOT\KEEP.EFI'
add_entry 0003 ubuntu "$DEAD_GUID" '\EFI\ubuntu\shimx64.efi'
add_entry 0004 otherdisk "$OTHER_GUID" '\EFI\BOOT\OTHER.EFI'
add_entry 0005 venhw '' ''
stale=$(efibootmgr -v | efi_stale_entries "$ESP_GUID" "$TMP/esp")
case "$stale" in *Boot0001*) ;; *) fail "our missing loader was not selected: $stale" ;; esac
case "$stale" in *Boot0003*) ;; *) fail "the dead partition entry was not selected: $stale" ;; esac
case "$stale" in *Boot0002*|*Boot0004*|*Boot0005*) fail "a live/foreign entry was selected: $stale" ;; esac
[ "$(printf '%s\n' "$stale" | grep -c '^Boot')" -eq 2 ] || fail "unexpected selections: $stale"
pass "stale selection (ours-missing + gone partition only)"

reset_state "0002,0001,0003"
add_entry 0001 ours-missing "$ESP_GUID" '\EFI\BOOT\BOOTX64.EFI'
add_entry 0002 ours-present "$ESP_GUID" '\EFI\BOOT\KEEP.EFI'
add_entry 0003 ubuntu "$DEAD_GUID" '\EFI\ubuntu\shimx64.efi'
efi_register_boot_entry /dev/sda "$TMP/esp" "$ESP_GUID" || fail "register returned nonzero"
[ -n "$EFI_ENTRY_NUM" ] || fail "no entry number returned"
[ "$(cat "$D/next")" = "$EFI_ENTRY_NUM" ] || fail "BootNext not set to the new entry"
[ "$EFI_BOOT_ORDER" = "$EFI_ENTRY_NUM,0002" ] || fail "BootOrder is $EFI_BOOT_ORDER, want $EFI_ENTRY_NUM,0002"
grep -q "^$EFI_ENTRY_NUM|1bit OS|$ESP_GUID|\\\\EFI\\\\BOOT\\\\BOOTX64.EFI$" "$D/entries" ||
    fail "entry is not '1bit OS' -> \\EFI\\BOOT\\BOOTX64.EFI: $(cat "$D/entries")"
case "$EFI_REMOVED" in *"loader missing"*"partition $DEAD_GUID"*) ;; *) fail "removed reasons incomplete: $EFI_REMOVED" ;; esac
pass "register success (loader path, first, BootNext, stale removed)"

reset_state "0002,0001"
add_entry 0001 ours-missing "$ESP_GUID" '\EFI\BOOT\BOOTX64.EFI'
add_entry 0002 ours-present "$ESP_GUID" '\EFI\BOOT\KEEP.EFI'
: > "$D/fail_order"
if efi_register_boot_entry /dev/sda "$TMP/esp" "$ESP_GUID"; then
    fail "register succeeded although efibootmgr -o failed"
fi
rm -f "$D/fail_order"
[ "$(cat "$D/order")" = "0002" ] || fail "BootOrder not restored: $(cat "$D/order")"
grep -q "^0003|" "$D/entries" && fail "the failed entry was not removed: $(cat "$D/entries")"
[ ! -s "$D/next" ] || fail "BootNext was left set on failure"
pass "rollback (entry removed, BootOrder restored)"

echo "PASS os_mini_efi_boot"
