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
# efi-boot.sh: the firmware boot-entry half of 1bit-install (engine #709). Sourced by
# os/mini/install.sh, and exercised by tests/os_mini_efi_boot_test.sh with a fake efibootmgr, so the
# selection and rollback rules run in CI without a device or real NVRAM.
#
# The only entries this removes are ours with the loader file gone and entries whose target GPT
# partition no longer exists anywhere on the machine. An entry on another disk/USB that still has its
# partition is never touched, even if its loader file happens to be missing from our ESP.

EFI_SYSFS_BLOCK=${EFI_SYSFS_BLOCK:-/sys/class/block}

efi_esp_guid() {  # <partition device> -> its GPT PARTUUID, or empty
    sed -n 's/^PARTUUID=//p' "$EFI_SYSFS_BLOCK/${1##*/}/uevent" 2>/dev/null | head -n 1
}

efi_all_part_guids() {
    for u in "$EFI_SYSFS_BLOCK"/*/uevent; do
        [ -r "$u" ] || continue
        sed -n 's/^PARTUUID=//p' "$u" 2>/dev/null
    done
}

efi_load_part_guids() { EFI_ALL_PART_GUIDS=$(efi_all_part_guids); }

efi_guid_exists() {
    printf '%s\n' "${EFI_ALL_PART_GUIDS:-}" | grep -F -x -q "$1"
}

# stdin: efibootmgr -v. Prints 'Boot<hex>|<reason>' for every entry whose loader is provably gone:
#   - it has a GPT partition GUID and a loader path; and
#   - the GUID is this ESP's and the file is missing under <esp_mount>, or
#   - the GUID is no partition on any current block device (the target was deleted).
efi_stale_entries() {  # <esp_guid> <esp_mount>
    esp_guid=$1
    esp_mount=$2
    if [ -z "${EFI_ALL_PART_GUIDS:-}" ]; then efi_load_part_guids; fi
    while IFS= read -r line; do
        case "$line" in Boot[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]*) ;; *) continue ;; esac
        num=$(printf '%s\n' "$line" | sed -n 's/^Boot\([0-9A-Fa-f]\{4\}\).*/\1/p')
        guid=$(printf '%s\n' "$line" | sed -n 's/.*HD([0-9]*,GPT,\([0-9A-Fa-f-]*\).*/\1/p')
        [ -n "$num" ] && [ -n "$guid" ] || continue
        path=$(printf '%s\n' "$line" | sed -n 's/.*File(\([^)]*\)).*/\1/p')
        if [ -z "$path" ]; then
            path=$(printf '%s\n' "$line" | sed -n 's/.*)\/\(.*\)$/\1/p')
        fi
        [ -n "$path" ] || continue
        path=$(printf '%s' "$path" | sed 's/^File(//; s/)$//')
        rel=$(printf '%s' "$path" | tr '\\' '/' | tr 'A-Z' 'a-z')
        case "$rel" in /*) ;; *) rel="/$rel" ;; esac
        if [ -n "$esp_guid" ] && [ "$guid" = "$esp_guid" ]; then
            [ -e "$esp_mount$rel" ] || printf 'Boot%s|loader missing from this ESP: %s\n' "$num" "$path"
        elif ! efi_guid_exists "$guid"; then
            printf 'Boot%s|partition %s no longer exists: %s\n' "$num" "$guid" "$path"
        fi
    done
}

efi_boot_order() { efibootmgr 2>/dev/null | sed -n 's/^BootOrder: //p' | head -n 1; }

# Create '1bit OS' for <dev> partition 1, put it first, set BootNext, remove provably-stale entries.
# On any failure deletes the entry it created and restores the original BootOrder minus the entries it
# removed. Sets EFI_ENTRY_NUM, EFI_BOOT_ORDER, EFI_BOOTNEXT, EFI_REMOVED on success; EFI_ORIG_ORDER.
efi_register_boot_entry() {  # <dev> <esp-mount> <esp-guid>
    EFI_ENTRY_NUM= EFI_BOOT_ORDER= EFI_BOOTNEXT= EFI_REMOVED= EFI_DELETED=
    EFI_ORIG_ORDER=$(efi_boot_order)
    efi_load_part_guids
    stale=$(efibootmgr -v 2>/dev/null | efi_stale_entries "$3" "$2") || true
    old_ifs=$IFS
    IFS='
'
    for one in $stale; do
        ent=${one%%|*}
        why=${one#*|}
        hex=${ent#Boot}
        if efibootmgr -b "$hex" -B >/dev/null 2>&1; then
            EFI_DELETED="${EFI_DELETED}${EFI_DELETED:+,}$hex"
            EFI_REMOVED="${EFI_REMOVED}${EFI_REMOVED:+; }$why"
        fi
    done
    IFS=$old_ifs
    created=$(efibootmgr -c -d "$1" -p 1 -L "1bit OS" -l '\EFI\BOOT\BOOTX64.EFI' 2>>/tmp/install-step.log |
              sed -n 's/^Boot\([0-9A-Fa-f]\{4\}\)\*.*1bit OS.*/\1/p' | head -n 1) || true
    if [ -z "$created" ]; then
        efi_rollback_boot_entry
        return 1
    fi
    EFI_ENTRY_NUM=$created
    order=$(efi_boot_order)
    rest=
    for n in $(printf '%s\n' "$order" | tr ',' ' '); do
        [ "$n" = "$created" ] && continue
        case ",${EFI_DELETED}," in *",$n,"*) continue ;; esac
        rest="$rest${rest:+,}$n"
    done
    if ! efibootmgr -o "$created${rest:+,$rest}" >/dev/null 2>>/tmp/install-step.log; then
        efi_rollback_boot_entry
        return 1
    fi
    if ! efibootmgr -n "$created" >/dev/null 2>>/tmp/install-step.log; then
        efi_rollback_boot_entry
        return 1
    fi
    EFI_BOOTNEXT=$created
    EFI_BOOT_ORDER=$(efi_boot_order)
    return 0
}

# Undo a failed efi_register_boot_entry: remove the entry it created, restore BootOrder to the original
# minus the provably-stale entries removed, and clear a BootNext it set. Never deletes or reorders
# anything that was there before the call.
efi_rollback_boot_entry() {
    [ -n "${EFI_ENTRY_NUM:-}" ] && efibootmgr -b "$EFI_ENTRY_NUM" -B >/dev/null 2>&1
    if [ -n "${EFI_ORIG_ORDER:-}" ]; then
        restored=
        for n in $(printf '%s\n' "$EFI_ORIG_ORDER" | tr ',' ' '); do
            case ",${EFI_DELETED:-}," in *",$n,"*) continue ;; esac
            restored="$restored${restored:+,}$n"
        done
        [ -n "$restored" ] && { efibootmgr -o "$restored" >/dev/null 2>&1 || true; }
    fi
    [ "${EFI_BOOTNEXT:-}" = "${EFI_ENTRY_NUM:-}" ] && efibootmgr -N >/dev/null 2>&1
    EFI_ENTRY_NUM= EFI_BOOTNEXT=
    return 0
}
