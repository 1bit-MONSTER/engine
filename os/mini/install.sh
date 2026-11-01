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
# 1bit-install /dev/nvme0n1 --yes
#
# Installs 1bit OS from this live boot onto ONE internal disk, laid out exactly as the live stick is:
# an EFI system partition (shim + signed grub + vmlinuz/initramfs + grub.cfg, the same files this boot
# came from) and a 1BIT-DATA ext4 partition filling the rest, carrying the stick's models, backends,
# configuration and SSH keys. The installed system boots the same init; it detects /data/.installed and
# says INSTALLED instead of LIVE.
#
# It writes a disk, so every refusal is checked before anything is touched: no --yes, a partition
# instead of a whole disk, a removable device, the disk this live boot is running from, a device with
# anything mounted, or a missing GPT tool. Then it prints the disk and waits 10 s.
#
# Exit: 0 installed, 1 usage, 2 refused (nothing written), 3 failed while writing.
set -u

ESP_MB=512
DATA_LABEL=1BIT-DATA
ESP_LABEL=1BIT-OS
# by absolute path, because this is a busybox shell: with CONFIG_FEATURE_SH_STANDALONE it runs its OWN
# applet for a name it knows before it looks at PATH, so `mke2fs` reached busybox's ext2-only applet
# instead of the e2fsprogs binary the image ships ("invalid option -- 't'", measured in the VM). The
# tools this installer depends on are therefore named, and checked, where mkimage.sh puts them.
SFDISK=/bin/sfdisk
MKE2FS=/bin/mke2fs

new_part() {   # <disk> <1|2> -> the kernel's device name for that partition
    # by name, not by asking the kernel whether the node exists yet: right after sfdisk it does not, and
    # "does /dev/nvme0n1p1 exist?" answered "no" for a disk whose partitions were simply not published
    # yet, which is how this wrote /dev/nvme0n11 (measured in the VM).
    case "${1##*/}" in
        *[0-9]) echo "$1p$2" ;;   # nvme0n1, mmcblk0: a digit glued to a number needs the p
        *)      echo "$1$2" ;;    # sda, vda, hda
    esac
}

usage() {
    # by pattern, not by line number: this file's header moved once (the SPDX line) and the numbered
    # range then printed half a sentence and the code below it
    sed -n '/^# 1bit-install \/dev/,/^# Exit:/p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//'
}

die() { printf '1bit-install: REFUSED: %s\n' "$*" >&2; exit 2; }

DEV=""
YES=0
for a in "$@"; do
    case "$a" in
        --yes|-y) YES=1 ;;
        --help|-h) usage; exit 0 ;;
        -*) printf '1bit-install: unknown option %s\n' "$a" >&2; exit 1 ;;
        *) if [ -z "$DEV" ]; then DEV=$a; else printf '1bit-install: one target disk, not two\n' >&2; exit 1; fi ;;
    esac
done

if [ -z "$DEV" ]; then
    printf '1bit-install: name the disk to install onto, and pass --yes when you mean it.\n\n' >&2
    usage >&2
    exit 1
fi

# --- refusals, before anything is written ------------------------------------------------------------
[ -b "$DEV" ] || die "$DEV is not a block device"
name=${DEV#/dev/}; name=${name##*/}
sys=/sys/class/block/$name
[ -d "$sys" ] || die "$DEV is not a disk this kernel knows"
[ -e "$sys/partition" ] && die "$DEV is a partition; name the whole disk (e.g. /dev/nvme0n1, not /dev/nvme0n1p1)"
[ "$YES" = 1 ] || die "refusing to write $DEV without --yes (this destroys its contents)"
[ "$(cat "$sys/removable" 2>/dev/null || echo 0)" = 1 ] &&
    die "$DEV is a removable device; 1bit OS installs onto an internal disk, and the live stick is left as it is"

# The disk this live boot is running from: /data is mounted from the stick's 1BIT-DATA partition, and its
# ESP (partition 1 of that same disk) holds the signed chain to copy. Ask /proc/mounts, not findfs: once a
# machine has been installed there are TWO 1BIT-DATA partitions and TWO 1BIT-OS ones, the stick's and the
# installed disk's, and findfs may answer with the installed one. In the VM that made the installer refuse
# the target disk as "this live boot's disk" (the disk it had just labelled), which would block every
# reinstall, and it would have copied the wrong ESP.
live=$(awk '$2 == "/data" { print $1; exit }' /proc/mounts)
live_esp=""
if [ -n "$live" ]; then
    lname=${live#/dev/}; lname=${lname##*/}
    # /sys/class/block/sda2/.. resolves to the disk's directory, so its basename is the disk
    ldisk=$(basename "$(readlink -f "/sys/class/block/$lname/.." 2>/dev/null)" 2>/dev/null || true)
    [ "$ldisk" = "$name" ] && die "$DEV carries this live boot's 1BIT-DATA partition; install onto a different disk"
    [ -n "$ldisk" ] && live_esp=$(new_part "/dev/$ldisk" 1)
fi

# anything mounted from this disk (any partition of it) means it is in use
der=$(awk -v d="/dev/$name" 'index($1, d) == 1 { print $1; exit }' /proc/mounts 2>/dev/null)
[ -n "${der:-}" ] && die "$DEV is in use: $der is mounted"

[ -x "$SFDISK" ] || die "sfdisk is missing from this image; cannot lay out a GPT disk"
[ -x "$MKE2FS" ] || die "mke2fs is missing from this image; cannot make the data partition"

# --- what it is about to destroy ---------------------------------------------------------------------
model=$(cat "$sys/device/model" 2>/dev/null | tr -s ' ' | sed 's/^ //;s/ $//')
[ -n "$model" ] || model="(no model reported)"
bytes=$(blockdev --getsize64 "$DEV" 2>/dev/null || echo 0)
gb=$((bytes / 1000000000))

# --- the install screen (#710) -----------------------------------------------------------------------
# The owner, after the first real install: "During the install you need to give the client visual feedback
# of the entire install." Everything below runs through this screen: one line per step, listed up front,
# flipping from pending to running to done with its elapsed time, and on to the next one as it happens. The
# same lines go to ttyS0 (the QEMU install gate reads them) and to /data/install.log on the stick and, once
# it is mounted, on the new disk. A failure stops on the step that failed, with the command's own output -
# never a bare exit, and never a screen that looks the same whether it is working or stuck.
ESC=$(printf '\033')
SCREEN=/dev/tty1
LOG=/data/install.log
STEPS=/tmp/install-steps
STEP_TOTAL=0
STEP_T0=0

log() { echo "$*" >> "$LOG" 2>/dev/null; [ -w /dev/ttyS0 ] && echo "$*" > /dev/ttyS0 2>/dev/null; return 0; }
screen() { [ -w "$SCREEN" ] && printf '%s\n' "$*" > "$SCREEN" 2>/dev/null; return 0; }

steps_add() {   # <title>: a pending step, drawn on the screen below the header
    STEP_TOTAL=$((STEP_TOTAL + 1))
    echo "$STEP_TOTAL|pending|$1" >> "$STEPS"
}

steps_draw() {  # the header and the pending list, then remember where the list starts
    [ -w "$SCREEN" ] || return 0
    printf '%s[H%s[J' "$ESC" "$ESC" > "$SCREEN"
    {
        echo "  1bit OS install"
        echo "  target : $DEV  ($model, ${gb} GB)"
        echo "  layout : GPT - partition 1 = ${ESP_MB} MB EFI system partition ($ESP_LABEL), the rest = ext4 ($DATA_LABEL)"
        echo "  source : this live boot (the ESP it booted from, and its /data)"
        echo
    } >> "$SCREEN"
    printf '%s7' "$ESC" > "$SCREEN"          # DECSC: the cursor stays at the first step row
    while IFS='|' read -r n st title; do screen "  [ ] $title"; done < "$STEPS"
}

step_line() {   # <n> <marker> <text>: rewrite one step's row in place
    [ -w "$SCREEN" ] || return 0
    printf '%s8' "$ESC" > "$SCREEN"          # DECRC: back to the first step row
    i=1
    while [ "$i" -lt "$1" ]; do printf '%s[B' "$ESC" > "$SCREEN"; i=$((i + 1)); done
    printf '\r  %s %s%s[K' "$2" "$3" "$ESC" > "$SCREEN"
    printf '%s8' "$ESC" > "$SCREEN"
}

step_begin() {  # <n>: the step is running; its start time goes in STEP_T0
    step_title=$(sed -n "s/^$1|pending|//p" "$STEPS")
    sed -i "s/^$1|pending|/$1|running|/" "$STEPS"
    step_line "$1" "[..]" "$step_title"
    log "  [$1] running: $step_title"
    STEP_T0=$(date +%s)
}

step_end() {    # <n> [OK|FAILED] [extra]
    step_state=OK
    [ "$#" -ge 2 ] && step_state=$2
    step_extra=
    [ "$#" -ge 3 ] && step_extra=$3
    step_title=$(sed -n "s/^$1|running|//p" "$STEPS")
    sed -i "s/^$1|running|/$1|$step_state|/" "$STEPS"
    el=$(( $(date +%s) - STEP_T0 ))
    if [ "$step_state" = OK ]; then
        step_line "$1" "[ok]" "$step_title  (${el}s${step_extra:+ $step_extra})"
        log "  [$1] done in ${el}s: $step_title${step_extra:+ $step_extra}"
    else
        step_line "$1" "[!!]" "$step_title  (${el}s)"
        log "  [$1] FAILED after ${el}s: $step_title"
    fi
}

fail() {        # <n> <why>: stop on the failing step, with the command's output, and do not touch it again
    step_end "$1" FAILED
    step_out=$(tail -n 12 /tmp/install-step.log 2>/dev/null)
    log "  install stopped at step $1: $2"
    [ -n "$step_out" ] && printf '%s\n' "$step_out" >> "$LOG" 2>/dev/null
    screen ""
    screen "  Install stopped at step $1: $2"
    if [ -n "$step_out" ]; then
        printf '%s\n' "$step_out" | while IFS= read -r l; do screen "    $l"; done
    fi
    screen ""
    screen "  Nothing else was written; the disk is left as the failed step left it."
    exit 3
}

run() {         # <n> <command...>: one step, its output kept, a failure stops the install
    run_n=$1
    shift
    step_begin "$run_n"
    if "$@" > /tmp/install-step.log 2>&1; then step_end "$run_n" OK
    else fail "$run_n" "$*"; fi
}

copy_tree() {       # <n> <src-dir> <dst-dir>: copy a tree, showing MB copied / total, MB/s and ETA
    cp_n=$1; cp_src=$2; cp_dst=$3
    cp_total=$(du -sb "$cp_src" 2>/dev/null | cut -f1); cp_total=${cp_total:-0}
    cp_tmb=$((cp_total / 1048576))
    cp_t0=$(date +%s); cp_done=0; cp_rate=0
    find "$cp_src" -type d > /tmp/install-dirs 2>/dev/null
    while IFS= read -r d; do
        mkdir -p "$cp_dst/${d#"$cp_src"}" || fail "$cp_n" "cannot create $cp_dst/${d#"$cp_src"}"
    done < /tmp/install-dirs
    find "$cp_src" \( -type f -o -type l \) > /tmp/install-files 2>/dev/null
    while IFS= read -r f; do
        rel=${f#"$cp_src"/}
        cp -a "$f" "$cp_dst/$rel" 2>>/tmp/install-step.log || fail "$cp_n" "copying $rel failed"
        [ -L "$f" ] || cp_done=$((cp_done + $(stat -c %s "$cp_dst/$rel" 2>/dev/null || echo 0)))
        el=$(( $(date +%s) - cp_t0 )); [ "$el" -lt 1 ] && el=1
        cp_mb=$((cp_done / 1048576))
        cp_rate=$((cp_mb / el))
        if [ "$cp_rate" -gt 0 ]; then cp_eta=$(( (cp_tmb - cp_mb) / cp_rate ))s; else cp_eta=unknown; fi
        step_line "$cp_n" "[..]" "copying $cp_src -> $cp_dst: ${cp_mb}/${cp_tmb} MB, ${cp_rate} MB/s, ETA $cp_eta"
    done < /tmp/install-files
    rm -f /tmp/install-dirs /tmp/install-files
    cp_summary="$cp_mb MB, ${cp_rate} MB/s"
}

echo "1bit OS install"
echo "  target : $DEV  ($model, ${gb} GB)"
echo "  current contents of $DEV, about to be destroyed:"
# sfdisk, not busybox fdisk: the image carries the real one for the layout, and it is the one tool here
# that can read a GPT back (busybox fdisk is MBR-only and calls a GPT disk unrecognised)
$SFDISK -l "$DEV" 2>/dev/null | sed 's/^/    /'
echo
echo "  Nothing is written for 10 seconds (Ctrl-C stops here, nothing has been touched)."
i=10
while [ "$i" -gt 0 ]; do
    printf '\r  starting in %2d s ' "$i"
    sleep 1
    i=$((i - 1))
done
printf '\r  writing now      \n'

# --- write -------------------------------------------------------------------------------------------
esp=$(new_part "$DEV" 1)
data=$(new_part "$DEV" 2)
esp_mb=$ESP_MB
# #711: this disk gets its own data-partition UUID, and the grub.cfg copied from the stick is pointed at it
# once the copy has been proved identical (step 6). Without this the disk would inherit the stick's UUID and
# the two systems would still be told apart by a label - the defect the first install exposed.
datauuid=$(cat /proc/sys/kernel/random/uuid)

: > "$STEPS"
steps_add "target disk confirmed: $DEV ($model, ${gb} GB)"
steps_add "partition table written (${esp_mb} MB ESP + $DATA_LABEL)"
steps_add "ESP formatted ($ESP_LABEL, vfat)"
steps_add "data partition formatted ($DATA_LABEL, ext4)"
steps_add "boot files copied from the live stick"
steps_add "boot files verified (sha256, read back from the disk)"
steps_add "data partition filled from the live boot"
steps_add ".installed and the key material carried over"
steps_add "firmware boot entry created (1bit OS, first in BootOrder)"
steps_add "final verification (both partitions, read-only)"
steps_draw

# 1: nothing to do - the checks above decided it, and the screen says which disk it is
step_begin 1
step_end 1 OK

# 2: the same script mkimage.sh uses to build the live stick, so an installed disk and a stick are the
# same layout: 2048-sector start, EFI system partition, then the data partition filling the disk
run 2 sh -c "printf 'label: gpt\nstart=2048, size=%s, type=U, name=\"EFI\"\ntype=L, name=\"%s\"\n' $((esp_mb * 2048)) '$DATA_LABEL' | $SFDISK -q '$DEV'"

# the kernel needs a moment to publish the new partitions
i=0
while [ ! -b "$esp" ] || [ ! -b "$data" ]; do
    i=$((i + 1))
    [ "$i" -gt 20 ] && fail 2 "the kernel never published $esp / $data"
    sleep 1
done
sleep 1

run 3 mkdosfs -F 32 -n "$ESP_LABEL" "$esp"
run 4 sh -c "$MKE2FS -q -t ext4 -L '$DATA_LABEL' -U '$datauuid' '$data'"

mkdir -p /tmp/install/esp /tmp/install/data /tmp/install/live-esp
[ -n "$live_esp" ] || live_esp=$(findfs LABEL=$ESP_LABEL 2>/dev/null || true)   # fallback: no /data mount
run 5 sh -c "mount -t vfat '$esp' /tmp/install/esp && mount -t vfat -o ro '$live_esp' /tmp/install/live-esp && cp -a /tmp/install/live-esp/. /tmp/install/esp/"
# the ESP copy is read back from the disk itself: a boot chain that is not byte-identical is not a boot
# chain that boots, and this is the step that would have caught a truncated kernel or initramfs
step_begin 6
ver_bad=0
for f in EFI/BOOT/BOOTX64.EFI EFI/BOOT/grubx64.efi EFI/BOOT/grub.cfg 1bit/vmlinuz 1bit/initramfs.img; do
    [ -e "/tmp/install/live-esp/$f" ] || continue
    a=$(sha256sum "/tmp/install/live-esp/$f" 2>/dev/null | cut -d' ' -f1)
    b=$(sha256sum "/tmp/install/esp/$f" 2>/dev/null | cut -d' ' -f1)
    if [ -z "$a" ] || [ "$a" != "$b" ]; then
        echo "sha256 mismatch on $f: source ${a:-missing}, written ${b:-missing}" >> /tmp/install-step.log
        ver_bad=$((ver_bad + 1))
    else
        log "    $f matches ($a)"
    fi
done
[ "$ver_bad" = 0 ] || fail 6 "$ver_bad boot file(s) do not match the source"
step_end 6 OK

# 7: the data partition. The live ESP is unmounted here rather than earlier, because step 6 read from it.
step_begin 7
# #711: before the ESP goes, point this disk's grub.cfg at THIS disk's data partition. It is done after
# step 6 on purpose: the copy is verified byte-for-byte against the source first, and only then does the
# installed system's own copy differ in the one value that must differ.
if [ -e /tmp/install/esp/EFI/BOOT/grub.cfg ]; then
    sed -i "s/1bit\.data=UUID=[0-9a-fA-F-]*/1bit.data=UUID=$datauuid/g" /tmp/install/esp/EFI/BOOT/grub.cfg ||
        fail 7 "cannot point the installed grub.cfg at this disk's data partition"
    log "    grub.cfg on $esp: 1bit.data=UUID=$datauuid (this disk's data partition)"
fi
sync
umount /tmp/install/live-esp 2>/dev/null || true
umount /tmp/install/esp 2>/dev/null || fail 7 "cannot flush the new ESP"
mount -t ext4 "$data" /tmp/install/data || fail 7 "cannot mount the new data partition"
if grep -q " /data " /proc/mounts; then
    # models, Lemonade's backends/config/cache, the SSH keys, the NPU kernel sets: the appliance the
    # machine already knows, moved to the disk - with progress, because this is the long step
    copy_tree 7 /data /tmp/install/data
    log "  install: copied /data ($(du -sh /data 2>/dev/null | cut -f1)) onto the disk"
    step_end 7 OK "$cp_summary"
else
    log "  install: no /data on this boot; the installed disk gets an empty data partition"
    step_end 7 OK "no /data on this boot"
fi

# 8: this is what makes the next boot say INSTALLED instead of LIVE (os/mini/init.sh). The key material
# travels inside /data, so the step checks it arrived rather than assuming it did.
step_begin 8
: > /tmp/install/data/.installed
if [ -f /data/lemonade/lemond.conf ] && [ ! -f /tmp/install/data/lemonade/lemond.conf ]; then
    echo "lemonade/lemond.conf did not arrive on the new data partition" >> /tmp/install-step.log
    fail 8 "the Lemonade configuration did not arrive on the new data partition"
fi
cp -a "$LOG" /tmp/install/data/install.log 2>/dev/null
if [ -f /data/ssh/authorized_keys ]; then
    [ -s /tmp/install/data/ssh/authorized_keys ] || fail 8 "the SSH keys did not arrive on the new data partition"
    log "    ssh keys carried over ($(grep -c . /tmp/install/data/ssh/authorized_keys) key lines)"
fi
step_end 8 OK

# 9: the firmware boot entry (#709).
# The first real install wrote the ESP, the data partition and /data/.installed and then told the owner to go
# and edit the firmware by hand. On the next boot the AMI firmware tried its stale `ubuntu` entry (the wiped
# system's shim), failed, and fell through to the USB stick - the owner saw a live boot and thought the
# install had failed. So: register \\EFI\\BOOT\\BOOTX64.EFI for this disk, put it first, set BootNext for the
# first reboot, and remove entries whose loader file is gone from this disk.
step_begin 9
if [ ! -d /sys/firmware/efi ]; then
    log "    no EFI firmware on this boot: nothing to register (the disk still boots as \\EFI\\BOOT\\BOOTX64.EFI)"
    step_end 9 OK "no EFI firmware on this boot"
elif ! command -v efibootmgr >/dev/null 2>&1; then
    fail 9 "efibootmgr is missing from this image"
elif [ ! -e /sys/firmware/efi/efivars/BootOrder-8be4df61-93ca-11d2-aa0d-00e098032b8c ]; then
    # this image's kernel tree has no efivarfs module, so no boot entry can be written from here. Say exactly
    # that - and leave the disk bootable the way it is today, by the firmware's own fallback - rather than
    # stopping an otherwise complete install or, worse, claiming an entry that does not exist.
    log "    efivars are not readable on this boot (no efivarfs in this image's kernel): no boot entry written"
    log "    the disk still boots as \\EFI\\BOOT\\BOOTX64.EFI - point the firmware at it, or boot it once"
    step_end 9 OK "no efivarfs in this kernel: firmware fallback"
else
    # the target ESP's partition GUID, so only entries for THIS disk are ever removed
    esp_guid=$(sed -n 's/^PARTUUID=//p' "/sys/class/block/${esp##*/}/uevent" 2>/dev/null | head -n 1)
    mkdir -p /tmp/install/esp-ro
    mount -t vfat -o ro "$esp" /tmp/install/esp-ro 2>/dev/null || fail 9 "cannot mount the new ESP read-only"
    removed=0
    efibootmgr -v 2>/dev/null | while IFS= read -r line; do
        case "$line" in Boot*File\(*\)*) ;; *) continue ;; esac
        num=$(printf '%s\n' "$line" | sed -n 's/^Boot\([0-9A-Fa-f]\{4\}\).*/\1/p')
        file=$(printf '%s\n' "$line" | sed -n 's/.*File(\([^)]*\)).*/\1/p' | tr '\\' '/' | tr 'A-Z' 'a-z')
        # only an entry that points into THIS ESP, and only when the file it names is not there any more
        [ -n "$esp_guid" ] || continue
        case "$line" in *"$esp_guid"*) ;; *) continue ;; esac
        [ -n "$file" ] || continue
        if [ ! -e "/tmp/install/esp-ro/$file" ]; then
            efibootmgr -b "$num" -B >/dev/null 2>&1 && { echo "efibootmgr: removed stale Boot$num ($file)" >> /tmp/install-step.log; removed=$((removed + 1)); }
        fi
    done
    new=$(efibootmgr -c -d "$DEV" -p 1 -L "1bit OS" -l '\EFI\BOOT\BOOTX64.EFI' 2>/tmp/install-step.log |
          sed -n 's/^Boot\([0-9A-Fa-f]\{4\}\)\*.*1bit OS.*/\1/p' | head -n 1)
    [ -n "$new" ] || fail 9 "efibootmgr could not create the 1bit OS entry"
    order=$(efibootmgr 2>/dev/null | sed -n 's/^BootOrder: //p' | head -n 1)
    rest=$(printf '%s\n' "$order" | tr ',' '\n' | grep -v -x -i "$new" | tr '\n' ',' | sed 's/,$//')
    efibootmgr -o "$new${rest:+,$rest}" >/dev/null 2>>/tmp/install-step.log || fail 9 "efibootmgr could not set BootOrder"
    efibootmgr -n "$new" >/dev/null 2>>/tmp/install-step.log || true
    sync
    log "    Boot$new* 1bit OS -> \\EFI\\BOOT\\BOOTX64.EFI on $DEV, BootNext set"
    log "    BootOrder: $(efibootmgr 2>/dev/null | sed -n 's/^BootOrder: //p' | head -n 1)"
    umount /tmp/install/esp-ro 2>/dev/null || true
    step_end 9 OK "Boot$new first"
fi

# 10: both partitions, read-only, the way the firmware will see them
step_begin 10
sync
umount /tmp/install/data 2>/dev/null || fail 10 "cannot flush the new data partition"
mkdir -p /tmp/install/esp-ro /tmp/install/data-ro
mount -t vfat -o ro "$esp" /tmp/install/esp-ro || fail 10 "cannot mount the new ESP read-only"
mount -t ext4 -o ro "$data" /tmp/install/data-ro || fail 10 "cannot mount the new data partition read-only"
log "    ESP: $(find /tmp/install/esp-ro -type f 2>/dev/null | wc -l) files, data: $(find /tmp/install/data-ro -maxdepth 1 2>/dev/null | wc -l) entries"
for f in EFI/BOOT/BOOTX64.EFI 1bit/vmlinuz 1bit/initramfs.img; do
    [ -e "/tmp/install/esp-ro/$f" ] && log "    $f $(sha256sum "/tmp/install/esp-ro/$f" | cut -d' ' -f1)"
done
[ -e /tmp/install/data-ro/.installed ] || fail 10 ".installed is missing on the new data partition"
umount /tmp/install/esp-ro 2>/dev/null || true
umount /tmp/install/data-ro 2>/dev/null || true
step_end 10 OK

echo
echo "1bit OS is installed on $DEV."
echo "  partition 1: $esp  ${esp_mb} MB, EFI system partition ($ESP_LABEL) - shim, grub, vmlinuz, initramfs, grub.cfg"
echo "  partition 2: $data  ext4, $DATA_LABEL - the models, backends, configuration and keys"
echo
echo "  Secure Boot needs no change: the kernel and initramfs are the same MOK-signed files the stick booted."
echo "  To get back to the live stick, boot the stick again - the installed disk is not touched by it."
log "  install complete: 1bit OS is installed on $DEV ($esp, $data)"

# #710: the machine reboots itself into the installed system, after saying so. Driven from a script (the
# QEMU install gate) the reboot is skipped: ONEBIT_INSTALL_NO_REBOOT=1, or no terminal on the console.
if [ "${ONEBIT_INSTALL_NO_REBOOT:-0}" = 1 ] || [ ! -e "$SCREEN" ]; then
    echo "  Remove the stick and boot the machine from this disk (no automatic reboot in this session)."
    exit 0
fi
i=10
while [ "$i" -gt 0 ]; do
    printf '\r  Install complete - remove the stick; rebooting into the installed system in %2d s ' "$i"
    screen "  Install complete - remove the stick; rebooting into the installed system in ${i} s"
    sleep 1
    i=$((i - 1))
done
printf '\r  rebooting now\n'
screen "  rebooting now"
sync
reboot -f 2>/dev/null || poweroff -f
exit 0
