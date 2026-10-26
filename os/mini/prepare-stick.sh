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
# prepare-stick.sh <image.img | /dev/sdX> [--mint | --key <api key>] [--ssh-pubkey FILE]...
#                   [--model FILE.gguf]... [--check] [--force]
#
# Prepares a 1bit OS stick or image from any Linux box: puts your SSH public keys where Dropbear reads
# them, sets Lemonade's API key, and copies GGUFs into the models directory - then REMOUNTS THE
# PARTITION READ-ONLY and proves what it wrote. Everything it refuses, it refuses before writing.
#
# The data partition is found the way the image describes it, never by a fixed offset: `sfdisk -d`
# reads the GPT, the partition with the Linux filesystem type (and, when there is more than one, the
# one named 1BIT-DATA) is the one, and its start sector is matched against the kernel's own partition
# nodes. The same code path therefore handles an image file (a loop device) and a real stick.
#
# Why the assertions exist: appending to Lemonade's lemond.conf is how the API key gets in, and the
# stock file has ended without a final newline. Appended naively, "LEMONADE_API_KEY=..." glues itself
# to the last line - "#LEMONADE_ADMIN_API_KEY=LEMONADE_API_KEY=xxx" - which Lemonade ignores, so the
# stick looks prepared and answers nothing (candidate 10, 26 Oct). This tool writes a newline first, and
# then checks, read-only, that the file holds exactly one key assignment, that the assignment is the key
# it was given, that there is no stray NUL byte, and that the SSH files have the modes Dropbear demands.
#
#   prepare-stick.sh /dev/sdb --mint --ssh-pubkey ~/.ssh/id_ed25519.pub --model qwen3-0.6b.gguf
#   prepare-stick.sh work/out/1bit-os.img --key "$(openssl rand -hex 24)" --ssh-pubkey my.pub
#   prepare-stick.sh /dev/sdb --check            # assertions only, nothing is written
#
# Exit: 0 prepared and proved; 1 refused or an assertion failed (the reason says which).
set -u

LINUX_FS=0FC63DAF-8483-4772-8E79-3D69D8477DE4
DATA_NAME=1BIT-DATA
MNT=
LOOP=
KEY=
KEYS=""
MODELS=""
CHECK=0
FORCE=0

usage() {
    sed -n '/^# prepare-stick.sh <image/,/^# Exit:/p' "$0" | sed 's/^# \{0,1\}//'
}
die() { printf 'prepare-stick: %s\n' "$*" >&2; exit 1; }

TARGET=
while [ $# -gt 0 ]; do
    case "$1" in
        --key)        shift; [ $# -gt 0 ] || die "--key needs a value"; KEY=$1 ;;
        --mint)       KEY=mint ;;
        --ssh-pubkey) shift; [ $# -gt 0 ] || die "--ssh-pubkey needs a file"; KEYS="$KEYS $1" ;;
        --model)      shift; [ $# -gt 0 ] || die "--model needs a file"; MODELS="$MODELS $1" ;;
        --check)      CHECK=1 ;;
        --force)      FORCE=1 ;;   # do not refuse a partition that is already mounted
        --help|-h)    usage; exit 0 ;;
        -*)           usage >&2; die "unknown option $1" ;;
        *)            [ -z "$TARGET" ] || die "one target, not two ($TARGET and $1)"; TARGET=$1 ;;
    esac
    shift
done

for p in "$TARGET" $KEYS $MODELS; do
    case "$p" in *" "*) die "paths with a space are not supported: $p" ;; esac
done
[ -n "$TARGET" ] || { usage >&2; exit 1; }
[ -e "$TARGET" ] || die "$TARGET does not exist"
[ "$(id -u)" = 0 ] || die "run me as root: it mounts a partition and writes root-owned files into it"
for t in sfdisk findmnt mount umount stat; do
    command -v "$t" >/dev/null 2>&1 || die "$t is missing; install util-linux/coreutils"
done

cleanup() {
    [ -n "${MNT:-}" ] && mountpoint -q "$MNT" 2>/dev/null && umount "$MNT" 2>/dev/null
    [ -n "${MNT:-}" ] && rmdir "$MNT" 2>/dev/null
    [ -n "${LOOP:-}" ] && losetup -d "$LOOP" 2>/dev/null
    return 0
}
trap 'cleanup' EXIT
trap 'cleanup; exit 1' INT TERM HUP

# ---- the data partition, from the GPT ------------------------------------------------------------
# sfdisk -d prints one line per partition: "<node> : start= <n>, size= <n>, type=<guid>, ..., name="x""
# (an image file's node is the image path with the partition number glued on). We keep start and size.
parts=$(sfdisk -d "$TARGET" 2>/dev/null) || die "$TARGET has no GPT partition table (sfdisk -d failed)"
data_lines=$(printf '%s\n' "$parts" | grep -i "type=$LINUX_FS") || true
[ -n "$data_lines" ] || die "no Linux filesystem partition in $TARGET's GPT; is this a 1bit OS stick or image?"

# prefer the partition the image names; if it does not name one, a single Linux-type partition is it
named=$(printf '%s\n' "$data_lines" | grep "name=\"$DATA_NAME\"") || true
count_named=$(printf '%s' "$named" | grep -c 'start=' || true)
count_all=$(printf '%s' "$data_lines" | grep -c 'start=' || true)
if [ "${count_named:-0}" -gt 0 ]; then
    chosen=$named
    [ "$count_named" -eq 1 ] || die "$TARGET has $count_named partitions named $DATA_NAME; refusing to guess"
else
    [ "$count_all" -eq 1 ] || die "$TARGET has $count_all Linux filesystem partitions and none is named $DATA_NAME; refusing to guess"
    chosen=$data_lines
fi
START=$(printf '%s\n' "$chosen" | sed -n 's/.*start=[[:space:]]*\([0-9]\+\).*/\1/p' | head -n 1)
SIZE=$(printf '%s\n' "$chosen" | sed -n 's/.*size=[[:space:]]*\([0-9]\+\).*/\1/p' | head -n 1)
[ -n "$START" ] && [ -n "$SIZE" ] || die "could not read the partition's start/size out of: $chosen"
printf 'target   %s\n' "$TARGET"
printf 'data     Linux filesystem partition at sector %s, %s sectors (%.1f GiB)\n' \
    "$START" "$SIZE" "$(awk -v s="$SIZE" 'BEGIN { printf "%.1f", s * 512 / 1073741824 }')"

# the partition's device node: a real stick publishes it, an image file needs a loop device first
partnode() {   # <disk-node-or-"/dev/loopN"> <start-sector> -> node on stdout
    pn=$1
    for p in "/sys/class/block/${pn##*/}"*/start; do
        [ -e "$p" ] || continue
        if [ "$(cat "$p" 2>/dev/null)" = "$2" ]; then
            echo "/dev/$(basename "$(dirname "$p")")"
            return 0
        fi
    done
    return 1
}

PART=
if [ -b "$TARGET" ]; then
    # a partition node has /sys/class/block/<name>/partition; a whole disk does not (and nvme0n1 ends
    # in a digit, so the name alone cannot tell them apart)
    [ -e "/sys/class/block/${TARGET##*/}/partition" ] &&
        die "$TARGET is a partition; name the whole disk (e.g. /dev/nvme0n1, /dev/sdb)"
    # the kernel may not have published the partition table yet on a freshly plugged stick
    i=0
    while [ $i -lt 10 ]; do
        PART=$(partnode "$TARGET" "$START" || true)
        [ -n "$PART" ] && break
        udevadm settle >/dev/null 2>&1 || sleep 1
        i=$((i + 1))
    done
    [ -n "$PART" ] || die "$TARGET: the kernel never published the partition at sector $START"
else
    command -v losetup >/dev/null 2>&1 || die "losetup is missing; it is needed to mount an image file"
    LOOP=$(losetup -f --show -P "$TARGET" 2>/dev/null) || die "losetup could not attach $TARGET"
    PART=$(partnode "$LOOP" "$START" || true)
    if [ -z "$PART" ]; then
        # no partition scanning on this kernel: point a second loop device straight at the partition,
        # at the offset the GPT gives (never a fixed one)
        losetup -d "$LOOP" 2>/dev/null; LOOP=
        LOOP=$(losetup -f --show --offset $((START * 512)) --sizelimit $((SIZE * 512)) "$TARGET" 2>/dev/null) ||
            die "losetup could not attach the data partition of $TARGET"
        PART=$LOOP
    fi
fi
printf 'node     %s\n' "$PART"

# ---- mount ---------------------------------------------------------------------------------------
if [ "$CHECK" = 1 ]; then MOUNT_OPTS=ro; else MOUNT_OPTS=rw; fi
if [ "$MOUNT_OPTS" = rw ] && [ "$FORCE" = 0 ] && findmnt -S "$PART" -n >/dev/null 2>&1; then
    die "$PART is already mounted; unmount it first (or --force, which does not unmount it for you)"
fi
MNT=$(mktemp -d /tmp/prepare-stick.XXXXXX) || die "mktemp failed"
mount -o "$MOUNT_OPTS" "$PART" "$MNT" 2>/dev/null ||
    die "cannot mount $PART $MOUNT_OPTS (not ext4, or the kernel has no ext4?)"
[ -f "$MNT/lemonade/lemond.conf" ] || [ -d "$MNT/lemonade" ] ||
    printf 'note: %s has no lemonade/ directory yet; this is not a 1bit OS data partition\n' "$PART"

CONF=$MNT/lemonade/lemond.conf
AUTH=$MNT/ssh/authorized_keys
printf 'mounted  %s (%s)\n' "$MNT" "$MOUNT_OPTS"

# ---- the writes ----------------------------------------------------------------------------------
write_key() {   # <key>: put LEMONADE_API_KEY=<key> on a line of its own in lemond.conf
    k=$1
    [ -d "$MNT/lemonade" ] || mkdir -p "$MNT/lemonade"
    [ -f "$CONF" ] || : > "$CONF"
    chown 0:0 "$CONF" 2>/dev/null
    chmod 600 "$CONF" 2>/dev/null
    if grep -q '^LEMONADE_API_KEY=' "$CONF"; then
        # an uncommented assignment is already there: replace its value rather than add a second line
        # (the "exactly one ^LEMONADE_API_KEY=" assertion below would fail otherwise). Read line by
        # line, so a key holding / or & needs no escaping.
        : > "$CONF.new" || die "could not write beside $CONF"
        while IFS= read -r l || [ -n "$l" ]; do
            case "$l" in
                LEMONADE_API_KEY=*) printf 'LEMONADE_API_KEY=%s\n' "$k" ;;
                *) printf '%s\n' "$l" ;;
            esac
        done < "$CONF" > "$CONF.new" || die "could not rewrite $CONF"
        # mv writes a NEW inode: the 0600 the file had is not inherited, it has to be set again
        mv "$CONF.new" "$CONF" || die "could not replace $CONF"
        chown 0:0 "$CONF" 2>/dev/null
        chmod 600 "$CONF" 2>/dev/null
        printf 'key      replaced the LEMONADE_API_KEY= line in lemonade/lemond.conf\n'
    else
        # THE GLUED-LINE TRAP: the stock file ends without a newline, so a bare append lands on the end
        # of the last line and Lemonade never sees the key. Write the newline first, always.
        if [ -s "$CONF" ] && [ "$(tail -c 1 "$CONF" | od -An -c | tr -d ' ')" != '\n' ]; then
            printf '\n' >> "$CONF"
            printf 'key      the file did not end in a newline; wrote one first (the c10 trap)\n'
        fi
        printf 'LEMONADE_API_KEY=%s\n' "$k" >> "$CONF" || die "could not append to $CONF"
        printf 'key      LEMONADE_API_KEY= appended in lemonade/lemond.conf\n'
    fi
}

if [ "$CHECK" = 0 ]; then
    if [ -n "$KEYS" ]; then
        [ -d "$MNT/ssh" ] || mkdir -p "$MNT/ssh"
        chown 0:0 "$MNT/ssh"; chmod 700 "$MNT/ssh"
        [ -f "$AUTH" ] || : > "$AUTH"
        chown 0:0 "$AUTH"; chmod 600 "$AUTH"
        for f in $KEYS; do
            [ -s "$f" ] || die "public key file $f is empty or missing"
            # one key per line, and not twice if this is re-run
            while IFS= read -r line; do
                case "$line" in ''|'#'*) continue ;; esac
                if grep -qxF "$line" "$AUTH"; then
                    printf 'ssh      already present: %s\n' "$(printf '%s' "$line" | cut -c1-40)..."
                else
                    printf '%s\n' "$line" >> "$AUTH" || die "could not write $AUTH"
                    printf 'ssh      added: %s\n' "$(printf '%s' "$line" | cut -c1-40)..."
                fi
            done < "$f"
        done
        chown 0:0 "$AUTH"; chmod 600 "$AUTH"
    fi
    if [ -n "$KEY" ]; then
        [ "$KEY" = mint ] && KEY=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
        case "$KEY" in
            *[![:print:]]*|*" "*) die "the key must be printable and hold no spaces: it goes into lemond.conf as one line" ;;
        esac
        write_key "$KEY"
    fi
    for m in $MODELS; do
        [ -s "$m" ] || die "model $m is empty or missing"
        [ -d "$MNT/models" ] || mkdir -p "$MNT/models"
        cp -f "$m" "$MNT/models/$(basename "$m")" || die "could not copy $m into $MNT/models"
        chown 0:0 "$MNT/models" "$MNT/models/$(basename "$m")" 2>/dev/null
        printf 'model    models/%s (%s bytes)\n' "$(basename "$m")" "$(stat -c %s "$m")"
    done
    sync
    # remount read-only: the assertions below read the stick the way the appliance will, and if one
    # needs to say "what is there", there is nothing left that could still be writing
    umount "$MNT" && mount -o ro "$PART" "$MNT" || die "could not remount $PART read-only"
    printf 'remount  %s read-only for the assertions\n' "$MNT"
fi

# ---- the assertions ------------------------------------------------------------------------------
FAILS=0
ok()   { printf '  PASS  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; FAILS=$((FAILS + 1)); }

printf '\nassertions (%s)\n' "$( [ "$CHECK" = 1 ] && echo --check || echo 'read-only remount' )"

[ -f "$CONF" ] || bad "lemonade/lemond.conf is missing"
if [ -f "$CONF" ]; then
    confbytes=$(stat -c %s "$CONF")
    nonzero=$(tr -d '\000' < "$CONF" | wc -c)
    [ "$confbytes" = "$nonzero" ] && ok "lemond.conf has no NUL bytes ($confbytes bytes)" ||
        bad "lemond.conf has $((confbytes - nonzero)) NUL byte(s); Lemonade would misread it"

    # no key at all is legitimate: init.sh mints one at boot when lemond.conf has no assignment. What
    # is never legitimate is asking for a key and not getting a line of its own (the c10 trap), or two.
    keylines=$(grep -c '^LEMONADE_API_KEY=' "$CONF" || true)
    wantkey=0
    [ -n "${KEY:-}" ] && wantkey=1
    case "$keylines" in
        1) ok "exactly one ^LEMONADE_API_KEY= line" ;;
        0) if [ "$wantkey" = 1 ]; then
               bad "no ^LEMONADE_API_KEY= line: the key is missing, or glued to the previous line (the c10 trap)"
           else
               ok "no key yet (init.sh mints one at boot)"
           fi ;;
        *) bad "$keylines ^LEMONADE_API_KEY= lines; Lemonade would take whichever it reads last" ;;
    esac
    # the glued-line trap, asked of the file itself so it is caught even without knowing the key: the
    # assignment must begin a line (or sit right after a leading "#", as the stock template does).
    # Anything further in - "#LEMONADE_ADMIN_API_KEY=LEMONADE_API_KEY=xxx" - is an append that landed on
    # the end of the previous line, which Lemonade ignores (candidate 10, 26 Oct).
    glued=$(awk 'index($0, "LEMONADE_API_KEY=") > 2 { print NR": "substr($0, 1, 100) }' "$CONF")
    if [ -n "$glued" ]; then
        bad "LEMONADE_API_KEY= is not on a line of its own (the c10 trap): $glued"
    else
        ok "LEMONADE_API_KEY= only ever starts its own line (nothing glued)"
    fi
    if [ "$keylines" = 1 ]; then
        last=$(sed -n 's/^LEMONADE_API_KEY=//p' "$CONF")
        if [ -z "$last" ]; then
            bad "LEMONADE_API_KEY= is empty"
        elif printf '%s' "$last" | grep -q '[^[:print:]]'; then
            bad "LEMONADE_API_KEY= holds a control character (glued or truncated?)"
        elif [ -n "${KEY:-}" ] && [ "$KEY" != mint ] && [ "$last" != "$KEY" ]; then
            bad "LEMONADE_API_KEY= does not equal the key that was asked for"
        else
            ok "LEMONADE_API_KEY= holds the key ($(printf '%s' "$last" | cut -c1-6)...)"
        fi
    fi
fi

if [ -e "$MNT/ssh/authorized_keys" ]; then
    [ "$(stat -c %a "$CONF")" = 600 ] && ok "lemonade/lemond.conf is 0600 (the key is in it)" ||
        bad "lemonade/lemond.conf is $(stat -c %a "$CONF"), wanted 600"
    [ "$(stat -c %a "$MNT/ssh")" = 700 ] && ok "ssh/ is 0700" || bad "ssh/ is $(stat -c %a "$MNT/ssh"), wanted 700"
    [ "$(stat -c %u "$MNT/ssh")" = 0 ] && ok "ssh/ is owned by root" || bad "ssh/ is owned by uid $(stat -c %u "$MNT/ssh"), wanted 0"
    [ "$(stat -c %a "$MNT/ssh/authorized_keys")" = 600 ] && ok "ssh/authorized_keys is 0600" ||
        bad "ssh/authorized_keys is $(stat -c %a "$MNT/ssh/authorized_keys"), wanted 600"
    [ "$(stat -c %u "$MNT/ssh/authorized_keys")" = 0 ] && ok "ssh/authorized_keys is owned by root" ||
        bad "ssh/authorized_keys is owned by uid $(stat -c %u "$MNT/ssh/authorized_keys"), wanted 0"
    nkeys=$(grep -c . "$MNT/ssh/authorized_keys" || true)
    [ "${nkeys:-0}" -ge 1 ] && ok "authorized_keys holds $nkeys key line(s)" || bad "authorized_keys is empty"
elif [ -n "$KEYS" ]; then
    bad "ssh/authorized_keys is missing although --ssh-pubkey was given"
fi

for m in $MODELS; do
    f=$MNT/models/$(basename "$m")
    [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$(stat -c %s "$m")" ] &&
        ok "models/$(basename "$m") is there, same size as the source" ||
        bad "models/$(basename "$m") is missing or a different size"
done

# a stick that would not survive a pull is worth knowing about now, not at the console
printf '\nfree space on the data partition: %s\n' "$(df -h "$MNT" | awk 'NR==2 {print $4}')"
if [ -n "${KEY:-}" ] && [ "$KEY" != mint ]; then printf 'api key (first 6): %s\n' "$(printf '%s' "$KEY" | cut -c1-6)"; fi

if [ "$FAILS" -ne 0 ]; then
    printf '\nprepare-stick: %d assertion(s) failed on %s; this stick would not work as it is\n' "$FAILS" "$PART" >&2
    exit 1
fi
printf '\nprepare-stick: %s is prepared and proved%s\n' "$TARGET" "$( [ "$CHECK" = 1 ] && echo ' (--check: nothing was written)' || echo '' )"
exit 0
