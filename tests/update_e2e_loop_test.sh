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
# engine#3342 end to end: the REAL `1bit-update <release>` against a loop-mounted installed-system disk, then the new
# release "booted" as Puppy boots it (its SFS as the overlay's lower, the save layer's upper on top), and what the
# machine then sees is checked - not the shadow_* functions on their own (tests/update_shadow_test.sh does that).
#
# engine#3374 is the same harness's second job: the NEW slot's GRUB line must carry the NEW release's own kernel
# parameters, with only this machine's keys (the data partition's UUID pin and the slot) from the running entry. The
# installed disk's line carries neither printk.always_kmsg_dump=1 nor log_buf_len=64M; release A's /1bit/cmdline adds
# the first (a parameter a release ADDED), release B's changes log_buf_len 32M -> 64M (a parameter a release CHANGED,
# on entries that already exist - the second update writes over the slot 1bit-install laid down).
# The ESP carries the image's real FOUR-entry menu (headless, desktop, install-to-disk, serial), so the entries the
# script rewrites are found by the index the artifact actually has, and the two modes it does not own are asserted
# untouched: 1bit-update writes a slot's headless and serial entries, not the image's desktop/installer modes.
#
# The disk: GPT, ESP labelled 1BIT-OS with EFI/BOOT/grub.cfg (one slot, /1bit), ext4 1BIT-DATA with /.installed, /1bit
# (vmlinuz, initrd.gz, the OLD release's puppy SFS) and the save layer /1bit/upup64rsave/upper, seeded with what a
# c35 -> v2026.48 machine carried (engine#3342): a hand-edited release script, a whiteout over a release tool, an
# opaque release directory made by the kernel (rm -rf + mkdir through a real overlay), and the machine's own state.
#
# It runs inside a private mount namespace (unshare -m): /data, /proc/cmdline (psubdir=1bit) and /var/log are the
# namespace's own, so nothing of the host is touched, and every loop device is detached on exit.
# Root, loop devices, mksquashfs/unsquashfs, sfdisk, mkfs.vfat/mkfs.ext4 and getfattr/setfattr are needed: without any
# of them it prints why and returns 77 (tools/ci/ctest_skips.txt: CI runs unprivileged).
#
# usage: tests/update_e2e_loop_test.sh [path/to/1bit-update]      (as root)
#        ONEBIT_E2E_CONTROL=<an older 1bit-update>  also runs that script on a fresh disk and reports what it leaves
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
upd=${1:-$here/os/v3/puppy/embed/usr/local/bin/1bit-update}
owned=$here/os/v3/puppy/embed/etc/1bit/release-owned
skip() { echo "SKIP update_e2e_loop: $*"; exit 77; }
[ "$(id -u)" = 0 ] || skip "needs root (loop devices, mounts, a private mount namespace)"
for t in losetup unshare mksquashfs unsquashfs sfdisk mkfs.vfat mkfs.ext4 blkid mknod getfattr setfattr mount; do
    command -v "$t" > /dev/null 2>&1 || skip "needs $t"
done
[ -f "$upd" ] || { echo "FAIL no 1bit-update at $upd"; exit 1; }

# re-enter in a private mount namespace: every mount below is gone when it exits, whatever happens
if [ -z "${ONEBIT_E2E_NS:-}" ]; then
    ONEBIT_E2E_NS=1 exec unshare -m --propagation private sh "$0" "$@"
fi

w=$(mktemp -d "${ONEBIT_E2E_TMP:-/var/tmp}/update-e2e.XXXXXX") || exit 1
loops=
cleanup() {
    for m in $(awk -v w="$w" 'index($2, w) == 1 || $2 == "/data" { print $2 }' /proc/mounts | sort -r); do umount -l "$m" 2> /dev/null; done
    for l in $loops; do losetup -d "$l" 2> /dev/null; done
    rm -rf "$w"
}
trap cleanup EXIT
bad=0
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
mk() { mkdir -p "$(dirname "$1")"; printf '%s\n' "$2" > "$1"; }
# grub_line CFG INDEX: the `linux` line of GRUB entry INDEX (0-based, the way `set default` counts)
grub_line() { awk -v want="$2" 'BEGIN { n = 0; i = 0 } /^[[:space:]]*menuentry/ { i = n; n++ }
                                /^[[:space:]]*linux / && i == want' "$1"; }
# has_tok LINE TOKEN: TOKEN is one of LINE's whitespace-separated kernel parameters
has_tok() { printf '%s\n' "$1" | tr ' ' '\n' | grep -qxF -- "$2"; }
# slot_head CFG SLOT: the `linux` line of the entry slot_index calls SLOT's headless one - the FIRST one, which is
# the entry the update rewrites (the image's desktop entry also carries psubdir= and no 1bit.serial=1)
slot_head() { awk -v slot="$2" '/^[[:space:]]*linux / && $0 ~ ("psubdir=" slot "([[:space:]]|$)") && $0 !~ /1bit\.serial=1/ { print; exit }' "$1"; }
# entry_line CFG TITLE: the `linux` line of the menuentry whose title contains TITLE (the image's own modes)
entry_line() { awk -v t="$2" 'BEGIN { on = 0 } /^[[:space:]]*menuentry/ { on = (index($0, t) > 0) }
                               /^[[:space:]]*linux / && on { print; exit }' "$1"; }

# ── two releases: the same layout, the code changed ─────────────────────────────────────────────────────────────
release_tree() {  # <dir> <tag>
    d=$1
    mk "$d/etc/init.d/05onboard-data" "#!/bin/sh
# 05onboard-data of $2"
    mk "$d/etc/rc.d/rc.shutdown" "rc.shutdown of $2"
    mk "$d/usr/local/bin/1bit-heavy" "#!/bin/sh
# 1bit-heavy of $2"
    mk "$d/opt/1bit-package/1bit/npu/kernels.$2" "the NPU kernel set of $2"
    mk "$d/etc/passwd" "root:x:0:0:release default:/root:/bin/sh"
    mk "$d/etc/1bit/release-identity" "payload=lemonade-onebit-$2"
    mkdir -p "$d/etc/1bit" && cp "$owned" "$d/etc/1bit/release-owned"
    chmod 755 "$d/etc/init.d/05onboard-data" "$d/usr/local/bin/1bit-heavy"
}
release_tree "$w/old" c35
release_tree "$w/new" v2026.99
mksquashfs "$w/old" "$w/old.sfs" -noappend -no-progress -all-root > /dev/null || exit 1
mksquashfs "$w/new" "$w/new.sfs" -noappend -no-progress -all-root > /dev/null || exit 1
mkdir -p "$w/release"
cp "$w/new.sfs" "$w/release/puppy_test.sfs"
echo "kernel of v2026.99" > "$w/release/vmlinuz"; echo "initrd of v2026.99" > "$w/release/initrd.gz"
# engine#3374: the release's own headless boot line, beside the set it ships (mkimage.sh writes /1bit/cmdline). It is
# what the update builds the new slot's entry from; the UUIDs in it are the release image's own and must NOT arrive.
echo "linux /1bit/vmlinuz pmedia=usbhd pupsfs=UUID=release 1bit.data=UUID=release psubdir=1bit log_buf_len=32M printk.always_kmsg_dump=1 pfix=copy,nox console=ttyS0,115200 console=tty0" > "$w/release/cmdline"
( cd "$w/release" && sha256sum vmlinuz initrd.gz puppy_test.sfs > SHA256SUMS )
# release B: the same set with a CHANGED parameter, for the second update (the target slot's entries already exist)
mkdir -p "$w/release2"
cp "$w/new.sfs" "$w/release2/puppy_test.sfs"
echo "kernel of v2026.98" > "$w/release2/vmlinuz"; echo "initrd of v2026.98" > "$w/release2/initrd.gz"
echo "linux /1bit/vmlinuz pmedia=usbhd pupsfs=UUID=release2 1bit.data=UUID=release2 psubdir=1bit log_buf_len=64M printk.always_kmsg_dump=1 pfix=copy,nox console=ttyS0,115200 console=tty0" > "$w/release2/cmdline"
( cd "$w/release2" && sha256sum vmlinuz initrd.gz puppy_test.sfs > SHA256SUMS )

# ── the installed disk ───────────────────────────────────────────────────────────────────────────────────────────
disk() {  # <name>: an installed c35 machine, returns its loop device in $dev
    img=$w/$1.img
    truncate -s 400M "$img"
    printf 'label: gpt\nsize=64M, type=U, name="1BIT-OS"\ntype=L, name="1BIT-DATA"\n' | sfdisk -q "$img" > /dev/null || exit 1
    dev=$(losetup -f --show -P "$img") || exit 1
    loops="$loops $dev"
    for i in 1 2 3 4 5 6 7 8 9 10; do [ -b "${dev}p2" ] && break; sleep 0.2; done
    mkfs.vfat -n 1BIT-OS "${dev}p1" > /dev/null || exit 1
    mkfs.ext4 -q -F -L 1BIT-DATA "${dev}p2" || exit 1
    mkdir -p "$w/esp"
    mount "${dev}p1" "$w/esp" && mkdir -p "$w/esp/EFI/BOOT"
    cat > "$w/esp/EFI/BOOT/grub.cfg" <<'EOF'
set default=0
set timeout=3
menuentry "1bit OS (headless console)" {
    linux /1bit/vmlinuz pmedia=atahd pupsfs=UUID=x 1bit.data=UUID=x psubdir=1bit nvme.quirks=1e49:1081:dmapool_align_512 log_buf_len=32M pfix=nocopy,nox console=ttyS0,115200 console=tty0
    initrd /1bit/initrd.gz
}
menuentry "1bit OS (desktop, JWM)" {
    linux /1bit/vmlinuz pmedia=atahd pupsfs=UUID=x 1bit.data=UUID=x psubdir=1bit nvme.quirks=1e49:1081:dmapool_align_512 log_buf_len=32M pfix=nocopy console=ttyS0,115200 console=tty0
    initrd /1bit/initrd.gz
}
menuentry "1bit OS - install to a disk (1bit.installer=1)" {
    linux /1bit/vmlinuz pmedia=atahd pupsfs=UUID=x 1bit.data=UUID=x psubdir=1bit nvme.quirks=1e49:1081:dmapool_align_512 log_buf_len=32M pfix=nocopy,nox console=ttyS0,115200 console=tty0 1bit.installer=1
    initrd /1bit/initrd.gz
}
menuentry "1bit OS - serial root shell (1bit.serial=1)" {
    linux /1bit/vmlinuz pmedia=atahd pupsfs=UUID=x 1bit.data=UUID=x psubdir=1bit nvme.quirks=1e49:1081:dmapool_align_512 log_buf_len=32M pfix=nocopy,nox console=ttyS0,115200 console=tty0 1bit.serial=1
    initrd /1bit/initrd.gz
}
EOF
    # what the firmware will read before the update: the four lines 1bit-install wrote, for the "untouched" rows below
    cp "$w/esp/EFI/BOOT/grub.cfg" "$w/$1.grub.installed"
    umount "$w/esp"
    mkdir -p /data && mount "${dev}p2" /data || exit 1
    touch /data/.installed
    mkdir -p /data/1bit
    cp "$w/old.sfs" /data/1bit/puppy_test.sfs
    echo "kernel of c35" > /data/1bit/vmlinuz; echo "initrd of c35" > /data/1bit/initrd.gz
    U=/data/1bit/upup64rsave/upper
    mkdir -p "$U" /data/1bit/upup64rsave/work
    # the machine's c35 life, through a real overlay (old SFS lower, this upper): what Puppy would have written
    mkdir -p "$w/lo" "$w/m"
    mount -o loop,ro "$w/old.sfs" "$w/lo" && loops="$loops $(losetup -j "$w/old.sfs" | cut -d: -f1 | tail -1)"
    mount -t overlay overlay -o "lowerdir=$w/lo,upperdir=$U,workdir=/data/1bit/upup64rsave/work" "$w/m" || exit 1
    printf '#!/bin/sh\n# 05onboard-data of c35, hand-edited on the machine\n' > "$w/m/etc/init.d/05onboard-data"
    echo "root:x:0:0:the machine's own root:/root:/bin/sh" > "$w/m/etc/passwd"
    rm "$w/m/usr/local/bin/1bit-heavy"                                   # -> a whiteout in upper
    rm -rf "$w/m/opt/1bit-package/1bit/npu" && mkdir "$w/m/opt/1bit-package/1bit/npu" &&
        echo "a set copied by hand" > "$w/m/opt/1bit-package/1bit/npu/kernels.hand"   # -> an opaque dir in upper
    mkdir -p "$w/m/root" && echo "the user's file" > "$w/m/root/notes"
    umount "$w/m"; umount "$w/lo"
}

# boot the slot as Puppy would: <slot> -> $w/b is the merged view (that slot's SFS + the shared upper)
boot() {
    umount "$w/b" 2> /dev/null; umount "$w/bl" 2> /dev/null
    mkdir -p "$w/b" "$w/bl" "$w/bw"; rm -rf "$w/bw"/*
    mount -o loop,ro "/data/$1/puppy_test.sfs" "$w/bl" || return 1
    loops="$loops $(losetup -j "/data/$1/puppy_test.sfs" | cut -d: -f1 | tail -1)"
    mount -t overlay overlay -o "lowerdir=$w/bl,upperdir=$U,workdir=/data/1bit/upup64rsave/work" "$w/b"
}

run_update() {  # <script> [slot] [release dir]: the real script, as a booted /1bit (or /1bit-alt) would run it
    _slot=${2:-1bit}
    echo "BOOT_IMAGE=/$_slot/vmlinuz pmedia=atahd psubdir=$_slot" > "$w/cmdline"
    mount --bind "$w/cmdline" /proc/cmdline
    mkdir -p "$w/log"; mount -t tmpfs tmpfs /var/log
    sh "$1" "${3:-$w/release}" > "$w/update.out" 2>&1; rc=$?
    cp /var/log/1bit-update.log "$w/update.log" 2> /dev/null
    umount /var/log; umount /proc/cmdline
    return $rc
}

# ── the control: an older 1bit-update leaves the machine running c35 code over the new release ───────────────────
if [ -n "${ONEBIT_E2E_CONTROL:-}" ]; then
    disk control
    run_update "$ONEBIT_E2E_CONTROL"; crc=$?
    boot 1bit-alt
    echo "control ($ONEBIT_E2E_CONTROL): rc=$crc; the booted new release shows"
    echo "   05onboard-data: $(sed -n 2p "$w/b/etc/init.d/05onboard-data")"
    echo "   1bit-heavy:     $([ -e "$w/b/usr/local/bin/1bit-heavy" ] && echo present || echo MISSING)"
    echo "   npu/:           $(ls "$w/b/opt/1bit-package/1bit/npu" | tr '\n' ' ')"
    umount "$w/b" "$w/bl"; umount /data; losetup -d "$dev"; loops=${loops% *}
fi

# ── the update ───────────────────────────────────────────────────────────────────────────────────────────────────
disk machine
[ "$(getfattr -h --only-values -n trusted.overlay.opaque "$U/opt/1bit-package/1bit/npu" 2> /dev/null)" = y ] &&
    [ -c "$U/usr/local/bin/1bit-heavy" ] && [ -f "$U/etc/init.d/05onboard-data" ]
check $? "the fixture: the kernel's overlay wrote a hand edit, a whiteout and an opaque directory into the save layer"

run_update "$upd"; rc=$?
sed 's/^/   | /' "$w/update.out"
check $rc "1bit-update <release> exits 0 on a loop-mounted installed disk"
[ -s /data/1bit-alt/puppy_test.sfs ] && cmp -s /data/1bit-alt/puppy_test.sfs "$w/release/puppy_test.sfs" &&
    [ "$(readlink /data/1bit-alt/upup64rsave)" = ../1bit/upup64rsave ]
check $? "the new set is in /1bit-alt and links the shared save layer"
mount "${dev}p1" "$w/esp" || exit 1
g=$w/esp/EFI/BOOT/grub.cfg
d=$(sed -n 's/^set default=//p' "$g")
line=$(grub_line "$g" "$d")
has_tok "$line" psubdir=1bit-alt && ! has_tok "$line" 1bit.serial=1
check $? "GRUB's default is the /1bit-alt headless entry"
echo "   /1bit-alt headless entry: $(printf '%s' "$line" | sed 's/^[[:space:]]*//')"
# engine#3374: the line is the NEW release's own, not the running entry's copy
has_tok "$line" printk.always_kmsg_dump=1
check $? "the /1bit-alt boot line carries the NEW release's added parameter (printk.always_kmsg_dump=1; the installed line had none)"
has_tok "$line" pupsfs=UUID=x && has_tok "$line" 1bit.data=UUID=x && ! has_tok "$line" pupsfs=UUID=release &&
    ! has_tok "$line" 1bit.data=UUID=release
check $? "and THIS machine's data-partition pin (UUID=x), not the release image's own UUID"
has_tok "$line" pfix=nocopy,nox && ! has_tok "$line" pfix=copy,nox
check $? "pfix=copy from the image's line becomes nocopy, as 1bit-install writes for an installed disk"
has_tok "$line" log_buf_len=32M && has_tok "$line" "console=ttyS0,115200"
check $? "the release's unchanged parameters are there too (log_buf_len=32M, console=)"
sline=$(awk -v slot=1bit-alt '/^[[:space:]]*linux / && $0 ~ ("psubdir=" slot "([[:space:]]|$)") && /1bit\.serial=1/' "$g")
has_tok "$sline" printk.always_kmsg_dump=1 && has_tok "$sline" 1bit.serial=1
check $? "the /1bit-alt serial root shell entry carries it too"
rline=$(slot_head "$g" 1bit)
! has_tok "$rline" printk.always_kmsg_dump=1 && has_tok "$rline" psubdir=1bit
check $? "the RUNNING /1bit entry was not touched (the rollback keeps its own boot line)"
grep -q "boot parameters from the NEW release's own (/1bit/cmdline)" "$w/update.out"
check $? "the update says where the /1bit-alt line came from"
# the image's own modes are not this script's entries: only the slot's headless and serial lines are rewritten
[ "$(entry_line "$g" 'desktop, JWM')" = "$(entry_line "$w/machine.grub.installed" 'desktop, JWM')" ]
check $? "the /1bit desktop entry keeps the installed line (1bit-update writes a slot's headless and serial entries only)"
[ "$(entry_line "$g" 'install to a disk')" = "$(entry_line "$w/machine.grub.installed" 'install to a disk')" ]
check $? "and the /1bit install-to-disk entry does too"
umount "$w/esp"

S=$(ls -d /data/1bit/upup64rsave/1bit-shadowed/*/ 2> /dev/null | head -1)
[ -n "$S" ] && [ "$(sed -n 2p "$S/etc/init.d/05onboard-data")" = "# 05onboard-data of c35, hand-edited on the machine" ]
check $? "the hand edit was moved out of the overlay, into 1bit-shadowed/<release>-<time>/ (still readable)"
[ -n "$S" ] && grep -qx usr/local/bin/1bit-heavy "$S/whiteouts" && grep -qx opt/1bit-package/1bit/npu "$S/opaque"
check $? "the whiteout and the opaque directory are recorded there"
[ ! -e "$U/etc/init.d/05onboard-data" ] && [ ! -e "$U/usr/local/bin/1bit-heavy" ] &&
    [ -z "$(getfattr -h --only-values -n trusted.overlay.opaque "$U/opt/1bit-package/1bit/npu" 2> /dev/null)" ]
check $? "nothing release-owned is left in the save layer: no copy, no whiteout, no opaque directory"

# ── boot the new release ─────────────────────────────────────────────────────────────────────────────────────────
boot 1bit-alt
check $? "the new slot mounts as Puppy mounts it (its SFS under the shared upper)"
[ "$(sed -n 2p "$w/b/etc/init.d/05onboard-data")" = "# 05onboard-data of v2026.99" ]
check $? "the booted machine runs the NEW 05onboard-data, not c35's hand-edited one (engine#3342)"
[ "$(sed -n 2p "$w/b/usr/local/bin/1bit-heavy")" = "# 1bit-heavy of v2026.99" ]
check $? "the release tool the whiteout hid is back"
[ -f "$w/b/opt/1bit-package/1bit/npu/kernels.v2026.99" ]
check $? "the new release's NPU kernel set shows through the directory that was opaque"
[ "$(cat "$w/b/etc/passwd")" = "root:x:0:0:the machine's own root:/root:/bin/sh" ] && [ "$(cat "$w/b/root/notes")" = "the user's file" ]
check $? "the machine's own state still wins: its accounts and the user's files"

# ── once is enough, and the old slot is the rollback ─────────────────────────────────────────────────────────────
umount "$w/b" "$w/bl"
echo "BOOT_IMAGE=/1bit-alt/vmlinuz psubdir=1bit-alt" > "$w/cmdline"; mount --bind "$w/cmdline" /proc/cmdline
sh "$upd" --shadowed > "$w/shadowed.out" 2>&1; src=$?
umount /proc/cmdline
check $src "--shadowed on the updated machine finds nothing ($(tail -n 1 "$w/shadowed.out"))"
boot 1bit
[ "$(sed -n 2p "$w/b/etc/init.d/05onboard-data")" = "# 05onboard-data of c35" ]
check $? "the rollback slot boots its own release's code (the hand edit is in 1bit-shadowed/, not on it)"
umount "$w/b" "$w/bl"

# ── the SECOND update: the target slot's entries already exist, and a CHANGED parameter must still arrive ─────────
# The machine now runs /1bit-alt, so the next update targets /1bit - the slot 1bit-install laid down, whose headless and
# serial entries are already in the config. Writing entries only when they are missing would leave that slot on the
# INSTALLED release's line for ever; release B changes log_buf_len 32M -> 64M and that value must arrive.
mount "${dev}p1" "$w/esp" || exit 1
alt_before=$(awk '/^[[:space:]]*linux / && $0 ~ ("psubdir=1bit-alt([[:space:]]|$)")' "$g" | sort)
one_before=$(slot_head "$g" 1bit)
umount "$w/esp"
run_update "$upd" 1bit-alt "$w/release2"; rc=$?
sed 's/^/   | /' "$w/update.out"
check $rc "1bit-update from /1bit-alt into the existing /1bit slot exits 0"
[ -s /data/1bit/puppy_test.sfs ] && cmp -s /data/1bit/puppy_test.sfs "$w/release2/puppy_test.sfs"
check $? "the second release's set is in /1bit"
mount "${dev}p1" "$w/esp" || exit 1
d=$(sed -n 's/^set default=//p' "$g")
line=$(grub_line "$g" "$d")
has_tok "$line" psubdir=1bit && has_tok "$line" log_buf_len=64M && ! has_tok "$line" log_buf_len=32M
check $? "GRUB's default is /1bit and its line carries the parameter release B CHANGED (log_buf_len=64M, not 32M)"
echo "   /1bit headless entry: $(printf '%s' "$line" | sed 's/^[[:space:]]*//')"
has_tok "$line" pupsfs=UUID=x && has_tok "$line" 1bit.data=UUID=x && ! has_tok "$line" pupsfs=UUID=release2 &&
    has_tok "$line" printk.always_kmsg_dump=1
check $? "with this machine's UUID pin (not release B's) and release B's added parameter"
case $line in *"linux /1bit/vmlinuz "*) true ;; *) false ;; esac
check $? "the existing entry kept its own kernel path (/1bit/vmlinuz)"
sline=$(awk -v slot=1bit '/^[[:space:]]*linux / && $0 ~ ("psubdir=" slot "([[:space:]]|$)") && /1bit\.serial=1/' "$g")
has_tok "$sline" log_buf_len=64M && has_tok "$sline" 1bit.serial=1
check $? "the /1bit serial entry got the changed value too"
[ "$(awk '/^[[:space:]]*linux / && $0 ~ ("psubdir=1bit-alt([[:space:]]|$)")' "$g" | sort)" = "$alt_before" ]
check $? "the RUNNING /1bit-alt entries are untouched (the rollback keeps release A's line)"
[ "$line" != "$one_before" ]
check $? "the /1bit line is not the one 1bit-install wrote any more"
grep -q "now carry the new release's boot parameters" "$w/update.out"
check $? "the update says it rewrote the existing entries"
[ "$(entry_line "$g" 'desktop, JWM')" = "$(entry_line "$w/machine.grub.installed" 'desktop, JWM')" ] &&
    [ "$(entry_line "$g" 'install to a disk')" = "$(entry_line "$w/machine.grub.installed" 'install to a disk')" ]
check $? "the /1bit desktop and installer entries still carry the installed line after the second update"
[ "$(grep -c '^[[:space:]]*menuentry' "$g")" = 6 ]
check $? "the existing entries were rewritten in place: still 6 menu entries, none duplicated"
umount "$w/esp"

echo "update_e2e_loop: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
