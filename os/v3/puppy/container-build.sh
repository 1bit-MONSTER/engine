#!/bin/bash
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
# container-build.sh - what runs INSIDE the build container. build.sh mounts this directory at /recipe and
# runs it; everything here is container-side (the image is ubuntu:24.04 with the packages build.sh installs).
#
# It was a 300-line single-quoted `bash -c '...'` string in build.sh until 2026-11-02 (cycle 14). Two reasons
# it is a file now: a single apostrophe anywhere in that block - in a comment, in prose - ends the quote and
# bash reports a syntax error a hundred lines away, which cost this lane five builds (the last two in cycle
# 13 alone); and shellcheck cannot parse a host file containing it, so a 300-line script was the one part of
# this lane with no static checking at all.
#
# Inputs (environment, set by build.sh): COMPAT, PAYLOAD=/payload, PAYLOAD_NAME, LIBHSA_DIR=/libhsa-runtime.
# Mounts: /woof (read-only clone), /work (host build root), /kernels, /out, /recipe, /repo, /seal.pub, /payload.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive HOME=/root XDG_CONFIG_HOME=/root/.config GITHUB_ACTIONS=true
apt-get update -qq
apt-get install -y -qq --no-install-recommends build-essential libc6-dev dc debootstrap librsvg2-bin \
    zstd xml2 syslinux-utils xorriso squashfs-tools wget curl git ca-certificates xz-utils file psmisc \
    sudo mtools dosfstools cpio python3
echo "dash dash/sh boolean false" | debconf-set-selections
dpkg-reconfigure dash >/dev/null 2>&1 || true
ln -sf bash /bin/ash
# the whole build happens in /work, which is a host filesystem (ext4), so the petbuild overlay mounts work
# REMOVED first: `cp -a /woof /work/woof-CE` into a directory that already exists copies /woof *inside* it and
# leaves the old tree in place - and the old tree is the one the previous cycle's patch-init patched. That is the
# second half of the staleness that cost cycles 15c and 15d (each wrote an ISO and was then refused by
# `patch-init --verify`, because the refreshed initrd-progs came from /work/woof-CE, which was itself two cycles
# old). The host clone is mounted read-only at /woof and is never patched, so it is the honest source.
rm -rf /work/woof-CE
cp -a /woof /work/woof-CE
# and root-owned, because `cp -a` carries the host user: we run as root in here, and git refuses a repository owned
# by another uid ("detected dubious ownership in repository at /work/woof-CE") - which merge2out hits when it undoes
# its VCS workarounds. Measured on the first cycle-16 build: that step failed once the fresh copy replaced the old
# root-owned tree. It also keeps the image's ownership policy true from the first file.
chown -R 0:0 /work/woof-CE
cd /work/woof-CE
# OWNERSHIP (pi-prs, blocker) second half: 3builddistro unpacks the PET packages into rootfs-complete AFTER the
# skeleton chown below, and mksquashfs keeps whatever ownership it finds - the frugalpup docs arrived owned by
# 6975/2030 and the packed-SFS check refused exactly those three files on cycle 16d. So the trees are normalised
# immediately before woof-CE packs them. -all-root is not available here (SFSCOMP is woof-CE's own), so the tree is
# chowned - but a bare `chown -R` ALSO strips every setuid/setgid bit it touches (POSIX: chown clears them), and
# that is what the c20 board hit as "Activated service 'org.freedesktop.login1' failed: The permission of the setuid
# helper is not correct" - measured, the built tree has 12 setuid helpers and the packed image had none, so passwd,
# su and the dbus launch helper could not work. /recipe/fix-ownership.sh does chown+restore and refuses a tree whose
# count changed. The patch is idempotent, counted, and `bash -n`'d, because a silently unapplied patch would be a
# silent regression of a security fix.
# The guard has to ask for the CALL, not for the comment: a build root or a woof-CE clone reused from a cycle that carried
# the previous version of this patch already holds the marker comment on a bare `chown -R 0:0 rootfs-complete ...` line, so
# a comment-based guard skipped the sed and then refused with a message that blamed upstream. Measured on the cycle-21
# build (2026-11-04): the woof-CE clone at $HOME/wt/1bit-puppy-build11 was dirty with exactly those two lines, and every
# build from it refused. So a line carrying the marker comment IS a previous version of this patch, it is dropped first,
# and the current one is inserted: the patch upgrades itself instead of testing for its own history.
if ! grep -q 'sh /recipe/fix-ownership.sh' woof-code/3builddistro; then
    sed -i '/1bit: ownership before pack/d' woof-code/3builddistro
    sed -i \
        -e 's@^\(\s*\)mksquashfs rootfs-complete @\1sh /recipe/fix-ownership.sh rootfs-complete   # 1bit: ownership before pack\n\1mksquashfs rootfs-complete @' \
        -e 's@^\(\s*\)mksquashfs sandbox3/devx @\1sh /recipe/fix-ownership.sh sandbox3/devx   # 1bit: ownership before pack\n\1mksquashfs sandbox3/devx @' \
        woof-code/3builddistro
fi
n=$(grep -c '1bit: ownership before pack' woof-code/3builddistro)
[ "$n" = 2 ] || {
    echo "build: REFUSED: the woof-CE ownership patch landed $n times in woof-code/3builddistro (expected 2) - upstream changed, re-measure" >&2
    exit 1; }
n=$(grep -c 'sh /recipe/fix-ownership.sh' woof-code/3builddistro)
[ "$n" = 2 ] || {
    echo "build: REFUSED: 3builddistro calls fix-ownership.sh $n times (expected 2) - a bare chown -R strips the image's" >&2
    echo "       setuid bits, so the image would ship without passwd, su or the D-Bus launch helper (the c20 board finding)" >&2
    exit 1; }
! grep -q 'chown -R .*rootfs-complete\|chown -R .*sandbox3/devx' woof-code/3builddistro || {
    echo "build: REFUSED: 3builddistro still chowns a tree with a bare chown -R next to fix-ownership.sh" >&2
    exit 1; }
bash -n woof-code/3builddistro || { echo "build: REFUSED: the patched 3builddistro does not parse" >&2; exit 1; }
echo "   ok: 3builddistro normalises rootfs-complete and devx before packing, keeping their setuid bits"
mkdir -p /work/local-repositories petbuild-sources petbuild-cache petbuild-output
[ -e ../local-repositories ] || ln -s /work/local-repositories ../local-repositories
[ -f local-repositories/vercmp ] || curl -fsSL https://raw.githubusercontent.com/puppylinux-woof-CE/initrd_progs/master/pkg/w_apps_static/w_apps/vercmp.c | gcc -x c -o local-repositories/vercmp -
install -m 755 local-repositories/vercmp /usr/local/bin/vercmp
install -D -m 644 woof-code/rootfs-skeleton/usr/local/petget/categories.dat /usr/local/petget/categories.dat
# initrd-progs packs its initrd with `find . | cpio -o -H newc > ../initrd 2>/dev/null`, so without cpio the
# redirect still creates ../initrd, empty, and gzip obligingly compresses it to 27 bytes: the build stays
# green all the way to a bootable-looking ISO whose initrd has 0 entries (paid for on attempt 7). The
# package list above carries cpio; this refuses rather than producing that ISO again.
command -v cpio >/dev/null || { echo "build: cpio missing, the initrd would come out empty" >&2; exit 1; }
# 3builddistro loops the ESP image to fill it, and losetup needs /dev/loopN to exist. The container /dev
# stops at loop7 while this box runs about twenty loop devices (snaps and other lanes), so loop-control
# hands out a number with no node and the ISO stage dies with "losetup: /dev/loop20: failed to set up loop
# device: No such file or directory" - after every package and every petbuild has been paid for.
for i in $(seq 0 63); do [ -e /dev/loop$i ] || mknod /dev/loop$i b 7 $i 2>/dev/null; done
echo "== merge2out $COMPAT"
# yes | cmd under pipefail: when merge2out exits, yes gets SIGPIPE and the pipeline fails, so the
# status has to be taken from merge2out and not from yes
set +o pipefail
yes "" | ./merge2out woof-distro/x86_64/ubuntu/"$COMPAT"
set -o pipefail   # (no stdin redirect: `set` never reads it, and shellcheck is right to flag it)
cd ../woof-out_*

# 3builddistro packs `../initrd-progs/0initrd` (relative to its own sandbox3), so that tree is what the appliance
# boots with - and a build root reused across cycles keeps the copy the PREVIOUS run patched. Cycle 15 paid for it:
# the second build in the same root was refused with "/init writes the progress lines to /dev/console only" because
# patch-init found the strings it greps for from the older patch and left the file alone, so the fix under test
# never reached the image (the ISO had already been written). The container's /work/woof-CE is a fresh copy of the
# clone every run, so the tree is refreshed from it before anything patches it.
if [ -d initrd-progs ]; then
    echo "== refresh initrd-progs from the woof-CE clone (a reused build root keeps the previous run's copy)"
    rm -rf initrd-progs
    cp -a /work/woof-CE/initrd-progs ./initrd-progs
fi

# The packages that the TPUPPY-RELEASE acceptance needs and the merged woof-CE set does not carry: on a
# built resolute64 rootfs, `sudo`, `apt`, `mokutil` and `openssh-server` are simply absent (openssh-client
# has a row, openssh-server does not) - which is the "sh: sudo: not found" class of gap from c20. The rows
# in packages-extra have to go INSIDE PKGS_SPECS_TABLE (before its closing quote) or 0setup never sees them.
if [ -s /recipe/packages-extra ]; then
    echo "== 1bit package rows into the spec table"
    grep -E "^yes\\|" /recipe/packages-extra > /work/1bit-packages-rows
    for spec in ../woof-out_*/DISTRO_PKGS_SPECS-ubuntu-*; do
        bash /recipe/spec-insert-rows.sh "$spec" /work/1bit-packages-rows
    done
fi
echo "== our kernel into kernel-kit/output"
mkdir -p kernel-kit/output
cp /kernels/huge-*.tar.* kernel-kit/output/
ls -l kernel-kit/output/
echo "== 0setup"; ./0setup </dev/null
echo "== 1download"; ./1download </dev/null
echo "== 2createpackages"; ./2createpackages </dev/null

# U5: our kernel ships ec_su_axb35 (the embedded controller of this board: fans, EC temperature, power caps) and it
# has no PCI modalias, so nothing auto-loads it - measured on the box, where lsmod has no entry and the fans
# ride the firmware curve. The Puppy hook is ADDLIST in /etc/rc.d/MODULESCONFIG, read by rc.sysinit at boot;
# it is a file woof-CE owns, so rewrite that one variable instead of shipping a whole replacement.
# The hook is rootfs-skeleton and not sandbox3/rootfs-complete: 3builddistro starts with
# rm -rf sandbox3/rootfs-complete and rebuilds it from rootfs-skeleton, so anything written into the
# sandbox copy after 2createpackages is thrown away - and nothing fails, the image just comes out without
# it. ADDLIST is read by rc.sysinit as `for MODULE in $ADDLIST`, so an unquoted value is enough.
skeleton=rootfs-skeleton
[ -f "$skeleton/etc/rc.d/MODULESCONFIG" ] || {
    echo "build: REFUSED: no rootfs-skeleton/etc/rc.d/MODULESCONFIG - cannot make the EC driver load at boot" >&2
    exit 1; }
echo "== ec_su_axb35 into the ADDLIST of MODULESCONFIG"
sed -i "s|^ADDLIST=.*|ADDLIST=ec_su_axb35|" "$skeleton/etc/rc.d/MODULESCONFIG"
grep -qx "ADDLIST=ec_su_axb35" "$skeleton/etc/rc.d/MODULESCONFIG" || {
    echo "build: REFUSED: could not set ADDLIST in /etc/rc.d/MODULESCONFIG" >&2; exit 1; }

# U1 (owner, cycle 12): root and every real user log in to bash, not to busybox ash. Measured in this tree:
# rootfs-skeleton/etc/passwd ships root:x:0:0:root:/root:/bin/sh, and bin/plogin ends with
# `exec getty -n -l /bin/autologin ... tty1`, so the login shell is exactly that passwd field.
echo "== login shells -> /bin/bash"
# In its own script (/recipe/shell-login.sh) because this block is single-quoted for docker and an awk
# program in it ends the block early - the quoting trap that has now cost four builds (2026-11-02).
bash /recipe/shell-login.sh "$skeleton" | sed "s/^/   /" || {
    echo "build: REFUSED: the login shells could not be moved to bash (shell-login.sh)" >&2; exit 1; }

# Cycle 13 (pi-release, 2026-11-02 18:39): the headless default must reach the autologin getty on its own.
# rc.country is what stops it: /tmp/bootcnt.txt exists on every pfix=nox boot (init creates it), so
# ASKKMAP and ASKLANG are both yes, /etc/keymap missing means keymap-set cli asks on /dev/console, a missing
# en_US.utf8 locale means chooselocale asks AND sets ASKCLOCK=yes, and timezone-set cli then asks for the
# timezone on /dev/console. Nothing ever answers a console dialog on an appliance, so tty1 never reaches the
# prompt. With the two flags off, rc.country takes its quiet path: keymap-set --startup --load (reads the
# /etc/keymap this image ships: us), `chooselocale cli quiet` (localedef, no questions), and
# set_hwclock_type -q (no prompt) with /etc/localtime -> UTC from the embed. The DESKTOP entry still offers
# the wizard: quicksetup runs on the first X start, which this does not touch.
echo "== rc.country: no console questions on a headless boot"
[ -f "$skeleton/etc/rc.d/rc.country" ] || {
    echo "build: REFUSED: no rootfs-skeleton/etc/rc.d/rc.country - the first-boot wizard would block tty1" >&2
    exit 1; }
sed -i -e "s|^ASKKMAP=\"yes\"|ASKKMAP=\"no\"|" -e "s|^ASKLANG=\"yes\"|ASKLANG=\"no\"|" \
    -e "s|^if \[ -f /tmp/bootcnt.txt \] ; then.*|if false ; then # 1bit: a headless boot must not ask (cycle 13)|" \
    "$skeleton/etc/rc.d/rc.country"
grep -q "^if false ; then # 1bit: a headless boot must not ask (cycle 13)" "$skeleton/etc/rc.d/rc.country" || {
    echo "build: REFUSED: could not stop rc.country from asking on the console" >&2; exit 1; }
echo "   ok: rc.country asks nothing on the console (the keymap and timezone it reads are checked with the embed)"

# tty1 login: inittab respawns `plogin`, whose tail is `exec getty -n -l /bin/autologin`. On the
# 2026-11-02 board boot that chain left no prompt on tty1 while the same image gave one in QEMU, and a
# console-only machine whose console has nothing to log into is unreachable rather than headless. The entry
# now names 1bit-console, which on a headless boot is that last step and nothing else, and on every other
# boot still execs plogin (it is what lays out the first X session).
echo "== tty1 login"
[ -f "$skeleton/etc/inittab" ] || {
    echo "build: REFUSED: no rootfs-skeleton/etc/inittab - tty1 would have no login" >&2; exit 1; }
sed -i 's|^tty1::respawn:plogin$|tty1::respawn:/usr/local/bin/1bit-console|' "$skeleton/etc/inittab"
grep -q '^tty1::respawn:/usr/local/bin/1bit-console$' "$skeleton/etc/inittab" || {
    echo "build: REFUSED: could not point the inittab tty1 entry at 1bit-console" >&2; exit 1; }
echo "   ok: inittab tty1 -> /usr/local/bin/1bit-console (plogin on the desktop entry, the console menu on a headless one)"

# Cycle 15, item B (owner): the appliance is `1bit-os`, not `puppypc22557`. rc.sysinit only invents a
# hostname when /etc/hostname still holds the stock word (`if [ "$(cat /etc/hostname)" = puppypc ]`), so a
# skeleton that already says 1bit-os is left alone - and /etc/hosts has to agree or sudo, sshd banners and
# every log line carry a name that does not resolve. Both are edited here rather than shipped as whole
# files, because /etc/hosts is woof-CE's and has two lines (v4 and v6).
# mDNS (1bit-os.local, cycle 21 item 10): avahi-daemon + libnss-mdns are package rows, and 18onboard-mdns starts the
# responder (and the D-Bus system bus it needs, which nothing else in a non-systemd image starts). Asserted here rather
# than assumed, because woof-CE's package table is one line per package and a split that drops the binary fails at boot
# on a machine nobody can log into. nsswitch gains the mdns line below so the name resolves ON the appliance too.
echo "== hostname 1bit-os"
[ -f "$skeleton/etc/hosts" ] || {
    echo "build: REFUSED: no rootfs-skeleton/etc/hosts - the appliance name would not resolve" >&2; exit 1; }
printf '1bit-os\n' > "$skeleton/etc/hostname"
sed -i 's/\bpuppypc\b/1bit-os/g' "$skeleton/etc/hosts"
grep -q '^1bit-os$' "$skeleton/etc/hostname" || {
    echo "build: REFUSED: could not write the appliance hostname" >&2; exit 1; }
grep -q '1bit-os' "$skeleton/etc/hosts" || {
    echo "build: REFUSED: /etc/hosts does not name the appliance" >&2; exit 1; }
echo "   ok: hostname 1bit-os, and /etc/hosts agrees (rc.sysinit only renames a stock puppypc)"

# item 10: the name a client clicks. The binary is a package row; its PRESENCE is asserted after packing (the
# packages land in the rootfs at 2createpackages, which runs after this block - measured: checking here refused a
# healthy build). The nss line is a file edit and has to happen now, before the sfs is packed.
if [ -f "$skeleton/etc/nsswitch.conf" ] && ! grep -q 'mdns4_minimal' "$skeleton/etc/nsswitch.conf"; then
    sed -i 's/^hosts:[[:space:]]*files[[:space:]]*dns[[:space:]]*$/hosts:          files mdns4_minimal [NOTFOUND=return] dns/' \
        "$skeleton/etc/nsswitch.conf"
fi
grep -q 'mdns4_minimal' "$skeleton/etc/nsswitch.conf" 2>/dev/null || {
    echo "build: REFUSED: nsswitch.conf has no mdns line, so `getent hosts 1bit-os.local` cannot work on the box" >&2
    exit 1; }
echo "   ok: nsswitch resolves .local (mdns4_minimal); avahi-daemon is asserted in the packed image"

# ...and the account the bus runs as. /usr/share/dbus-1/system.conf carries <user>messagebus</user>, and woof-CE
# installs a package by unpacking it, so Debian's postinst never ran adduser: measured in the c20 image, /etc/passwd has
# no messagebus and dbus-daemon --system exits with "Failed to drop privileges". Restored here with a fixed id below
# 1000, which is also what the packed-image uid refusal expects.
if ! grep -q '^messagebus:' "$skeleton/etc/passwd"; then
    printf 'messagebus:x:110:110::/nonexistent:/usr/sbin/nologin\n' >> "$skeleton/etc/passwd"
fi
grep -q '^messagebus:' "$skeleton/etc/group" || printf 'messagebus:x:110:\n' >> "$skeleton/etc/group"
grep -q '^messagebus:' "$skeleton/etc/passwd" && grep -q '^messagebus:' "$skeleton/etc/group" || {
    echo "build: REFUSED: could not create the messagebus account dbus-daemon --system needs (avahi would not start)" >&2
    exit 1; }
echo "   ok: messagebus account for the system bus (dbus-daemon refuses to run as root without it)"

# U3/U4: the release package goes into the rootfs BEFORE 3builddistro, which packs it into the puppy SFS.
# Nothing is installed into it - the payload is relocatable and lemond-onebit finds bin/ and 1bit/ beside
# itself - so this is a copy plus two symlinks and the boot service.
if [ -n "${PAYLOAD:-}" ]; then
    echo "== embed the release package"
    mkdir -p "$skeleton/opt/1bit-package" "$skeleton/usr/local/bin"
    cp -a /payload/bin /payload/1bit /payload/lemond-onebit "$skeleton/opt/1bit-package/"
    ln -sf /opt/1bit-package/lemond-onebit "$skeleton/usr/local/bin/lemond-onebit"
    ln -sf /opt/1bit-package/1bit/1bit "$skeleton/usr/local/bin/1bit"
    cp -a /recipe/embed/. "$skeleton/"
    # the HRX row of release-check.sh asks for ${ONEBIT_HRX_LLAMA_SERVER:-/opt/1bit-llama/llama-server}
    # (os/mini/release-check.sh:454), which is the path os/mini uses for the same binary the package keeps in
    # 1bit/hrx/llama/bin. One compatibility symlink, so the check reads the appliance without being told
    # a new path - pi-release asked for exactly this. The check asserts the symlink and the real file
    # separately: a path THROUGH a symlink is not a path in the squashfs.
    ln -sfn /opt/1bit-package/1bit/hrx/llama/bin "$skeleton/opt/1bit-llama"
    # Cycle 12, U3/U4/U6: the CLI the owner is told to use, the two tools the acceptance rows run, and what
    # the sealed-set rows need. release-check.sh is the SAME file os/mini ships: its catalogue/bench paths
    # were made layout-aware there rather than forked here, so one tool answers for both images.
    ln -sfn /opt/1bit-package/bin/lemonade "$skeleton/usr/local/bin/lemonade"
    install -D -m 755 /repo/os/mini/release-check.sh "$skeleton/usr/local/bin/1bit-release-check"
    # The one-command proof that the engine answers from THIS boot (os/mini ships it as 1bit-selftest and the
    # install sheet asks for it; the Puppy image never installed it, so pi-release's install gate has been
    # reporting `NOT SHIPPED` since the appliance took over from os/mini at c15). selftest.sh is the SAME file
    # both images ship, like release-check.sh above: its /data paths and its engine-log path are layout-aware
    # rather than forked here.
    install -D -m 755 /repo/os/mini/selftest.sh "$skeleton/usr/local/bin/1bit-selftest"
    install -D -m 755 /recipe/1bit-install.sh "$skeleton/usr/local/bin/1bit-install"
    # the board service the `fans` row of release-check reads (os/mini has shipped it all along), and the guarded
    # console installer the "Install 1bit OS to a disk" entry runs
    install -D -m 755 /repo/os/mini/board.sh "$skeleton/usr/sbin/1bit-board"
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-console-install "$skeleton/usr/local/bin/1bit-console-install"
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-console "$skeleton/usr/local/bin/1bit-console"
    # Cycle 15, item A (owner: "no menu system"): the console menu tty1 gets on a headless boot. It needs
    # dialog, which the merged package set already ships - asserted below, because a menu without dialog is a
    # box that falls back to a text prompt with no explanation.
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-console-menu "$skeleton/usr/local/bin/1bit-console-menu"
    # Cycle 18 (pi-tui, ~/briefs/TCONSOLE-TUI.md): the full-screen console TUI replaces the dialog menu on tty1.
    # It is a Rust/ratatui binary built OFF-image (console-tui/build-static.sh, musl-static so it needs nothing at
    # run time) and handed in as CONSOLE_TUI_BIN, because a Rust toolchain has no business in this container. Two
    # rules, both paid for elsewhere in this file: a binary that cannot RUN here is refused rather than installed
    # (init skips a broken program in silence, which is how a console ends up a dead screen), and its absence is
    # only a note - tty1 then keeps the dialog menu, which is the documented fallback.
    if [ -n "${CONSOLE_TUI_BIN:-}" ]; then
        [ -f "$CONSOLE_TUI_BIN" ] || {
            echo "build: REFUSED: CONSOLE_TUI_BIN=$CONSOLE_TUI_BIN does not exist" >&2; exit 1; }
        tuiv=$("$CONSOLE_TUI_BIN" --version 2>&1) || {
            echo "build: REFUSED: the console TUI does not run here (--version said: $tuiv)" >&2; exit 1; }
        case $tuiv in
            1bit-console-tui*) ;;
            *) echo "build: REFUSED: the console TUI printed '$tuiv' for --version" >&2; exit 1;;
        esac
        install -D -m 755 "$CONSOLE_TUI_BIN" "$skeleton/usr/local/bin/1bit-console-tui"
        echo "   ok: the console TUI is in the image ($tuiv)"
    else
        echo "   note: CONSOLE_TUI_BIN is not set - tty1 keeps the dialog menu (the documented fallback)"
    fi
    # Cycle 16, from the demo recording task (TDEMO-RECORD): the appliance has to be able to record its own demo
    # (`1bit-record`), and the desktop has to open a chat screen that WORKS in the browser this image ships -
    # measured 2026-11-02: Pale Moon renders Lemonade's UI but cannot run its streaming chat (it lacks
    # TextDecoderStream and AbortSignal.timeout), while it streams fine from plain fetch + getReader, which is what
    # /opt/1bit/webapp/index.html uses. 24onboard-webapp serves that page on loopback, the autostart opens it.
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-record "$skeleton/usr/local/bin/1bit-record"
    # Cycle 17: the key never goes in the URL (1bit-webapp-proxy injects the header on 8080), the desktop item is
    # 1bit-desktop (it logs the attempt and refuses a reboot), and the boot's logs go to /data/logs (1bit-logs).
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-webapp-proxy "$skeleton/usr/local/bin/1bit-webapp-proxy"
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-desktop "$skeleton/usr/local/bin/1bit-desktop"
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-logs "$skeleton/usr/local/bin/1bit-logs"
    # engine#712: the installed machine's OS-level update (the other slot plus one GRUB default flip)
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-update "$skeleton/usr/local/bin/1bit-update"
    # engine#712: the persistent network identity (a static address on /data, applied at boot, one command to forget)
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-network "$skeleton/usr/local/bin/1bit-network"
    # engine#1474: the BASE layer's keep-list ships so the board row can print what decided it. The normalisation itself
    # runs in fix-ownership.sh, which 3builddistro calls immediately before mksquashfs - the last moment the tree exists,
    # and the same hook that already refuses to pack when the setuid count moved.
    install -D -m 644 /recipe/base-modes-keep "$skeleton/etc/1bit/base-modes-keep"
    install -D -m 755 /recipe/embed/etc/init.d/06onboard-logs "$skeleton/etc/init.d/06onboard-logs"
    install -D -m 755 /recipe/embed/etc/init.d/08onboard-presets "$skeleton/etc/init.d/08onboard-presets"
    install -D -m 755 /recipe/embed/root/Startup/1bit-lemonade-webapp \
        "$skeleton/root/Startup/1bit-lemonade-webapp"
    install -D -m 644 /recipe/embed/etc/personal_settings_popup_disabled \
        "$skeleton/etc/personal_settings_popup_disabled"
    # Pale Moon's first-run prefs live in /etc/1bit, NOT in /opt/palemoon: woof-CE installs the browser with
    # `mv palemoon/ /opt/palemoon`, which refuses a target that already exists - measured 2026-11-03, one build
    # lost to it ("cannot move ... Directory not empty"). 1bit-lemonade-webapp copies the file into the
    # application directory before the browser creates its profile.
    install -D -m 644 /recipe/embed/etc/1bit/palemoon-prefs.js "$skeleton/etc/1bit/palemoon-prefs.js"
    # Cycle 23: the NPU's hang-detector timeout goes in the image, because the stock default resets the device
    # under a chat that crosses ~512 positions (c22 board, 4 Nov; the file's own header carries the numbers).
    install -D -m 644 /recipe/embed/etc/modprobe.d/amdxdna.conf "$skeleton/etc/modprobe.d/amdxdna.conf"
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-lemonade-webapp "$skeleton/usr/local/bin/1bit-lemonade-webapp"
    install -D -m 755 /recipe/embed/etc/init.d/24onboard-webapp "$skeleton/etc/init.d/24onboard-webapp"
    # item 10: the responder that makes http://1bit-os.local:13305 work from another machine (cycle 21, owner).
    install -D -m 755 /recipe/embed/etc/init.d/18onboard-mdns "$skeleton/etc/init.d/18onboard-mdns"
    install -D -m 644 /recipe/embed/opt/1bit/webapp/index.html "$skeleton/opt/1bit/webapp/index.html"
    install -D -m 644 /recipe/embed/etc/xdg/autostart/1bit-lemonade-webapp.desktop \
        "$skeleton/etc/xdg/autostart/1bit-lemonade-webapp.desktop"
    install -D -m 644 /repo/tools/seal_manifest.py "$skeleton/usr/lib/1bit/seal_manifest.py"
    install -D -m 644 /seal.pub "$skeleton/etc/1bit/seal-pub.txt"
    ln -sfn /usr/local/bin/1bit-release-check "$skeleton/bin/1bit-release-check"
    ln -sfn /usr/local/bin/1bit-install "$skeleton/bin/1bit-install"
    # The onebit entries this image ships, written from the payload catalogue it ships: the models row of
    # release-check compares the two, so a catalogue that lost its onebit entries is a FAIL and not a skip.
    python3 /recipe/catalogue-manifest.py /payload/bin/resources/server_models.json \
        "$skeleton/etc/1bit/onebit-models.txt" || {
        echo "build: REFUSED: could not write /etc/1bit/onebit-models.txt from the payload catalogue" >&2
        exit 1; }
    # engine#1557: `payload=` names the payload, and a name is only a claim. This records a fact about the same
    # /payload tree - the sorted list of its file paths - so release-check can compare the identity with the tree
    # the image actually carries. Cheap on both sides (metadata only, and the image-side walk already happens for
    # the payload-modes row), and it needs no packer: the build is what has the payload, so the build is the writer.
# The payload's own identity, algorithmically: relative file paths, sorted, hashed. release-check.sh recomputes it
# over /opt/1bit-package with the SAME expression (grep for "engine#1557" there); the two must stay identical, which
# is why it is one line of find|sed|sort|sha256sum and not a script.
payload_manifest() {   # <payload dir>
    find "$1" -type f 2>/dev/null | sed "s|^$1/||" | LC_ALL=C sort | sha256sum | cut -d' ' -f1
}

    { echo "payload=${PAYLOAD_NAME:-unknown}"
      echo "payload_manifest=$(payload_manifest /payload)"
      echo "engine_sha256=$(sha256sum /payload/1bit/1bit | cut -c1-64)"
      if [ -n "${LIBHSA_DIR:-}" ]; then
          echo "libhsa_sha256=$(sha256sum /libhsa-runtime/libhsa-runtime64.so.1 | cut -c1-64)"
      else echo "libhsa_sha256=none"; fi
      echo "built=$(date -u +%FT%TZ)"; } > "$skeleton/etc/1bit/release-identity"
    # root owns them: /recipe is a host bind mount, so cp -a carries the host uid into the squashfs and
    # sshd StrictModes then refuses the key it was given (measured: authorized_keys owned by ftp)
    chown -R 0:0 "$skeleton/opt/1bit-package" "$skeleton/etc" "$skeleton/etc/1bit" "$skeleton/etc/init.d" \
        "$skeleton/etc/profile.d" "$skeleton/root" "$skeleton/usr/local/bin" 2> /dev/null || true
    chmod 0755 "$skeleton/usr/local/bin/1bit-tty1"
    chmod 0755 "$skeleton"/etc/init.d/03onboard-efivars "$skeleton"/etc/init.d/05onboard-data \
        "$skeleton"/etc/init.d/06onboard-logs "$skeleton"/etc/init.d/07onboard-user \
        "$skeleton"/etc/init.d/08onboard-presets "$skeleton"/etc/init.d/zz-onboard-net \
        "$skeleton"/etc/init.d/10lemonade \
        "$skeleton"/etc/init.d/18onboard-mdns \
        "$skeleton"/etc/init.d/20onboard-ssh "$skeleton"/etc/init.d/21serial-console \
        "$skeleton"/etc/init.d/24onboard-webapp "$skeleton"/etc/init.d/26onboard-board \
        "$skeleton"/etc/init.d/27onboard-installer "$skeleton"/etc/init.d/zz-onboard-ports
    echo "   $(du -sh "$skeleton/opt/1bit-package" | cut -f1) in /opt/1bit-package, service in /etc/init.d"
    if [ -n "${LIBHSA_DIR:-}" ]; then
        # The whole runtime set, flat, at the path the engine own fallback scan walks: app/serve.cpp:207
        # looks for a file named libhsa-runtime64.so.1 anywhere under /opt/rocm-therock and exports
        # IREE_HAL_AMDGPU_LIBHSA_PATH for it. Everything it NEEDs sits beside it, because the RUNPATH of the
        # pinned libhsa is $ORIGIN - that is the whole arrangement, and it is what the working Ubuntu install
        # and the upstream HRX kit both do. No apostrophes in this block: it is single-quoted, and one costs
        # the build 20 minutes plus a syntax error 90 lines later (five times in this lane).
        hsa_set="libhsa-runtime64.so.1 librocm_sysdeps_elf.so.1 librocm_sysdeps_drm.so.2 \
librocm_sysdeps_drm_amdgpu.so.1 librocm_sysdeps_numa.so.1 librocprofiler-register.so.0 \
librocm_sysdeps_z.so.1 librocm_sysdeps_zstd.so.1 librocm_sysdeps_liblzma.so.5 librocm_sysdeps_bz2.so"
        mkdir -p "$skeleton/opt/rocm-therock/lib"
        for f in $hsa_set; do
            cp "/libhsa-runtime/$f" "$skeleton/opt/rocm-therock/lib/$f"
            chmod 0755 "$skeleton/opt/rocm-therock/lib/$f"
        done
        # v2 and /etc/1bit/libhsa.env named /opt/1bit/libhsa; keep the path, point it at the set. ONE level up
        # from /opt/1bit is /opt, so the target is ../rocm-therock/lib - with ../../ it is /rocm-therock/lib, a
        # dangling link mksquashfs then drops, and the image check was the only thing that noticed (measured:
        # the cycle 14 build refused its own ISO with "the image is missing: /opt/1bit/libhsa"). The image
        # check asserts the symlink and the real directory separately: a path THROUGH a symlink is not a path
        # in the squashfs listing (the /opt/1bit-llama lesson).
        rm -rf "$skeleton/opt/1bit/libhsa"
        mkdir -p "$skeleton/opt/1bit"
        ln -sfn ../rocm-therock/lib "$skeleton/opt/1bit/libhsa"
        # The loader is the arbiter, and it is the only thing that can say whether the set is complete:
        # `ldd` on the staged copy, with the set as the search path. 4 lines of "not found" is exactly what
        # the cycle-13 image would have printed had anything asked it.
        nf=$(LD_LIBRARY_PATH="$skeleton/opt/rocm-therock/lib" ldd "$skeleton/opt/rocm-therock/lib/libhsa-runtime64.so.1" 2>&1 | grep -c 'not found' || true)
        [ "${nf:-1}" = 0 ] || {
            echo "build: REFUSED: the HSA runtime set does not load ($nf unresolved):" >&2
            LD_LIBRARY_PATH="$skeleton/opt/rocm-therock/lib" ldd "$skeleton/opt/rocm-therock/lib/libhsa-runtime64.so.1" 2>&1 | sed 's/^/   /' >&2
            exit 1; }
        echo "   HSA runtime set ($(ls "$skeleton/opt/rocm-therock/lib" | wc -l) files, libhsa $(sha256sum /libhsa-runtime/libhsa-runtime64.so.1 | cut -c1-16)) into /opt/rocm-therock/lib, ldd clean"
    else
        # pi-prs' review guide asked why an embedded appliance may be built without an HRX runtime. It may not: an
        # image with the release engine and no libhsa is exactly the cycle-13 appliance that booted, served CPU, and
        # had HRX register NO device on the board (the board's `hrx-device`/`chat-hrx` rows failed on it). The
        # runtime is part of the appliance, not an option, so this is a refusal now - for a PAYLOAD image only, since
        # a plain Puppy base (no PAYLOAD) has no engine that could use one.
        rm -f "$skeleton/etc/1bit/libhsa.env"
        echo "build: REFUSED: PAYLOAD is set but LIBHSA_DIR is not - an appliance with the release engine and no" >&2
        echo "  HSA runtime cannot serve HRX (that is the cycle-13 board failure). Pass LIBHSA_DIR=<dir> with the" >&2
        echo "  10-file runtime set." >&2
        exit 1
    fi
fi

# SECURITY (pi-prs on cycle 15, blocker): nothing in the image may be owned by the user this container runs as.
# Measured on the cycle-15 artifact: /etc/modprobe.d (the directory AND 1bit-quiet.conf), /etc/hostname, /etc/keymap,
# /usr/local/petget/find_cat and the whole zdrv module tree were owned by uid/gid 1000 - and inside the image uid
# 1000 is `ftp`. A group-writable file in /etc/modprobe.d is a `install <module> /bin/sh` line away from root code,
# and a module tree owned by 1000 is a loadable .ko planted by anyone in that group. So: the whole skeleton is
# normalised here (the per-directory chown above only covered directories this recipe writes), group/other write is
# stripped from /etc and from the root-run petget scripts, and the packed SFS is checked again below - the check is
# what a future edit cannot silently undo.
echo "== ownership: the whole skeleton -> root:root, nothing group/other-writable in /etc"
chown -R 0:0 "$skeleton" 2> /dev/null || true
chmod -R go-w "$skeleton/etc" "$skeleton/usr/local/petget" 2> /dev/null || true
# regular files and directories only: a SYMLINK is always 777 and chmod cannot change it, so five links in the
# stock skeleton (/etc/localtime, /etc/ppp/resolv.conf, ...) were reported as group-writable files by the first
# version of this check and refused a good tree (measured). What matters is the mode of the file the link names.
gw=$(find "$skeleton/etc" "$skeleton/usr/local/petget" \( -type f -o -type d \) \
    \( -perm -0020 -o -perm -0002 \) 2> /dev/null | head -5)
[ -z "$gw" ] || { echo "build: REFUSED: still group/other-writable in the tree:" >&2; printf '   %s\n' $gw >&2; exit 1; }
echo "   ok: skeleton is root-owned and /etc is not group/other-writable"

# The initrd /init must not be able to exit: on the 2 Nov board boot PID 1 exited cleanly through the
# init script failure path and the kernel panicked at 26.7 s (Attempted to kill init, exitcode 0).
# 3builddistro packs ../initrd-progs/0initrd (woof-code/3builddistro:649-657), so the guard goes into
# that source BEFORE the packer runs - then the ISO and the disk image mkimage.sh builds both carry it,
# and nothing depends on patching an artifact afterwards.
echo "== init guard into the initrd /init"
bash /recipe/patch-init.sh /work 2>&1 | sed "s/^/   /" || {
    echo "build: REFUSED: could not install the 1bit init guard (patch-init.sh)" >&2; exit 1; }

echo "== 3builddistro"; ./3builddistro release </dev/null
# 3builddistro consumes rootfs-complete, so the image is checked where it exists now: in the SFS set that
# sandbox3/build ships and that Puppy unions at boot. One listing of every SFS, each line sfs-name/path,
# because the answer to "is sudo in the image" is bdrv (debootstrap base), not the main puppy SFS.
compgen -G "sandbox3/build/*.sfs" > /dev/null || {
    echo "build: REFUSED: no SFS in sandbox3/build after 3builddistro" >&2; exit 1; }
# the packed initrd has to carry the guard too: patch-init.sh patched the source, this is the archive
# 3builddistro actually produced, and it is the one mkimage.sh puts on the stick.
bash /recipe/patch-init.sh --verify sandbox3/build/initrd.gz 2>&1 | sed "s/^/   /" || {
    echo "build: REFUSED: the packed initrd lost the 1bit init guard" >&2; exit 1; }
listing=$(for f in sandbox3/build/*.sfs; do
    unsquashfs -l "$f" 2>/dev/null | sed -n "s|^squashfs-root/|$(basename "$f")/|p"
done | sort -u)
# Written to a file and grepped there, never piped into grep -q: with pipefail, grep -q exits at the first
# match, the producer takes SIGPIPE, and the pipeline reports 141 - which reads as "not in the image" for
# every file that IS in the image. That cost cycle 4 a refusal of its own good ISO (measured 2026-11-02).
printf "%s\n" "$listing" > /tmp/1bit-sfs-listing
in_image() { grep -q "/$1$" /tmp/1bit-sfs-listing ; }
# SECURITY, checked on the ARTIFACT and not on the tree: any packed entry owned by a uid or gid >= 1000 fails the
# build, with the entries printed. unsquashfs prints a name when the id resolves in THIS container (root, shadow,
# mail, man, syslog, ubuntu...) and a number when it does not, so both are resolved through id/getent; anything
# unresolvable is treated as an unknown user (> 1000), which is the safe direction.
echo "== uid/gid check on the packed SFS"
: > /tmp/1bit-bad-owners
for f in sandbox3/build/*.sfs; do
    for owner in $(unsquashfs -ll "$f" 2> /dev/null | awk -F'[ \t]+' 'NF > 1 {print $2}' | sort -u); do
        u=${owner%%/*}; g=${owner##*/}
        case $u in ''|*[!0-9]*) uid=$(id -u "$u" 2> /dev/null || echo 65534) ;; *) uid=$u ;; esac
        case $g in ''|*[!0-9]*) gid=$(getent group "$g" 2> /dev/null | cut -d: -f3) ;; *) gid=$g ;; esac
        case ${gid:-} in ''|*[!0-9]*) gid=65534 ;; esac
        [ "$uid" -ge 1000 ] || [ "$gid" -ge 1000 ] || continue
        printf '%s: owner %s (uid %s, gid %s)\n' "$(basename "$f")" "$owner" "$uid" "$gid" >> /tmp/1bit-bad-owners
        unsquashfs -ll "$f" 2> /dev/null | awk -F'[ \t]+' -v o="$owner" '$2 == o {print "   " $6}' | head -20 >> /tmp/1bit-bad-owners
    done
done
if [ -s /tmp/1bit-bad-owners ]; then
    echo "build: REFUSED: the image carries entries owned by a non-system uid/gid (uid 1000 is 'ftp' inside the image:" >&2
    echo "  a group-writable modprobe.d or a module tree owned by it is root code for anyone in that group):" >&2
    sed 's/^/   /' /tmp/1bit-bad-owners >&2
    exit 1
fi
echo "   ok: every packed entry is owned by a system account (uid and gid < 1000)"
# SECURITY, on the ARTIFACT: the chown that makes the ownership policy true also clears setuid/setgid (POSIX), so the
# restore has to be proved in the packed image and not in the tree. A helper that lost its bit still LOOKS present,
# and what it breaks (login1 activation, passwd, su) is not in the boot path, so nothing else would ever notice.
echo "== setuid/setgid check on the packed SFS"
: > /tmp/1bit-suid-report
for f in sandbox3/build/*.sfs; do
    unsquashfs -ll "$f" 2> /dev/null | awk -v F="$(basename "$f")" '$1 ~ /^-.{2}[sS]/ || $1 ~ /^d.{2}[sS]/ {print F": "$1" "$6}' >> /tmp/1bit-suid-report
done
suid_n=$(grep -cv "dbus-daemon-launch-helper" /tmp/1bit-suid-report 2> /dev/null || true)
if ! grep -q 'dbus-daemon-launch-helper' /tmp/1bit-suid-report; then
    echo "build: REFUSED: no setuid dbus-daemon-launch-helper in the packed image. The pre-pack chown stripped it," >&2
    echo "       and without it D-Bus cannot activate login1/hostname1/wpa_supplicant (the c20 board's" >&2
    echo "       'permission of the setuid helper is not correct'). Entries found: $suid_n" >&2
    sed 's/^/   /' /tmp/1bit-suid-report >&2
    exit 1
fi
[ "${suid_n:-0}" -ge 5 ] || {
    echo "build: REFUSED: only ${suid_n:-0} setuid/setgid entries in the packed image (expected the 12 helpers the" >&2
    echo "       packages ship: passwd, chsh, chfn, sudo, ssh-keysign, mount/umount/su-FULL, fusermount3, ntfs-3g," >&2
    echo "       gpasswd, the dbus launch helper); the pre-pack chown stripped them again." >&2
    sed 's/^/   /' /tmp/1bit-suid-report >&2
    exit 1
}
echo "   ok: $suid_n setuid/setgid entries survive the pre-pack chown, incl. the D-Bus launch helper"
# an image that is missing sudo/apt/sshd/mokutil is not a 1bit OS, and nothing else in the build says so
miss=""
# the demo path needs a browser, a recorder and an encoder; an image without them cannot show or record itself
for b in usr/bin/sudo usr/bin/apt usr/sbin/sshd usr/bin/mokutil usr/sbin/sfdisk usr/bin/bash \
         usr/bin/ffmpeg usr/bin/ffprobe usr/bin/python3 opt/palemoon/palemoon usr/sbin/avahi-daemon; do
    in_image "$b" || miss="$miss $b"
done
[ -z "$miss" ] || { echo "build: REFUSED: the image is missing:$miss (packages-extra did not land)" >&2; exit 1; }
# item 10: the responder AND the line that lets the box resolve the name it publishes, both read out of the packed sfs
packed_nss=$(unsquashfs -cat sandbox3/build/puppy_*.sfs etc/nsswitch.conf 2>/dev/null)
case "$packed_nss" in
    *mdns4_minimal*) ;;
    *) echo "build: REFUSED: the packed nsswitch.conf has no mdns line (1bit-os.local would not resolve on the box)" >&2
       exit 1 ;;
esac
echo "   ok: sudo, apt, sshd, mokutil, sfdisk, bash, avahi-daemon are in the image and .local resolves"
# item 10: and the responder must actually be STARTED. rc.services runs every executable /etc/init.d script with
# `start`, but skips any whose text contains the bus binary's name (`grep -qF`, see /etc/rc.d/rc.services - Puppy's
# way of not starting the bus twice). Applied here to the packed script, i.e. the exact file a boot runs.
packed_mdns=$(unsquashfs -cat sandbox3/build/puppy_*.sfs etc/init.d/18onboard-mdns 2>/dev/null)
if [ -z "$packed_mdns" ]; then
    echo "build: REFUSED: the packed image has no etc/init.d/18onboard-mdns (1bit-os.local is never announced)" >&2
    exit 1
fi
if printf '%s' "$packed_mdns" | grep -qF dbus-daemon; then
    echo "build: REFUSED: etc/init.d/18onboard-mdns names the bus binary, so rc.services would SKIP it and the" >&2
    echo "       responder would never start (measured on the c20 image's /etc/rc.d/rc.services). rc.sysinit already" >&2
    echo "       starts that bus; this script must only wait for it." >&2
    exit 1
fi
echo "   ok: etc/init.d/18onboard-mdns will be executed by rc.services (no bus-binary token, so no skip)"
rc_country=$(unsquashfs -cat sandbox3/build/puppy_*.sfs etc/rc.d/rc.country 2>/dev/null |
    grep -c "^if false ; then # 1bit: a headless boot must not ask" || true)
[ "${rc_country:-0}" = 1 ] || {
    echo "build: REFUSED: the packed rc.country would still ask on the console (headless boot never reaches tty1)" >&2
    exit 1; }
in_image etc/keymap || { echo "build: REFUSED: the image has no /etc/keymap" >&2; exit 1; }
in_image etc/localtime || { echo "build: REFUSED: the image has no /etc/localtime" >&2; exit 1; }
# the packed inittab must carry the tty1 entry, not the source file: 3builddistro rebuilds the tree from
# rootfs-skeleton, so a check against the source alone has lied before (the ADDLIST of cycle 1).
unsquashfs -cat sandbox3/build/puppy_*.sfs etc/inittab 2> /dev/null |
    grep -q '^tty1::respawn:/usr/local/bin/1bit-console$' || {
    echo "build: REFUSED: the packed inittab tty1 entry is not 1bit-console (no login on a headless console)" >&2
    exit 1; }
echo "   ok: the packed inittab gives tty1 a login (1bit-console, i.e. the console menu), and /etc/keymap + /etc/localtime are in the image"
# Cycle 15: the console menu is the appliance's whole on-box surface, so the packed image - not the tree - has
# to carry it, the dialog it draws with, the hostname it prints, and the quiet boot. Each of these fails
# silently otherwise: a missing menu is a respawn loop, a missing dialog is a text prompt.
miss=""
for b in usr/bin/dialog etc/hostname etc/modprobe.d/1bit-quiet.conf; do
    in_image "$b" || miss="$miss $b"
done
# the menu itself is installed with the release package (it is part of the embed), so it is asserted with the
# rest of the payload below - and dialog/modprobe/hostname stand alone, because they are the skeleton's.
[ -z "$miss" ] || { echo "build: REFUSED: the image is missing:$miss (the console menu would not run)" >&2; exit 1; }
hn=$(unsquashfs -cat sandbox3/build/puppy_*.sfs etc/hostname 2>/dev/null | head -1)
[ "$hn" = 1bit-os ] || { echo "build: REFUSED: the packed hostname is '${hn:-<none>}' (board row reads 1bit-os)" >&2; exit 1; }
echo "   ok: dialog + hostname 1bit-os + the quiet-boot modprobe file are in the image"
# Cycle 23: the TDR timeout has to be in the PACKED image - and no other packed kmod config may contradict it.
# kmod merges every file in /etc/modprobe.d and a later `options` line wins, so a stale copy silently shadows the
# installed one (the strixhalo lesson, and the reason the board row reads the RUNNING module's parameter).
packed_tdr=$(unsquashfs -cat sandbox3/build/puppy_*.sfs etc/modprobe.d/amdxdna.conf 2> /dev/null)
case "$packed_tdr" in
    *"options amdxdna tdr_timeout_ms=15000"*) ;;
    *) echo "build: REFUSED: the packed image has no 'options amdxdna tdr_timeout_ms=15000' in" >&2
       echo "       /etc/modprobe.d/amdxdna.conf - a long NPU chat would reset the device mid-answer (c22 board)" >&2
       exit 1 ;;
esac
# every other packed modprobe.d file that names amdxdna, across every SFS, must carry the same value. Collected
# into a file first: a `... | while read` loop runs in a subshell, where an exit only ends the loop.
: > /tmp/1bit-tdr-shadow
grep -E 'modprobe\.d/[^/]*amdxdna[^/]*$' /tmp/1bit-sfs-listing | while read -r entry; do
    case "$entry" in */etc/modprobe.d/amdxdna.conf) continue ;; esac
    sfs=${entry%%/*}; rel=${entry#*/}
    body=$(unsquashfs -cat "sandbox3/build/$sfs" "$rel" 2> /dev/null) || continue
    case "$body" in
        *amdxdna*tdr_timeout_ms=15000*) ;;
        *amdxdna*) printf '%s\n' "   $entry" >> /tmp/1bit-tdr-shadow ;;
    esac
done
if [ -s /tmp/1bit-tdr-shadow ]; then
    echo "build: REFUSED: these packed files also configure amdxdna and would shadow the 15000 ms TDR timeout:" >&2
    cat /tmp/1bit-tdr-shadow >&2
    exit 1
fi
echo "   ok: tdr_timeout_ms=15000 for amdxdna is in the packed image, with no shadowing kmod config"
if [ -n "${PAYLOAD:-}" ]; then
    in_image usr/local/bin/1bit-console-menu || {
        echo "build: REFUSED: the image has no /usr/local/bin/1bit-console-menu - tty1 would respawn nothing" >&2; exit 1; }
    # and it has to be EXECUTABLE in the packed SFS: init skips a non-executable script in silence, which is the
    # trap the Lemonade service and 03onboard-efivars already paid for
    # Written to a file and grepped THERE, for the reason spelled out above: under `set -o pipefail` a
    # `producer | grep -q` pipeline reports 141 as soon as grep matches and exits, so the producer takes SIGPIPE
    # and the check reads "not in the image" for a file that is in the image. That is exactly what happened on the
    # first cycle-15 build: it refused its own good SFS (measured: the packed line is
    # `-rwxr-xr-x root/root 23818 ... squashfs-root/usr/local/bin/1bit-console-menu`). The existing listing above
    # already avoids this; this check did not.
    unsquashfs -ll sandbox3/build/puppy_*.sfs > /tmp/1bit-sfs-long 2> /dev/null || true
    grep -q '^-rwxr-xr-x .*usr/local/bin/1bit-console-menu$' /tmp/1bit-sfs-long || {
        echo "build: REFUSED: the packed 1bit-console-menu is not executable" >&2; exit 1; }
    # and the TUI, when this build was given one: it is what tty1 actually execs first, so "in the image" is not
    # enough - it has to be executable in the packed SFS, and `1bit-console` has to still name the fallback.
    if [ -n "${CONSOLE_TUI_BIN:-}" ]; then
        in_image usr/local/bin/1bit-console-tui || {
            echo "build: REFUSED: the image has no /usr/local/bin/1bit-console-tui although CONSOLE_TUI_BIN was given" >&2; exit 1; }
        grep -q '^-rwxr-xr-x .*usr/local/bin/1bit-console-tui$' /tmp/1bit-sfs-long || {
            echo "build: REFUSED: the packed 1bit-console-tui is not executable" >&2; exit 1; }
    fi
    echo "   ok: the console menu is in the image and executable (tty1's whole surface on a headless boot)"
fi
# Cycle 22 (TOS3-NOTE-c21-logs): the boot's log capture has to survive /data mounting AFTER the service started.
# Both spellings that lost four of five boots' logs are refused by name, because each one fails silently: the
# one-shot "is /data mounted?" guard in the service (nothing re-ran the test), and a 1bit-logs whose dir() only
# tested "can I mkdir" ($DATA is a directory even when nothing is mounted on it, so the log went to RAM and
# reported success).
packed_logsvc=$(unsquashfs -cat sandbox3/build/puppy_*.sfs etc/init.d/06onboard-logs 2> /dev/null || true)
packed_logs=$(unsquashfs -cat sandbox3/build/puppy_*.sfs usr/local/bin/1bit-logs 2> /dev/null || true)
case $packed_logsvc in *"this boot's logs stay in RAM"*)
    echo "build: REFUSED: the packed 06onboard-logs gives up when /data is not mounted yet, so those boots keep no log" >&2
    exit 1;; esac
printf '%s\n' "$packed_logs" | grep -q '^mounted() ' || {
    echo "build: REFUSED: the packed 1bit-logs has no mount-point test, so it would log to RAM when /data is not mounted" >&2
    exit 1; }
echo "   ok: the boot's log capture waits for /data (1bit-logs carries the mount-point test)"
modconf=$(unsquashfs -cat sandbox3/build/puppy_*.sfs etc/rc.d/MODULESCONFIG 2>/dev/null | grep "^ADDLIST=" || true)
[ "$modconf" = "ADDLIST=ec_su_axb35" ] || {
    echo "build: REFUSED: the image would not load the EC driver at boot (ADDLIST is $modconf)" >&2; exit 1; }
echo "   ok: $modconf in the image, so rc.sysinit loads the EC driver"
if [ -n "${PAYLOAD:-}" ]; then
    miss=""
    for b in opt/1bit-package/1bit/1bit opt/1bit-package/lemond-onebit etc/init.d/03onboard-efivars \
             etc/init.d/05onboard-data etc/init.d/10lemonade \
             usr/local/bin/1bit etc/rc.d/MODULESCONFIG root/.ssh/authorized_keys \
             etc/init.d/20onboard-ssh etc/init.d/21serial-console etc/init.d/zz-onboard-ports \
             opt/1bit-llama opt/1bit-package/1bit/hrx/llama/bin/llama-server \
             usr/local/bin/lemonade usr/local/bin/1bit-release-check usr/local/bin/1bit-install \
             usr/local/bin/1bit-selftest usr/local/bin/1bit-update \
             usr/local/bin/1bit-tty1 usr/local/bin/1bit-console usr/local/bin/1bit-console-install \
             usr/local/bin/1bit-console-menu \
             usr/sbin/1bit-board etc/init.d/26onboard-board \
             usr/local/bin/1bit-record usr/local/bin/1bit-lemonade-webapp \
             usr/local/bin/1bit-webapp-proxy usr/local/bin/1bit-desktop usr/local/bin/1bit-logs \
             etc/init.d/06onboard-logs etc/init.d/07onboard-user etc/init.d/08onboard-presets \
             etc/init.d/zz-onboard-net usr/local/bin/1bit-network \
             root/Startup/1bit-lemonade-webapp \
             etc/personal_settings_popup_disabled etc/1bit/palemoon-prefs.js \
             etc/init.d/24onboard-webapp opt/1bit/webapp/index.html \
             etc/xdg/autostart/1bit-lemonade-webapp.desktop \
             etc/init.d/27onboard-installer \
             usr/lib/1bit/seal_manifest.py etc/1bit/seal-pub.txt etc/1bit/onebit-models.txt \
             etc/1bit/release-identity etc/keymap etc/localtime etc/timezone; do
        in_image "$b" || miss="$miss $b"
    done
    [ -z "${LIBHSA_DIR:-}" ] || { in_image opt/rocm-therock/lib/libhsa-runtime64.so.1 || miss="$miss the HRX libhsa"; \
        in_image opt/rocm-therock/lib/librocm_sysdeps_elf.so.1 || miss="$miss the HRX libhsa sysdeps"; \
        in_image opt/1bit/libhsa || miss="$miss /opt/1bit/libhsa"; }
    [ -z "$miss" ] || { echo "build: REFUSED: the image is missing:$miss" >&2; exit 1; }
    echo "   ok: engine, launcher, service, /usr/local/bin/{1bit,lemonade}, 1bit-release-check, 1bit-install, 1bit-selftest, 1bit-update"
    echo "      onebit-models: $(grep -c ^hrx "$skeleton/etc/1bit/onebit-models.txt") HRX + $(grep -c ^npu "$skeleton/etc/1bit/onebit-models.txt") NPU entries from the payload catalogue"
fi
echo "== output"
ls -l woof-output-* 2>/dev/null | head -20
cp -f woof-output-*/*.iso woof-output-*/*.sfs /out/ 2>/dev/null || true
# the frugal/persistent install - the installed mode 1bit OS wants - is these SFS files on a partition,
# with no installer writing a rootfs at all, so they belong in <out-dir> next to the ISO. 3builddistro
# leaves them in sandbox3/build/ for the ISO, which is why they are not in woof-output-*.
cp -f sandbox3/build/*.sfs /out/ 2>/dev/null || true
echo "== in /out:"; ls -l /out
