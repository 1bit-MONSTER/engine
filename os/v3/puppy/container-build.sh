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
cp -a /woof /work/woof-CE
cd /work/woof-CE
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
echo "   ok: inittab tty1 -> /usr/local/bin/1bit-console (plogin on the desktop entry, a getty on a headless one)"

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
    install -D -m 755 /recipe/1bit-install.sh "$skeleton/usr/local/bin/1bit-install"
    # the board service the `fans` row of release-check reads (os/mini has shipped it all along), and the guarded
    # console installer the "Install 1bit OS to a disk" entry runs
    install -D -m 755 /repo/os/mini/board.sh "$skeleton/usr/sbin/1bit-board"
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-console-install "$skeleton/usr/local/bin/1bit-console-install"
    install -D -m 755 /recipe/embed/usr/local/bin/1bit-console "$skeleton/usr/local/bin/1bit-console"
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
    { echo "payload=${PAYLOAD_NAME:-unknown}"
      echo "engine_sha256=$(sha256sum /payload/1bit/1bit | cut -c1-64)"
      if [ -n "${LIBHSA_DIR:-}" ]; then
          echo "libhsa_sha256=$(sha256sum /libhsa-runtime/libhsa-runtime64.so.1 | cut -c1-64)"
      else echo "libhsa_sha256=none"; fi
      echo "built=$(date -u +%FT%TZ)"; } > "$skeleton/etc/1bit/release-identity"
    # root owns them: /recipe is a host bind mount, so cp -a carries the host uid into the squashfs and
    # sshd StrictModes then refuses the key it was given (measured: authorized_keys owned by ftp)
    chown -R 0:0 "$skeleton/opt/1bit-package" "$skeleton/etc/1bit" "$skeleton/etc/init.d" \
        "$skeleton/etc/profile.d" "$skeleton/root" "$skeleton/usr/local/bin" 2> /dev/null || true
    chmod 0755 "$skeleton/usr/local/bin/1bit-tty1"
    chmod 0755 "$skeleton"/etc/init.d/03onboard-efivars "$skeleton"/etc/init.d/05onboard-data \
        "$skeleton"/etc/init.d/10lemonade \
        "$skeleton"/etc/init.d/20onboard-ssh "$skeleton"/etc/init.d/21serial-console \
        "$skeleton"/etc/init.d/26onboard-board "$skeleton"/etc/init.d/27onboard-installer \
        "$skeleton"/etc/init.d/zz-onboard-ports
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
        rm -f "$skeleton/etc/1bit/libhsa.env"
        echo "   WARNING: no LIBHSA_DIR=<dir>: this image has no HRX runtime (U4 needs one)"
    fi
fi

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
# an image that is missing sudo/apt/sshd/mokutil is not a 1bit OS, and nothing else in the build says so
miss=""
for b in usr/bin/sudo usr/bin/apt usr/sbin/sshd usr/bin/mokutil usr/sbin/sfdisk usr/bin/bash; do
    in_image "$b" || miss="$miss $b"
done
[ -z "$miss" ] || { echo "build: REFUSED: the image is missing:$miss (packages-extra did not land)" >&2; exit 1; }
echo "   ok: sudo, apt, sshd, mokutil, sfdisk, bash are in the image"
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
echo "   ok: the packed inittab gives tty1 a login (1bit-console), and /etc/keymap + /etc/localtime are in the image"
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
             usr/local/bin/1bit-tty1 usr/local/bin/1bit-console usr/local/bin/1bit-console-install \
             usr/sbin/1bit-board etc/init.d/26onboard-board \
             etc/init.d/27onboard-installer \
             usr/lib/1bit/seal_manifest.py etc/1bit/seal-pub.txt etc/1bit/onebit-models.txt \
             etc/1bit/release-identity etc/keymap etc/localtime; do
        in_image "$b" || miss="$miss $b"
    done
    [ -z "${LIBHSA_DIR:-}" ] || { in_image opt/rocm-therock/lib/libhsa-runtime64.so.1 || miss="$miss the HRX libhsa"; \
        in_image opt/rocm-therock/lib/librocm_sysdeps_elf.so.1 || miss="$miss the HRX libhsa sysdeps"; \
        in_image opt/1bit/libhsa || miss="$miss /opt/1bit/libhsa"; }
    [ -z "$miss" ] || { echo "build: REFUSED: the image is missing:$miss" >&2; exit 1; }
    echo "   ok: engine, launcher, service, /usr/local/bin/{1bit,lemonade}, 1bit-release-check, 1bit-install"
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
