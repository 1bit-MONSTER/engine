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
# build.sh <compat> <kernel-dir> <out-dir> [woof-ce-dir]
#
# Build a 1bit Puppy from woof-CE in a throwaway container: no host package is installed, no host device
# is touched, the container is capped (12 GB memory, 4 CPUs) and nothing survives it except <out-dir>.
#
#   <compat>      resolute64 (Ubuntu 26.04) - the only one that can run the release: our `1bit` and
#                 `lemond` need GLIBC_2.43 (measured), and noble64's Puppy has 2.39
#   <kernel-dir>  the directory os/v3/puppy/mk-huge-kernel.sh wrote: huge-<kver>.tar.zst
#   <out-dir>     where the built .iso and .sfs land
#   [woof-ce-dir] the woof-CE clone (default $HOME/wt/woof-CE)
#
# PAYLOAD=<dir> embeds the release package (TPUPPY-RELEASE U3/U4): the unpacked
# lemonade-onebit-<tag>/ tree, whose lemond-onebit launcher resolves bin/ and 1bit/ BESIDE ITSELF. It is
# copied to /opt/1bit-package, /etc/init.d/10lemonade from os/v3/puppy/embed starts it at boot (Puppy's
# rc.services runs every executable /etc/init.d script), and the build refuses an image whose rootfs lacks
# the engine or the launcher. Without PAYLOAD the image is the plain Puppy base, as before.
#
# Two things the woof-CE build assumes that are not obvious, both measured here:
#   * merge2out refuses to run unless $GITHUB_ACTIONS is set or the host is a running Puppy - CI sets it,
#     so a container build has to as well;
#   * `3builddistro` takes the kernel from `woof-out_*/kernel-kit/output/` and otherwise downloads a stock
#     Ubuntu kernel. Dropping our huge tarball there is what makes the image boots OUR kernel.
#
# Inside the container: merge2out -> kernel into kernel-kit/output -> 0setup (package indexes) ->
# 1download (the Ubuntu packages) -> 2createpackages (Puppy's split into puppy/zdrv/fdrv/bdrv SFS) ->
# 3builddistro release (the SFS files and the bootable ISO).
set -euo pipefail
[ $# -ge 3 ] || { sed -n '/^# build.sh/,/^# Inside the container/p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
compat=$1 kernel=$2 out=$3
# /tmp survives a session, not a reboot: the default clone lives under $HOME (measured 2026-11-01, the
# first re-run after a restart stopped at "no woof-CE at /tmp/woof-ce")
woof=${4:-$HOME/wt/woof-CE}
# The build tree must live on a real filesystem that allows device nodes, for two measured reasons:
#   * 3builddistro mounts an overlay for its petbuild chroots and overlay-on-overlay is refused ("mount:
#     ... bad superblock on petbuild", the chroot has no bash, the build stops at busybox) - a container's
#     own rootfs is overlay, so the build root is a host directory bind-mounted in;
#   * debootstrap refuses a target on a nodev mount ("cannot create ... test-dev-null ... mounted with
#     noexec or nodev"), and /tmp on this box is exactly that: tmpfs, nodev.
# So the default is under $HOME on the root filesystem, and a build root that cannot host the build is
# refused before 20 minutes of downloads.
build_root=${BUILD_ROOT:-$HOME/wt/1bit-puppy-build}
mkdir -p "$build_root"; build_root=$(cd "$build_root" && pwd)
mnt_opts=$(findmnt -T "$build_root" -no OPTIONS 2>/dev/null || true)
case ",$mnt_opts," in
*,nodev,* | *,noexec,*)
    printf 'build: REFUSED: %s is mounted %s; the build needs device nodes and executables there\n' \
        "$build_root" "$mnt_opts" >&2
    exit 1
    ;;
esac
[ -f "$woof/merge2out" ] || { echo "build: no woof-CE at $woof - clone https://github.com/puppylinux-woof-CE/woof-CE there, or pass the directory as the 4th argument" >&2; exit 1; }
kernel=$(cd "$kernel" && pwd)
ls "$kernel"/huge-*.tar.* >/dev/null 2>&1 || {
    echo "build: no huge-*.tar.* in $kernel (run mk-huge-kernel.sh first)" >&2; exit 1; }
mkdir -p "$out"; out=$(cd "$out" && pwd)
recipe=$(cd "$(dirname "$0")" && pwd)   # os/v3/puppy, mounted at /recipe: packages-extra lives there
payload=""
if [ -n "${PAYLOAD:-}" ]; then
    payload=$(cd "$PAYLOAD" 2>/dev/null && pwd) || { echo "build: PAYLOAD=$PAYLOAD does not exist" >&2; exit 1; }
    [ -x "$payload/lemond-onebit" ] && [ -x "$payload/1bit/1bit" ] || {
        echo "build: $payload is not the release package: want lemond-onebit and 1bit/1bit in it" >&2; exit 1; }
fi
free_gb=$(free -g | awk '/^Mem:/{print $7}')
[ "${free_gb:-0}" -ge 16 ] || echo "build: WARNING: ${free_gb} GB available; the container is capped at 12 GB" >&2

echo "build: woof-CE $woof, kernel $kernel, out $out, build root $build_root ($mnt_opts)"
exec docker run --rm --privileged --memory=12g --cpus=4 \
    -v "$woof":/woof:ro -v "$build_root":/work -v "$kernel":/kernels:ro -v "$out":/out \
    -v "$recipe":/recipe:ro \
    ${payload:+-v "$payload":/payload:ro} \
    ${LOOP_DEVICE:+--device /dev/loop-control:/dev/loop-control} \
    -e COMPAT="$compat" ubuntu:24.04 bash -c '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive HOME=/root XDG_CONFIG_HOME=/root/.config GITHUB_ACTIONS=true
apt-get update -qq
apt-get install -y -qq --no-install-recommends build-essential libc6-dev dc debootstrap librsvg2-bin \
    zstd xml2 syslinux-utils xorriso squashfs-tools wget curl git ca-certificates xz-utils file psmisc \
    sudo mtools dosfstools cpio
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
echo "== merge2out $COMPAT"
# yes | cmd under pipefail: when merge2out exits, yes gets SIGPIPE and the pipeline fails, so the
# status has to be taken from merge2out and not from yes
set +o pipefail
yes "" | ./merge2out woof-distro/x86_64/ubuntu/"$COMPAT"
set -o pipefail </dev/null
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

# U3/U4: the release package goes into the rootfs BEFORE 3builddistro, which packs it into the puppy SFS.
# Nothing is installed into it - the payload is relocatable and lemond-onebit finds bin/ and 1bit/ beside
# itself - so this is a copy plus two symlinks and the boot service.
if [ -n "${PAYLOAD:-}" ]; then
    echo "== embed the release package"
    rootfs=$(find sandbox3 -maxdepth 2 -type d -name rootfs-complete 2>/dev/null | head -n 1)
    [ -n "$rootfs" ] || { echo "build: REFUSED: no rootfs-complete to embed into" >&2; exit 1; }
    mkdir -p "$rootfs/opt/1bit-package" "$rootfs/usr/local/bin"
    cp -a /payload/bin /payload/1bit /payload/lemond-onebit "$rootfs/opt/1bit-package/"
    ln -sf /opt/1bit-package/lemond-onebit "$rootfs/usr/local/bin/lemond-onebit"
    ln -sf /opt/1bit-package/1bit/1bit "$rootfs/usr/local/bin/1bit"
    cp -a /recipe/embed/. "$rootfs/"
    chmod 0755 "$rootfs/etc/init.d/10lemonade"
    echo "   $(du -sh "$rootfs/opt/1bit-package" | cut -f1) in /opt/1bit-package, service in /etc/init.d"
fi

echo "== 3builddistro"; ./3builddistro release </dev/null
# an image that is missing sudo/apt/sshd/mokutil is not a 1bit OS, and nothing else in the build says so
rootfs=$(find sandbox3 -maxdepth 2 -type d -name rootfs-complete 2>/dev/null | head -n 1)
if [ -z "$rootfs" ]; then
    echo "build: REFUSED: no sandbox3/rootfs-complete after 3builddistro, so the package set cannot be verified" >&2
    exit 1
fi
miss=""
for b in usr/bin/sudo usr/bin/apt usr/sbin/sshd usr/bin/mokutil; do
    [ -e "$rootfs/$b" ] || miss="$miss $b"
done
[ -z "$miss" ] || { echo "build: REFUSED: the rootfs is missing:$miss (packages-extra did not land)" >&2; exit 1; }
echo "   ok: sudo, apt, sshd, mokutil are in $rootfs"
if [ -n "${PAYLOAD:-}" ]; then
    miss=""
    [ -x "$rootfs/opt/1bit-package/1bit/1bit" ] || miss="$miss the engine"
    [ -x "$rootfs/opt/1bit-package/lemond-onebit" ] || miss="$miss the lemond-onebit launcher"
    [ -x "$rootfs/etc/init.d/10lemonade" ] || miss="$miss the boot service (not executable: rc.services only runs executable ones)"
    [ -f "$rootfs/usr/local/bin/1bit" ] || miss="$miss /usr/local/bin/1bit"
    [ -z "$miss" ] || { echo "build: REFUSED: the image is missing:$miss" >&2; exit 1; }
    echo "   ok: the engine, the launcher, the service and /usr/local/bin/1bit are in the rootfs"
fi
echo "== output"
ls -l woof-output-* 2>/dev/null | head -20
cp -f woof-output-*/*.iso woof-output-*/*.sfs /out/ 2>/dev/null || true
# the frugal/persistent install - the installed mode 1bit OS wants - is these SFS files on a partition,
# with no installer writing a rootfs at all, so they belong in <out-dir> next to the ISO. 3builddistro
# leaves them in sandbox3/build/ for the ISO, which is why they are not in woof-output-*.
cp -f sandbox3/build/*.sfs /out/ 2>/dev/null || true
echo "== in /out:"; ls -l /out
'
