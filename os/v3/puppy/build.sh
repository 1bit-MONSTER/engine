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
woof=${4:-/tmp/woof-ce}
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
[ -f "$woof/merge2out" ] || { echo "build: no woof-CE at $woof" >&2; exit 1; }
kernel=$(cd "$kernel" && pwd)
ls "$kernel"/huge-*.tar.* >/dev/null 2>&1 || {
    echo "build: no huge-*.tar.* in $kernel (run mk-huge-kernel.sh first)" >&2; exit 1; }
mkdir -p "$out"; out=$(cd "$out" && pwd)
free_gb=$(free -g | awk '/^Mem:/{print $7}')
[ "${free_gb:-0}" -ge 16 ] || echo "build: WARNING: ${free_gb} GB available; the container is capped at 12 GB" >&2

echo "build: woof-CE $woof, kernel $kernel, out $out, build root $build_root ($mnt_opts)"
exec docker run --rm --privileged --memory=12g --cpus=4 \
    -v "$woof":/woof:ro -v "$build_root":/work -v "$kernel":/kernels:ro -v "$out":/out \
    -e COMPAT="$compat" ubuntu:24.04 bash -c '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive HOME=/root XDG_CONFIG_HOME=/root/.config GITHUB_ACTIONS=true
apt-get update -qq
apt-get install -y -qq --no-install-recommends build-essential libc6-dev dc debootstrap librsvg2-bin \
    zstd xml2 syslinux-utils xorriso squashfs-tools wget curl git ca-certificates xz-utils file psmisc \
    sudo mtools
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
echo "== merge2out $COMPAT"
# yes | cmd under pipefail: when merge2out exits, yes gets SIGPIPE and the pipeline fails, so the
# status has to be taken from merge2out and not from yes
set +o pipefail
yes "" | ./merge2out woof-distro/x86_64/ubuntu/"$COMPAT"
set -o pipefail </dev/null
cd ../woof-out_*
echo "== our kernel into kernel-kit/output"
mkdir -p kernel-kit/output
cp /kernels/huge-*.tar.* kernel-kit/output/
ls -l kernel-kit/output/
echo "== 0setup"; ./0setup </dev/null
echo "== 1download"; ./1download </dev/null
echo "== 2createpackages"; ./2createpackages </dev/null
echo "== 3builddistro"; ./3builddistro release </dev/null
echo "== output"
ls -l woof-output-* 2>/dev/null | head -20
cp -f woof-output-*/*.iso woof-output-*/*.sfs /out/ 2>/dev/null || true
echo "== in /out:"; ls -l /out
'
