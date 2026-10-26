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
# mkrootfs.sh <repo-dir> <rootfs-dir> [ssh-authorized-key.pub]
#
# Builds the appliance's root filesystem: a minimal Ubuntu from one snapshot (pin.txt) with
# mmdebstrap, and 1bit-appliance from the repository this script is handed - the same install an
# owner's box does from 1bit.gg/apt, only out of a directory.
#
# MIRROR=<url> overrides the pinned snapshot for one run. That is for a snapshot.ubuntu.com outage,
# never for a release: the same packages from a moving archive are not the release pin.txt describes.
#
# Nothing of the appliance is unpacked by hand: the units, the banner, the /data mount, the first-boot
# step and the engine arrive as packages. The only additions are the two a test image needs - a way in
# over SSH (the real live image bakes its own key) and the hostname - both done in the customize hook.
#
# The repository goes in at /srv/1bit-apt and apt reads it as file://, so the signed-by chain is the
# same one https would use: an unsigned or wrongly signed Release file fails the build here.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '21,32p' "$0"; exit 2; }
repo=$(cd "$1" && pwd) rootfs=$2 sshkey=${3:-}
here=$(cd "$(dirname "$0")" && pwd)
. "$here/pin.txt"
mirror=${MIRROR:-$mirror}
[ -d "$repo/dists/stable" ] || { echo "mkrootfs: $repo is not an apt repository (mkrepo.sh)"; exit 2; }

# The repository is served over HTTP for the install: apt's file:// method drops to the _apt sandbox
# user, and a directory only the build user can traverse fails there in ways that look like a missing
# Release file. The owner's box fetches over https; the difference here is only the URL.
port=$((20000 + RANDOM % 20000))
( cd "$repo" && exec python3 -m http.server "$port" --bind 127.0.0.1 >/dev/null 2>&1 ) &
httpsrv=$!
trap 'kill $httpsrv 2>/dev/null' EXIT
sleep 1
curl -sf "http://127.0.0.1:$port/dists/stable/Release" >/dev/null ||
    { echo "mkrootfs: the local repository server did not answer on :$port"; exit 1; }

# the trust anchor comes out of the package that carries it, so the keyring used to build is exactly
# the one the box will trust
keydeb=$(ls "$repo"/pool/main/*/*/1bit-archive-keyring_*.deb 2>/dev/null | head -1)
[ -n "$keydeb" ] || { echo "mkrootfs: no 1bit-archive-keyring deb in $repo"; exit 2; }
kr=$(mktemp)
dpkg-deb --fsys-tarfile "$keydeb" | tar -xO ./usr/share/keyrings/1bit-archive-keyring.gpg > "$kr"
[ -s "$kr" ] || { echo "mkrootfs: the keyring deb carries an empty keyring (mkrepo.sh signed?)"; exit 2; }

# what a minimal appliance needs on top of minbase. The kernel is Ubuntu's here: it is the VM's, and
# our own 1bit-kernel-image takes its place on the machine (design §2) - and 1bit-appliance depends on
# it, so apt installs it too when the repository carries it. Mesa's RADV is there for the same reason it
# is in the os/mini image - Lemonade's vulkan backend has to run in the VM. grub, shim, efibootmgr and
# squashfs-tools are what mklive.sh needs to build the live medium out of this rootfs.
inc="systemd,systemd-sysv,dbus,systemd-resolved,openssh-server,initramfs-tools,linux-image-virtual,mesa-vulkan-drivers,libvulkan1,libgl1,e2fsprogs,dosfstools,fdisk,ca-certificates,wget,curl,zstd,gnupg,vim-tiny,less,procps,grub-efi-amd64-bin,shim-signed,efibootmgr,squashfs-tools,kmod"
# fdisk is not decoration: it is where Ubuntu's sfdisk lives, and 1bit-install lays out a GPT disk with it
# (minbase does not carry it - measured: rootfs5 had no /usr/sbin/sfdisk at all)
# the setup hook runs before the skeleton has /srv and sources.list.d, so it makes them
# The build-time source trusts the key through trusted.gpg.d: mmdebstrap's first apt run happens
# before any package exists, and there signed-by pointing at a file it cannot read yet is a puzzle
# without a payoff. The source the box actually keeps (from 1bit-appliance) uses signed-by and is
# asserted in the VM.
setup="mkdir -p \"\$1\"/usr/share/keyrings \"\$1\"/etc/apt/trusted.gpg.d \"\$1\"/etc/apt/sources.list.d && cp $kr \"\$1\"/usr/share/keyrings/1bit-archive-keyring.gpg && chmod 644 \"\$1\"/usr/share/keyrings/1bit-archive-keyring.gpg && cp $kr \"\$1\"/etc/apt/trusted.gpg.d/1bit-archive-keyring.gpg && chmod 644 \"\$1\"/etc/apt/trusted.gpg.d/1bit-archive-keyring.gpg && printf 'deb http://127.0.0.1:$port stable main\\n' > \"\$1\"/etc/apt/sources.list.d/00-1bit-build.list"
# the build-time source goes away; what stays is the production one from 1bit-appliance plus a
# test-only file:// source over the copy in /srv, so the VM can prove apt works without the internet
customize='rm -f /etc/apt/sources.list.d/00-1bit-build.list; mkdir -p /srv; cp -a /tmp/1bit-apt /srv/1bit-apt 2>/dev/null || true; chmod -R a+rX /srv/1bit-apt 2>/dev/null || true; printf "deb [signed-by=/usr/share/keyrings/1bit-archive-keyring.gpg] file:///srv/1bit-apt stable main\\n" > /etc/apt/sources.list.d/99-1bit-test.list; echo 1bit-os > /etc/hostname; mkdir -p /data; if [ -s /tmp/1bit-test-key.pub ]; then install -d -m 700 /root/.ssh; cp /tmp/1bit-test-key.pub /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys; fi; update-initramfs -u -k all >/dev/null 2>&1 || true; rm -rf /var/lib/apt/lists/* /tmp/1bit-test-key.pub'
setup="$setup && mkdir -p \"\$1\"/tmp && cp -a $repo \"\$1\"/tmp/1bit-apt"
if [ -n "$sshkey" ]; then
    setup="$setup && cp $sshkey \"\$1\"/tmp/1bit-test-key.pub && chmod 644 \"\$1\"/tmp/1bit-test-key.pub"
fi

# mmdebstrap needs to create the tree as root: in unprivileged mode it cannot write under a home
# directory that is not world-writable, which is how this host is set up
if [ "$(id -u)" = 0 ]; then
    rm -rf "$rootfs"; MMD=(mmdebstrap)
else
    # root, because a rootfs has /dev nodes and a home directory here is not world-writable
    sudo -n rm -rf "$rootfs"; MMD=(sudo -n mmdebstrap)
fi
"${MMD[@]}" --variant=minbase --arch=amd64 \
    --components="$components" \
    --include="$inc,1bit-appliance" \
    --keyring="$kr" \
    --aptopt='Acquire::Retries "5"' \
    --setup-hook="$setup" \
    --chrooted-customize-hook="$customize" \
    resolute "$rootfs" "$mirror"
rm -f "$kr"
echo "== rootfs: $(du -sh "$rootfs" | cut -f1) at $rootfs"
