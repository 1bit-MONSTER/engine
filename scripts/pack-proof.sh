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
# pack-proof.sh <package-dir> <version>
#
# The proof for the engine's own packages (engine #1279 item 4): every family installs it, the engine
# answers, the shim runs, and it uninstalls cleanly; the tarball is unpacked as root the way
# docs/install-any-distro.md says and must not touch a directory that is not ours; and the four
# packages are unpacked side by side to check they carry the same thing. One PASS/FAIL row per check,
# non-zero exit if any row fails.
#
# Needs docker and nothing else on the host: every install happens inside a throwaway container of the
# family it is for, and the host is left with one temporary directory. Run from the release's package
# stage (scripts/weekly.sh) or by hand:
#
#   scripts/pack-proof.sh <out-dir with the four packages> <tag or version>
#
# The version is normalized exactly as the packer normalizes it (scripts/pack-version.sh), so the tag a
# cut carries works here too: rc-2026-11-07-c32 finds 1bit-engine_2026.11.07.c32_amd64.deb.
#
# The rows and what each one means:
#   deb      ubuntu:26.04          apt-get install ./1bit-engine_<version>_amd64.deb
#   rpm      fedora:latest         dnf install ./1bit-engine-<version>-1.x86_64.rpm
#   rpm      opensuse/tumbleweed   zypper install ./... (the same file: one rpm for both families)
#   arch     archlinux:latest      pacman -U ./1bit-engine-<version>-1-x86_64.pkg.tar.zst
#   tarball  ubuntu:26.04          tar -C / --zstd --no-same-owner --no-overwrite-dir -xf ...
#   same     ubuntu:26.04          the four unpacked as root: paths, modes, owner, sizes, sha256
#
# The deb/rpm/arch rows check removal too (nothing of the payload may be left); the tarball row does
# not, because a tarball has nothing to uninstall. The images are the distributions the release says
# it supports, named once here so a release cannot quietly prove itself on something else.
set -uo pipefail

# usage(): the header comment as prose, anchored on the comment block rather than a line range -- two of these ranges had
# drifted past the end of their header and printed code to whoever asked for help (engine#2129).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

[ $# -eq 2 ] || { usage; exit 2; }
dir=$(cd "$1" && pwd) || exit 2
# shellcheck source=scripts/pack-version.sh
. "$(dirname "$0")/pack-version.sh"
# one version per family: a candidate's has to sort below its release, and pacman spells that
# differently from deb and rpm (scripts/pack-version.sh)
v_deb=$(pack_version "$2" deb) || exit 2
v_rpm=$(pack_version "$2" rpm) || exit 2
v_arch=$(pack_version "$2" arch) || exit 2
v_tar=$(pack_version "$2" tarball) || exit 2
deb=1bit-engine_${v_deb}_amd64.deb
rpm=1bit-engine-${v_rpm}-1.x86_64.rpm
arch_pkg=1bit-engine-${v_arch}-1-x86_64.pkg.tar.zst
tarball=1bit-engine-${v_tar}-linux-x86_64.tar.zst
for f in "$deb" "$rpm" "$arch_pkg" "$tarball"; do
    [ -f "$dir/$f" ] || { echo "pack-proof: no $f in $dir (the release builds all four: scripts/mk-plugin-package.sh)" >&2; exit 2; }
done
command -v docker >/dev/null || { echo "pack-proof: no docker on PATH; the rows are container installs" >&2; exit 2; }

rows=0 failed=0

# run_row <label> <image> <in-container script>
run_row() {
    local label=$1 image=$2 script=$3 out rc
    rows=$((rows + 1))
    out=$(docker run --rm -v "$dir:/pkg:ro" "$image" bash -uo pipefail -c "$script" 2>&1)
    rc=$?
    if [ "$rc" = 0 ]; then
        printf 'PASS  %-34s %s\n' "$label" "$image"
    else
        failed=$((failed + 1))
        printf 'FAIL  %-34s %s\n' "$label" "$image"
        printf '%s\n' "$out" | tail -10 | sed 's/^/      /'
    fi
}

# the body the three package-manager rows share: install, answer, run the shim, remove, and nothing left
# (the shim is llama.cpp underneath and prints its version banner on stderr, so 2>&1 is not optional)
body='
engine=$(/usr/bin/1bit-engine --version 2>&1) || { echo "1bit-engine --version failed"; exit 1; }
case $engine in "1bit "*) ;; *) echo "unexpected version output: $engine"; exit 1 ;; esac
echo "version: $engine"
shim=$(/opt/1bit/engine/lemonade-bin/llama-server --version 2>&1 | head -2 | tr "\n" " ")
case $shim in *"version: 1 ("*) ;; *) echo "the shim did not run: $shim"; exit 1 ;; esac
echo "shim: $shim"
'

run_row "deb, ubuntu 26.04" ubuntu:26.04 "
apt-get update -qq >/dev/null 2>&1 || { echo 'apt-get update failed'; exit 1; }
apt-get install -y -qq /pkg/$deb >/dev/null 2>&1 || { echo 'apt-get install failed'; exit 1; }
$body
apt-get remove -y -qq 1bit-engine >/dev/null 2>&1 || { echo 'apt-get remove failed'; exit 1; }
[ ! -e /opt/1bit/engine/1bit ] || { echo 'the payload survived the removal'; exit 1; }
"

run_row "rpm, fedora" fedora:latest "
dnf -y -q install /pkg/$rpm >/dev/null 2>&1 || { echo 'dnf install failed'; exit 1; }
$body
dnf -y -q remove 1bit-engine >/dev/null 2>&1 || { echo 'dnf remove failed'; exit 1; }
[ ! -e /opt/1bit/engine/1bit ] || { echo 'the payload survived the removal'; exit 1; }
"

run_row "rpm, opensuse tumbleweed" opensuse/tumbleweed:latest "
zypper --non-interactive --gpg-auto-import-keys install --allow-unsigned-rpm /pkg/$rpm >/dev/null 2>&1 ||
    { echo 'zypper install failed'; exit 1; }
$body
zypper --non-interactive remove 1bit-engine >/dev/null 2>&1 || { echo 'zypper remove failed'; exit 1; }
[ ! -e /opt/1bit/engine/1bit ] || { echo 'the payload survived the removal'; exit 1; }
"

run_row "arch, pacman" archlinux:latest "
pacman -Sy --noconfirm >/dev/null 2>&1 || { echo 'pacman -Sy failed'; exit 1; }
pacman -U --noconfirm /pkg/$arch_pkg >/dev/null 2>&1 || { echo 'pacman -U failed'; exit 1; }
$body
pacman -R --noconfirm 1bit-engine >/dev/null 2>&1 || { echo 'pacman -R failed'; exit 1; }
[ ! -e /opt/1bit/engine/1bit ] || { echo 'the payload survived the removal'; exit 1; }
"

# the tarball, exactly as the guide documents it, as root: the archive's members are root's, and the
# extraction leaves the directories that are not ours alone even without the flags that would protect them
run_row "tarball, extracted as root" ubuntu:26.04 "
apt-get update -qq >/dev/null 2>&1 || { echo 'apt-get update failed'; exit 1; }
apt-get install -y -qq zstd libgomp1 libssl3t64 libzstd1 libpcre2-8-0 >/dev/null 2>&1 ||
    { echo 'installing the deps the .deb declares failed'; exit 1; }
notroot=\$(tar --zstd --numeric-owner -tvf /pkg/$tarball | awk -v z=0/0 '\$2 != z' | wc -l)
[ \"\$notroot\" = 0 ] || { echo \"\$notroot tarball members are not owned by root (a bare tar -xf as root would take them over)\"; exit 1; }
before=\$(stat -c '%u:%g' / /usr /usr/bin /opt | tr '\n' ' ')
tar -C / --zstd --no-same-owner --no-overwrite-dir -xf /pkg/$tarball || { echo 'tar failed'; exit 1; }
after=\$(stat -c '%u:%g' / /usr /usr/bin /opt | tr '\n' ' ')
[ \"\$before\" = \"\$after\" ] || { echo \"the extraction changed a system directory: \$before -> \$after\"; exit 1; }
stray=\$(find /opt/1bit ! -user root | wc -l)
[ \"\$stray\" = 0 ] || { echo \"\$stray payload paths are not owned by root\"; exit 1; }
$body
"

# and the four carry the same thing, compared as root so each archive's own owner is what is read
run_row "the four packages are one payload" ubuntu:26.04 "
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq rpm2cpio cpio zstd >/dev/null 2>&1 ||
    { echo 'installing rpm2cpio/cpio/zstd failed'; exit 1; }
w=\$(mktemp -d); cd \"\$w\"; mkdir deb rpm arch tar
dpkg-deb -x /pkg/$deb deb || { echo \"dpkg-deb -x failed on $deb\"; exit 1; }
# one step per command, so a failure names the step that failed rather than the pair
rpm2cpio /pkg/$rpm > payload.cpio || { echo \"rpm2cpio failed on $rpm (is the rpm package installed?)\"; exit 1; }
(cd rpm && cpio -idm --quiet < ../payload.cpio) || { echo \"cpio failed on $rpm\"; exit 1; }
tar --numeric-owner -C arch -xf <(zstd -dc /pkg/$arch_pkg) --exclude=.PKGINFO --exclude=.MTREE --exclude=.BUILDINFO ||
    { echo 'pacman package extraction failed'; exit 1; }
tar --numeric-owner -C tar -xf <(zstd -dc /pkg/$tarball) || { echo 'tarball extraction failed'; exit 1; }
for d in deb rpm arch tar; do
    (cd \$d && find . -mindepth 1 | LC_ALL=C sort | xargs -d'\n' stat -c '%n %a %u:%g %s') > \$d.attrs || exit 1
    (cd \$d && find . -mindepth 1 -type f -exec sha256sum {} + | LC_ALL=C sort -k2) > \$d.files || exit 1
done
bad=0
for d in rpm arch tar; do
    if ! diff -q deb.attrs \$d.attrs >/dev/null; then
        echo \"\$d carries different paths, modes, owners or sizes than the .deb:\"
        diff deb.attrs \$d.attrs | head -6; bad=1
    fi
    if ! diff -q deb.files \$d.files >/dev/null; then
        echo \"\$d carries different bytes than the .deb:\"; diff deb.files \$d.files | head -6; bad=1
    fi
done
[ \$bad = 0 ] || exit 1
echo \"\$(cat deb.attrs | wc -l) paths, owners: \$(cat *.attrs | awk '{print \$3}' | sort -u | tr '\n' ' ')\"
"

printf '%d/%d rows PASS\n' "$((rows - failed))" "$rows"
[ "$failed" = 0 ] || exit 1
