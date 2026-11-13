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
# pack-distro.sh <format> <root> <out-dir>
#
# One prepared payload tree, wrapped in one distribution format. <root> IS the installed tree: every
# path in it is final (/opt/1bit/engine/1bit lands at /opt/1bit/engine/1bit), so the caller
# assembles the payload once and every family installs the same bytes at the same paths. It is copied
# into a working directory and never modified in place.
# mk-plugin-package.sh is the engine's caller; the kernel and driver packages call it the same way
# (engine #1279, #1280).
#
# format: deb | rpm | arch | tarball
#
# What each family declares about the package - name, version, dependencies, licence - stays with
# that family: the .rpm's and the Arch package's from the recipes in packaging/ (this repository's
# own, so a maintainer of either family reads their file, not this script), the .deb's from the
# environment, since Debian has no recipe file here.
#
# Environment:
#   PKG_VERSION     the release version                           (required)
#   PKG_RELEASE     rpm Release / pacman pkgrel                    (default 1)
#   PKG_RPM_SPEC    default <repo>/packaging/rpm/<PKG_NAME>.spec   (for rpm)
#   PKG_ARCH_PKGBUILD   default <repo>/packaging/arch/PKGBUILD     (for arch)
#   and, for deb only:
#   PKG_NAME, PKG_SUMMARY, PKG_DESC, PKG_MAINTAINER                (required)
#   PKG_DEB_DEPENDS, PKG_DEB_RECOMMENDS                            (optional, comma separated)
#
# The version is normalized to what every family accepts, because the tag a cut carries is not one
# (scripts/pack-version.sh: rc-2026-11-07-c32 packs as 2026.11.07.c32, and a tag with no digit at all as
# 0.<tag>). It is the same function scripts/pack-proof.sh uses to find the packages again.
#
# deb and tarball are built here. rpm and arch are built in containers (fedora:latest,
# archlinux:latest): rpmbuild and makepkg are the only correct builders for those two formats and the
# release box has neither, and the image pins what builds them.
#
# The produced file's path is the only thing on stdout.
set -euo pipefail

# usage(): the header comment as prose, anchored on the comment block rather than a line range -- two of these ranges had
# drifted past the end of their header and printed code to whoever asked for help (engine#2129).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

[ $# -eq 3 ] || { usage; exit 2; }
fmt=$1 root=$2 out=$3
repo=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=scripts/pack-version.sh
. "$(dirname "$0")/pack-version.sh"
# shellcheck source=scripts/pack-elf.sh
. "$(dirname "$0")/pack-elf.sh"      # pack_normalise_modes, pack_writable_files

# The payload is a system tree, so its modes are the ones a distribution expects, and that rule is the
# fleet's (#1352), not this script's: pack_normalise_modes in scripts/pack-elf.sh, the same helper
# package-linux.sh and the lemonade fold use, so the three packing sites cannot drift. A tree staged under
# a 002 umask is group-writable and rpm refuses it outright - /usr/bin at 0775 conflicts with the
# filesystem package's 0755, which is how this was found: `dnf install` on fedora:latest failed on a
# payload that had installed cleanly as a .deb and a pacman package. It runs on the formatter's own copy,
# never on the caller's tree (the kernel packages share this script, #1280).
#
# No assertion follows it: the copy is ours, so the chmod cannot fail, and a check that cannot fire is
# decoration. `pack_writable_files` (the reporter the fold refuses with) was tried here and dropped for
# that reason - a root-owned 0664 file handed in as the payload comes out 0644, because cp makes it ours
# first. The mode is checked where it can actually be wrong: on the finished archives, by
# scripts/pack-proof.sh.

need() { [ -n "${!1:-}" ] || { echo "pack-distro.sh: $1 is not set ($2)" >&2; exit 2; }; }
need PKG_VERSION "the release version"
[ -d "$root" ] || { echo "pack-distro.sh: no payload tree at $root" >&2; exit 2; }
[ -n "$(ls -A "$root")" ] || { echo "pack-distro.sh: the payload tree at $root is empty" >&2; exit 2; }

# the version and how a tag becomes one lives in one place, shared with the proof that looks the
# packages up by name (scripts/pack-version.sh); it is per family, because a candidate's version has to
# sort below its release and only pacman needs a different spelling to do that
rel=${PKG_RELEASE:-1}

# The distribution's own filesystem skeleton: its filesystem package owns these paths, so no package
# claims them and the tarball's member list leaves them out too. A member for `/` or `/usr` is what
# lets `tar` extracted as root rewrite the ownership or the mode of a directory that is not ours.
skeleton='^(usr|usr/bin|usr/sbin|usr/lib|usr/lib64|usr/libexec|usr/include|usr/share|usr/share/doc|usr/share/man|usr/local|etc|var|var/lib|opt|run|srv|media|mnt)$'

# the members a package carries: every path in the payload, minus that skeleton. Relative, sorted, and
# without the "./" member tar writes for the top directory itself.
payload_members() {  # payload_members <tree>
    (cd "$1" && find . -mindepth 1 | sed -e 's,^\./,,' | LC_ALL=C sort | { grep -Ev "$skeleton" || true; })
}

mkdir -p "$out"
out=$(cd "$out" && pwd)

# a working directory on the payload's own filesystem (every formatter copies the payload into it and
# the containers mount it)
work=$(mktemp -d "$(dirname "$root")/.pack-distro.XXXXXX")
trap 'rm -rf "$work"' EXIT

case $fmt in
deb)
    need PKG_NAME "the package name"
    need PKG_SUMMARY "the one-line summary"
    need PKG_DESC "the description"
    need PKG_MAINTAINER "the maintainer, \"Name <mail>\""
    [ ! -e "$root/DEBIAN" ] || { echo "pack-distro.sh: $root already has a DEBIAN directory" >&2; exit 2; }
    ver=$(pack_version "$PKG_VERSION" deb) || exit 2
    # a real copy, not hardlinks: the modes are normalised inside it, and the caller's tree is left as
    # it was
    cp -a "$root" "$work/root"
    pack_normalise_modes "$work/root"
    size=$(du -sk "$work/root" | cut -f1)
    mkdir -p "$work/root/DEBIAN"
    {
        echo "Package: $PKG_NAME"
        echo "Version: $ver"
        echo "Architecture: amd64"
        echo "Maintainer: $PKG_MAINTAINER"
        echo "Installed-Size: $size"
        echo "Section: misc"
        echo "Priority: optional"
        if [ -n "${PKG_DEB_DEPENDS:-}" ]; then echo "Depends: ${PKG_DEB_DEPENDS}"; fi
        if [ -n "${PKG_DEB_RECOMMENDS:-}" ]; then echo "Recommends: ${PKG_DEB_RECOMMENDS}"; fi
        echo "Description: $PKG_SUMMARY"
        # Debian continuation lines: one leading space, and a blank line is " ."
        printf '%s\n' "$PKG_DESC" | sed -e 's/^[[:space:]]*$/./' -e 's/^/ /'
    } > "$work/root/DEBIAN/control"
    f="$out/${PKG_NAME}_${ver}_amd64.deb"
    dpkg-deb --root-owner-group --build "$work/root" "$f" >/dev/null
    ;;
tarball)
    # the name keeps the name the guide has always used for this route
    ver=$(pack_version "$PKG_VERSION" tarball) || exit 2
    f="$out/${PKG_NAME:-1bit}-${ver}-linux-x86_64.tar.zst"
    cp -a "$root" "$work/root"
    pack_normalise_modes "$work/root"
    payload_members "$work/root" > "$work/members"
    # --owner/--group/--numeric-owner: every member is root's, so extracting as root cannot hand a
    # payload path to the builder's uid. Before this, `tar -C / -xf` as root left uid 1000 owning /,
    # /usr, /usr/bin and /opt (review of #1279). -T with --no-recursion: the member list is the
    # package's, the filesystem skeleton is not in it, and no "./" member can appear.
    tar -C "$work/root" --owner=0 --group=0 --numeric-owner --no-recursion \
        -cf - -T "$work/members" | zstd -q -T4 -19 -o "$f"
    # and it is checked, not assumed: a member that is not root's, or a "./" member, fails the build
    bad=$(tar --zstd --numeric-owner -tvf "$f" | awk '$2 != "0/0" {print; exit}' || true)
    [ -z "$bad" ] || { echo "pack-distro.sh: tarball member not owned by root: $bad" >&2; exit 1; }
    bad=$(tar --zstd -tf "$f" | grep -xE '\./?' || true)
    [ -z "$bad" ] || { echo "pack-distro.sh: the tarball carries a '$bad' member" >&2; exit 1; }
    ;;
rpm)
    need PKG_NAME "the package name"
    spec=${PKG_RPM_SPEC:-$repo/packaging/rpm/$PKG_NAME.spec}
    [ -f "$spec" ] || { echo "pack-distro.sh: no spec at $spec" >&2; exit 2; }
    ver=$(pack_version "$PKG_VERSION" rpm) || exit 2
    p="$PKG_NAME-$ver"
    mkdir -p "$work/$p"
    cp -a "$root/." "$work/$p/"
    pack_normalise_modes "$work/$p"
    tar -C "$work" -czf "$work/$p.tar.gz" "$p"
    # %files -f: every path, directories included (rpm does not read a listed directory as its
    # contents), built from the payload so it can never list a file the payload does not have - except
    # the distribution's own skeleton. Its filesystem package owns /usr, /usr/bin, /usr/lib and the
    # rest of the FHS bases (on Fedora /usr/bin and /usr/lib at 0555), so a package that lists them
    # claims them at its own mode and rpm refuses to install it: "file /usr/bin from install of
    # 1bit-engine conflicts with file from package filesystem". What is left is what the payload
    # really creates - the same member list the tarball is built from, with a leading slash.
    payload_members "$work/$p" | sed -e 's,^,/,' > "$work/files.list"
    [ -s "$work/files.list" ] || { echo "pack-distro.sh: the payload's file list came out empty" >&2; exit 1; }
    cp "$spec" "$work/"
    command -v docker >/dev/null ||
        { echo "pack-distro.sh: the .rpm is built with rpmbuild in a fedora container: no docker on PATH" >&2; exit 3; }
    # a script in the bind mount rather than a one-liner: it is what ran, and it is readable when it fails
    cat > "$work/in-container.sh" <<'SH'
set -euo pipefail
# rpmbuild writes as root in here; hand the tree back so the caller can read its output and clean up
trap 'chown -R "$HOSTUID:$HOSTGID" /build' EXIT
command -v rpmbuild >/dev/null || dnf -y -q install rpm-build >/dev/null
rpmbuild -bb --define "_topdir /build/rpmbuild" --define "_sourcedir /build" \
    --define "ver $VER" --define "rel $REL" --define "chlogdate $CHLOGDATE" "/build/$SPEC"
SH
    docker run --rm -v "$work:/build" -w /build -e HOSTUID="$(id -u)" -e HOSTGID="$(id -g)" \
        -e VER="$ver" -e REL="$rel" -e SPEC="$(basename "$spec")" \
        -e CHLOGDATE="$(LC_ALL=C date '+%a %b %d %Y')" \
        fedora:45@sha256:a7d72ba2745b100a3351087472e5ee7b2450af7cf331e7673007e3817612839c bash /build/in-container.sh > "$work/rpmbuild.log" 2>&1 ||
        { tail -30 "$work/rpmbuild.log" >&2; exit 1; }
    built=$(find "$work/rpmbuild/RPMS" -name '*.rpm' -type f | head -1)
    [ -n "$built" ] || { echo "pack-distro.sh: rpmbuild produced no package" >&2; exit 1; }
    f="$out/$(basename "$built")"
    cp "$built" "$f"
    ;;
arch)
    pkgbuild=${PKG_ARCH_PKGBUILD:-$repo/packaging/arch/PKGBUILD}
    [ -f "$pkgbuild" ] || { echo "pack-distro.sh: no PKGBUILD at $pkgbuild" >&2; exit 2; }
    ver=$(pack_version "$PKG_VERSION" arch) || exit 2
    mkdir -p "$work/payload"
    cp -a "$root/." "$work/payload/"
    pack_normalise_modes "$work/payload"
    cp "$pkgbuild" "$work/PKGBUILD"
    command -v docker >/dev/null ||
        { echo "pack-distro.sh: the Arch package is built with makepkg in an archlinux container: no docker on PATH" >&2; exit 3; }
    cat > "$work/in-container.sh" <<'SH'
set -euo pipefail
# makepkg builds as an unprivileged user (it refuses root) and fakeroot gives the payload its root
# ownership; the trap hands the tree back to the caller either way
trap 'chown -R "$HOSTUID:$HOSTGID" /build' EXIT
if ! command -v makepkg >/dev/null || ! command -v fakeroot >/dev/null || ! command -v zstd >/dev/null; then
    pacman -Sy --noconfirm --needed base-devel zstd >/dev/null
fi
id -u builder >/dev/null 2>&1 || useradd -m builder
chown -R builder /build
su builder -c 'cd /build && makepkg -f --nodeps --noconfirm'
SH
    docker run --rm -v "$work:/build" -w /build -e HOSTUID="$(id -u)" -e HOSTGID="$(id -g)" \
        -e PKGVER="$ver" -e PKGREL="$rel" \
        archlinux:base-20261108.0.623074@sha256:d4594bb2300cba11f41c2f27923eb56f63d9988909c402f9c95517d1e362c899 bash /build/in-container.sh > "$work/makepkg.log" 2>&1 ||
        { tail -30 "$work/makepkg.log" >&2; exit 1; }
    built=$(find "$work" -maxdepth 1 -name '*.pkg.tar.zst' -type f | head -1)
    [ -n "$built" ] || { echo "pack-distro.sh: makepkg produced no package" >&2; exit 1; }
    f="$out/$(basename "$built")"
    cp "$built" "$f"
    ;;
*)
    echo "pack-distro.sh: unknown format '$fmt' (deb, rpm, arch or tarball)" >&2
    exit 2
    ;;
esac

echo "$f"
