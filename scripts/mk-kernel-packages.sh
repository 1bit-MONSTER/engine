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
# mk-kernel-packages.sh <out-dir> <version> [options]
#
# Builds the kernel-side package set for engine#1280 in the three native families:
#
#   deb   linux-image-<rel> / linux-headers-<rel>   (the OS build's own debs, passed through)
#         amdxdna-1bit-dkms                         our amdxdna (0.18.0 at the pin + the series)
#         ec-su-axb35-dkms                          the AXB35 board's EC driver
#         1bit-npu-config                           memlock limits, the keep-awake udev rule,
#                                                   modules-load.d, the public MOK certificate
#   rpm   1bit-kernel (kernel + 1bit-kernel-devel), amdxdna-1bit-dkms, ec-su-axb35-dkms,
#         1bit-npu-config
#   arch  linux-1bit (+ linux-1bit-headers), amdxdna-1bit-dkms, ec-su-axb35-dkms, 1bit-npu-config
#
# <version> is the release tag of the three 1bit packages (`1`, or a weekly like `2026.45.0`); the
# kernel keeps the version it was built with, read out of its deb.
#
# Inputs, all optional except the kernel debs:
#
#   --kernel-debs DIR   where the OS build left linux-image-*_<ver>_amd64.deb and
#                       linux-headers-*_<ver>_amd64.deb (scripts/mkimage.sh, docs/kernel.md).
#                       All three families carry THAT build: rpm and Arch rearrange the same payload
#                       (modules under /usr/lib/modules, headers where that family's dkms looks),
#                       they do not rebuild the kernel, so the three families ship identical bits.
#                       --skip-kernel builds only the driver and config packages.
#   --driver-src DIR    the amdxdna checkout at the pinned commit (default:
#                       <repo>/third_party/xdna-driver). Nothing is fetched: init the submodule.
#   --ec-src DIR        the ec_su_axb35 source (default: <repo>/os/licenses/ec_su_axb35/src, which
#                       is the pinned upstream commit plus its LICENSE).
#   --mok-cert FILE     the PUBLIC kernel-signing certificate, DER or PEM. Required. A file that
#                       carries a private key is refused by design: the private key never leaves the
#                       signing machine (engine#1280 rule 5).
#   --families LIST     comma-separated subset of deb,rpm,arch (default: all three).
#
# What this script deliberately does not do: it does not install, remove or reload anything, does not
# sign anything (the kernel debs arrive signed, and the DKMS packages sign with the USER's own MOK at
# build time through their distro's dkms), and does not publish. It needs only the packaging tool of
# the family it is building, so each family is proven in that family's container:
# docs/install-any-distro.md.
set -euo pipefail
umask 022

[ $# -ge 2 ] || { sed -n '16,62p' "$0"; exit 2; }
out=$1 ver=$2; shift 2
repo=$(cd "$(dirname "$0")/.." && pwd)

kdebs=''
driver_src=$repo/third_party/xdna-driver
ec_src=$repo/os/licenses/ec_su_axb35/src
mok_cert=''
families=deb,rpm,arch
skip_kernel=0
while [ $# -gt 0 ]; do
    case $1 in
        --kernel-debs) kdebs=$2; shift 2 ;;
        --driver-src)  driver_src=$2; shift 2 ;;
        --ec-src)      ec_src=$2; shift 2 ;;
        --mok-cert)    mok_cert=$2; shift 2 ;;
        --families)    families=$2; shift 2 ;;
        --skip-kernel) skip_kernel=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
case $families in *[!a-z,]*|'') echo "bad --families: $families" >&2; exit 2 ;; esac

# The driver pin and the series, as third_party/patches/amdxdna/README.md defines them: 0001-0005 are
# what the fleet runs (c26's amdxdna.ko is reproducible from exactly this set); 0006-0007 are carried
# and owner-gated, so no package applies them on a user's machine either.
DRV_VER=0.18.0
DRV_PIN=526c6081974ea4f0f7dd22329186c5d6adb8fad5
DRV_SERIES="0001 0002 0004 0005 0003"
DRV_VER_STR="2.21.0_20261104,$DRV_PIN"
EC_VER=0.1
CFG_VER=1
MAINT='bong-water-water-bong <bong-water-water-bong@1bit.gg>'

die() { echo "mk-kernel-packages: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

mkdir -p "$out"; out=$(cd "$out" && pwd)
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT

# ---- the public MOK certificate -------------------------------------------------------------
# Only ever the certificate. The guard is cheap and deliberate: a file carrying a private key would
# otherwise be copied into three packages before anyone looked at it.
[ -n "$mok_cert" ] || die "--mok-cert is required (the PUBLIC kernel-signing certificate)"
[ -f "$mok_cert" ] || die "no such certificate: $mok_cert"
if grep -q 'PRIVATE KEY' "$mok_cert"; then
    die "$mok_cert carries a PRIVATE KEY: that file is never packaged"
fi
if [ "$(head -c1 "$mok_cert" | od -An -tx1 | tr -d ' \n')" = 30 ]; then
    cp "$mok_cert" "$work/MOK-Kernel.der"
elif grep -q 'BEGIN CERTIFICATE' "$mok_cert"; then
    if have openssl; then openssl x509 -in "$mok_cert" -outform DER -out "$work/MOK-Kernel.der"
    else die "a PEM certificate needs openssl to convert it to DER"; fi
else
    die "$mok_cert is neither DER nor PEM"
fi
if have openssl; then
    subj=$(openssl x509 -in "$work/MOK-Kernel.der" -inform DER -noout -subject)
    case $subj in *1bit*) ;; *) echo "warning: $mok_cert subject is '$subj' (expected the 1bit kernel cert)" >&2 ;; esac
    echo "   MOK cert: $subj"
fi

# ---- payload: 1bit-npu-config ---------------------------------------------------------------
cfg=$work/payload/npu-config
mkdir -p "$cfg/etc/security/limits.d" "$cfg/etc/udev/rules.d" "$cfg/etc/modules-load.d" \
         "$cfg/usr/share/1bit/mok" "$cfg/usr/share/doc/1bit-npu-config" "$cfg/usr/share/licenses/1bit-npu-config"
# The NPU runtime is compiled into 1bit and locks command buffers; without unlimited memlock a loaded
# model's allocations fail. The same two lines the box runs.
cat > "$cfg/etc/security/limits.d/99-amdnpu-memlock.conf" <<'EOF'
*    soft    memlock    unlimited
*    hard    memlock    unlimited
EOF
# The XDNA2 NPU (AMD 1022:17f0) must stay out of runtime autosuspend: the driver's resume path
# destroys the firmware context and a dispatch into the recreated one hangs (engine#1051). The engine
# half holds a runtime-PM reference while a model is loaded; this is the other half.
cat > "$cfg/etc/udev/rules.d/90-1bit-npu-power.rules" <<'EOF'
# 1bit (owner 2026-10-12): keep the XDNA2 NPU (AMD 1022:17f0) out of runtime autosuspend so the lane can keep a persistent context.
ACTION=="add|bind", SUBSYSTEM=="pci", ATTR{vendor}=="0x1022", ATTR{device}=="0x17f0", TEST=="power/control", ATTR{power/control}="on"
EOF
# The EC driver loads on every machine that takes the set; the driver itself refuses to attach on a
# board that is not an AXB35 (third_party/patches/ec_su_axb35), so this entry is inert there.
echo ec_su_axb35 > "$cfg/etc/modules-load.d/ec_su_axb35.conf"
chmod 0644 "$cfg"/etc/security/limits.d/*.conf "$cfg"/etc/udev/rules.d/*.rules \
         "$cfg"/etc/modules-load.d/*.conf
install -m 0644 "$work/MOK-Kernel.der" "$cfg/usr/share/1bit/mok/MOK-Kernel.der"
cp "$repo/LICENSE" "$cfg/usr/share/licenses/1bit-npu-config/LICENSE"
cat > "$cfg/usr/share/doc/1bit-npu-config/README.md" <<'EOF'
<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->
# 1bit-npu-config

The runtime configuration the 1bit NPU packages need. Configuration only: no binaries, no modules.

| file | why |
|---|---|
| `/etc/security/limits.d/99-amdnpu-memlock.conf` | the NPU runtime is compiled into `1bit` and locks command buffers; without unlimited memlock a loaded model's allocations fail |
| `/etc/udev/rules.d/90-1bit-npu-power.rules` | keeps the XDNA2 NPU (`1022:17f0`) out of runtime autosuspend, whose resume path loses the firmware context (engine#1051) |
| `/etc/modules-load.d/ec_su_axb35.conf` | loads the AXB35 board's EC driver (fans, EC temperature, EC power mode); the driver attaches only on an AXB35, so it is inert on other boards |
| `/usr/share/1bit/mok/MOK-Kernel.der` | the **public** certificate the 1bit kernel and its modules are signed with, for Secure Boot |

## Secure Boot

The 1bit kernel is signed with the `1bit-MONSTER` kernel key, which no machine has enrolled yet.
Enrol the public half once, and confirm what it says before you enrol it:

    sudo mokutil --test-key /usr/share/1bit/mok/MOK-Kernel.der    # "not enrolled" on a new machine
    openssl x509 -inform DER -in /usr/share/1bit/mok/MOK-Kernel.der -noout -subject
    # subject=CN=strixhalo Secure Boot Kernel, O=1bit-MONSTER
    sudo mokutil --import /usr/share/1bit/mok/MOK-Kernel.der      # sets a password; reboot -> MokManager

The private half never leaves the machine that signs the release, and is part of no package.

The DKMS packages are the other way round: dkms signs with **your** key (`/var/lib/shim-signed/mok` on
Debian/Ubuntu, `/etc/dkms/` elsewhere), so a module built on your machine chains to the MOK you
enrolled for it, not to ours.

## See also

`docs/install-any-distro.md` in the engine repository: the per-family install and how to verify it.
EOF

# ---- payload: amdxdna-1bit-dkms -------------------------------------------------------------
# Source = the engine's submodule (the pin) plus third_party/patches/amdxdna, pruned to what the module
# build reads: drivers/accel/amdxdna, the tools/configure_kernel.sh that generates config_kernel.h by
# feature-testing the TARGET kernel, and include/ (uapi + trace).
adx_ver=$DRV_VER; adx=$work/payload/amdxdna/usr/src/amdxdna-1bit-$adx_ver
mkdir -p "$adx"
if pin=$(git -C "$driver_src" rev-parse HEAD 2>/dev/null); then
    [ "$pin" = "$DRV_PIN" ] || die "$driver_src is at $pin, the engine pins $DRV_PIN"
    git -C "$driver_src" archive HEAD | tar -x -C "$work" --one-top-level=pin
else
    # A plain copy of the tree: what a container sees when the submodule's gitfile points outside the
    # mount, and what a source tarball gives. The pin cannot be checked from it, so say so.
    [ -f "$driver_src/drivers/accel/amdxdna/Kbuild" ] || die "no amdxdna checkout at $driver_src (git submodule update --init third_party/xdna-driver)"
    echo "   driver source: $driver_src (no git metadata: cannot verify the pin, the tree is used as given)" >&2
    mkdir -p "$work/pin"; cp -a "$driver_src"/. "$work/pin/"; rm -rf "$work/pin/.git"
fi
for keep in drivers/accel/amdxdna drivers/accel/tools/configure_kernel.sh include LICENSE.amdnpu; do
    mkdir -p "$adx/$(dirname "$keep")"
    cp -a "$work/pin/$keep" "$adx/$keep"
done
for p in $DRV_SERIES; do
    f=$(ls "$repo"/third_party/patches/amdxdna/"$p"-*.patch)
    patch -d "$adx" -p1 -s < "$f" || die "$(basename "$f") does not apply to the pin: series and pin disagree"
done
# dkms: build through the driver's own target (it generates config_kernel.h for the target kernel),
# install into updates/ so it takes precedence over the in-tree amdxdna the way the OS image does, and
# let dkms sign with the user's own MOK.
cat > "$adx/dkms.conf" <<EOF
PACKAGE_NAME="amdxdna-1bit"
PACKAGE_VERSION="$adx_ver"
BUILD_EXCLUSIVE_KERNEL_MIN="6.10"

MAKE[0]="make -C drivers/accel/amdxdna modules KERNEL_VER=\${kernelver} ARCH=x86_64 MODULE_VER_STR='$DRV_VER_STR'"
CLEAN="make -C drivers/accel/amdxdna clean KERNEL_VER=\${kernelver}"

BUILT_MODULE_NAME[0]="amdxdna"
BUILT_MODULE_LOCATION[0]="drivers/accel/amdxdna"
DEST_MODULE_LOCATION[0]="/updates"

AUTOINSTALL="yes"
EOF
cat > "$adx/README.dkms" <<EOF
amdxdna-1bit $adx_ver — AMD's XDNA driver at $DRV_PIN plus the engine's series.

Patches applied, in this order (third_party/patches/amdxdna in the engine repository):
$(for p in $DRV_SERIES; do basename "$(ls "$repo"/third_party/patches/amdxdna/"$p"-*.patch)"; done | sed 's/^/  /')

0006 and 0007 (the imported dma-buf fixes, engine#812) are carried in the repository and are NOT
applied here: loading them is the owner's call, so a package does not decide it.

Build: make -C drivers/accel/amdxdna modules KERNEL_VER=<ver> ARCH=x86_64 MODULE_VER_STR='$DRV_VER_STR'
       (drivers/accel/tools/configure_kernel.sh generates config_kernel.h for the target kernel)

Source offer: https://github.com/amd/xdna-driver at $DRV_PIN (GPL-2.0, LICENSE.amdnpu) plus the
patch files named above (Apache-2.0); both are in https://github.com/1bit-MONSTER/engine.
EOF

# ---- payload: ec-su-axb35-dkms --------------------------------------------------------------
[ -f "$ec_src/src/ec_su_axb35.c" ] || die "no ec_su_axb35 source at $ec_src"
ec=$work/payload/ec/usr/src/ec-su-axb35-$EC_VER
mkdir -p "$ec"
cp -a "$ec_src"/. "$ec"/
for drop in contrib python-gui hwmon scripts .git .gitignore; do rm -rf "${ec:?}/$drop"; done
for f in "$repo"/third_party/patches/ec_su_axb35/[0-9]*.patch; do
    patch -d "$ec" -p1 -s < "$f" || die "$(basename "$f") does not apply to $ec_src"
done
cat > "$ec/README.dkms" <<EOF
ec-su-axb35 $EC_VER — the AXB35-02 board's embedded-controller driver (fans, EC temperature, EC power
mode), GPL-2.0, from https://github.com/cmetz/ec-su_axb35-linux at f62c2c22 (the source in
os/licenses/ec_su_axb35/src, LICENSE beside it), plus the DMI gate in
third_party/patches/ec_su_axb35/0001 in the engine repository: the driver attaches only when the board
name contains "AXB35", so the package is harmless on other machines.

Build: make KERNEL_BUILD=<kernel build dir> modules
EOF
cat > "$ec/dkms.conf" <<EOF
PACKAGE_NAME="ec-su-axb35"
PACKAGE_VERSION="$EC_VER"
BUILD_EXCLUSIVE_KERNEL_MIN="6.1"

MAKE[0]="make KERNEL_BUILD=\${kernel_source_dir} modules"
CLEAN="make KERNEL_BUILD=\${kernel_source_dir} clean"

BUILT_MODULE_NAME[0]="ec_su_axb35"
DEST_MODULE_LOCATION[0]="/updates"

AUTOINSTALL="yes"
EOF

# A .deb is an ar archive with control.tar.* and data.tar.*: reading it needs ar, tar and the stream's
# decompressor, which every family has. dpkg-deb is used for BUILDING a deb and for nothing else, so
# the rpm and Arch builds do not need dpkg installed.
_deb_member() { ar t "$1" | grep "^$2\\.tar" | head -1; }
_deb_stream() {   # <deb> <control|data>: the member, decompressed to stdout
    local m; m=$(_deb_member "$1" "$2")
    [ -n "$m" ] || die "no $2.tar.* in $1"
    case ${m##*.} in
        zst) ar p "$1" "$m" | zstd -dc ;;
        xz)  ar p "$1" "$m" | xz -dc ;;
        gz)  ar p "$1" "$m" | gzip -dc ;;
        *)   die "unknown compression in $1: $m" ;;
    esac
}
deb_field() { _deb_stream "$1" control | tar -xO ./control | sed -n "s/^$2: //p" | head -1; }
deb_extract() { mkdir -p "$2"; _deb_stream "$1" data | tar -x -C "$2"; }

# ---- the kernel payload, and the two families that rearrange it -------------------------------
drel=; kimg=; kdevel=
if [ "$skip_kernel" = 0 ]; then
    [ -n "$kdebs" ] || die "--kernel-debs is required (or --skip-kernel for the driver packages only)"
    set -- "$kdebs"/linux-image-*.deb
    [ $# -eq 1 ] && [ -f "$1" ] || die "expected exactly one linux-image-*.deb in $kdebs"
    kimg=$1
    set -- "$kdebs"/linux-headers-*.deb
    [ $# -eq 1 ] && [ -f "$1" ] || die "expected exactly one linux-headers-*.deb in $kdebs"
    kdevel=$1
    drel=$(deb_field "$kimg" Package); drel=${drel#linux-image-}
    dver=$(deb_field "$kimg" Version)
    [ -n "$drel" ] && [ -n "$dver" ] || die "cannot read Package/Version out of $kimg"
    echo "   kernel: $drel ($dver)"
    deb_extract "$kimg" "$work/kernel/image"
    deb_extract "$kdevel" "$work/kernel/headers"
    [ -f "$work/kernel/image/boot/vmlinuz-$drel" ] || die "the image deb has no /boot/vmlinuz-$drel"
    [ -d "$work/kernel/image/lib/modules/$drel" ] || die "the image deb carries no module tree for $drel"
    [ -d "$work/kernel/headers/usr/src/linux-headers-$drel" ] || die "the headers deb has no /usr/src/linux-headers-$drel"
    # rpm/Arch: same payload, that family's layout. /lib is a symlink to /usr/lib on both.
    krel=$work/kernel/relocated
    mkdir -p "$krel/usr/lib/modules" "$krel/boot"
    cp -a "$work/kernel/image/lib/modules/$drel" "$krel/usr/lib/modules/"
    for f in vmlinuz config System.map; do cp -a "$work/kernel/image/boot/$f-$drel" "$krel/boot/"; done
fi

# ---- emitters --------------------------------------------------------------------------------
# deb: a control file and dpkg-deb. /etc files are conffiles, so a user's edit survives an upgrade.
mk_deb() {   # <root> <name> <version> <desc-one-line> <body> [depends]
    local root=$1 name=$2 version=$3 desc=$4 body=$5 depends=${6:-} size
    size=$(du -sk --exclude=DEBIAN "$root" | cut -f1)
    mkdir -p "$root/DEBIAN"
    if [ -d "$root/etc" ]; then
        find "$root/etc" -type f | sed "s|^$root||" > "$root/DEBIAN/conffiles"
    else
        rm -f "$root/DEBIAN/conffiles"
    fi
    {   echo "Package: $name"
        echo "Version: $version"
        echo "Architecture: all"
        echo "Maintainer: $MAINT"
        echo "Installed-Size: $size"
        echo "Section: kernel"
        echo "Priority: optional"
        [ -z "$depends" ] || echo "Depends: $depends"
        echo "Description: $desc"
        printf '%s\n' "$body" | sed -e 's/^/ /' -e 's/^ $/ ./'
    } > "$root/DEBIAN/control"
    dpkg-deb --root-owner-group --build "$root" "$out/${name}_${version}_all.deb" >/dev/null
    echo "   ${name}_${version}_all.deb"
}

# rpm: a generated spec whose %install copies the staged tree, so the payload is the bytes the deb
# carries and only the metadata and layout differ. Strip and build-id are off: the payload holds the
# build's own binaries (vmlinuz, signed .ko) and rewriting them would be a lie about what was built.
mk_rpm() {   # <name> <version> <release> <summary> <license> <root> [post] [requires] [preun]
    local name=$1 version=$2 release=$3 summary=$4 lic=$5 root=$6 post=${7:-} req=${8:-} preun=${9:-} spec f
    if ! have rpmbuild; then die "rpmbuild is not installed: build the rpm family in a Fedora container"; fi
    mkdir -p "$work/rpm/SPECS" "$work/rpm/BUILD" "$work/rpm/RPMS" "$work/rpm/SRPMS"
    spec=$work/rpm/SPECS/$name.spec
    {   echo "%global __strip /bin/true"
        echo "%global debug_package %{nil}"
        echo "%global _build_id_links none"
        echo "%global _binary_payload w19.zstdio"
        echo "Name: $name"
        echo "Version: $version"
        echo "Release: $release"
        echo "Summary: $summary"
        echo "License: $lic"
        echo "URL: https://github.com/1bit-MONSTER/engine"
        # The payload's binaries are the build's own (signed .ko, vmlinuz, a prebuilt kernel tree);
        # letting rpm re-derive Requires from them would invent dependencies on the build host's
        # libraries. AutoReq AND AutoProv, per package: rpm 6 still generates for the subpackages.
        echo "AutoReqProv: no"
        echo "AutoReq: no"
        echo "AutoProv: no"
        [ -z "$req" ] || echo "Requires: $req"
        echo "%description"
        echo "$summary"
        echo "%install"
        echo "mkdir -p %{buildroot}"
        echo "cp -a $root/. %{buildroot}/"
        echo "%files"
        # %dir only for the package's own deep directories: /boot, /usr/lib/modules and friends belong
        # to the `filesystem` package, and claiming them makes rpm refuse the install (measured on
        # fc45: "file /usr/lib from install of 1bit-kernel conflicts with file from package filesystem").
        ( cd "$root" && find . -mindepth 1 -type d -printf '/%P\n' | awk -F/ 'NF>=5' | sed 's|^|%dir |' )
        for f in $(cd "$root" && find . -mindepth 1 \( -type f -o -type l \) -printf '/%P\n'); do
            case $f in /etc/*) echo "%config(noreplace) $f" ;; *) echo "$f" ;; esac
        done
        if [ -n "$post" ]; then echo "%post"; echo "$post"; fi
        if [ -n "$preun" ]; then echo "%preun"; echo "$preun"; fi
        echo "%changelog"
        echo "* $(date '+%a %b %d %Y') $MAINT - $version-$release"
        echo "- $name $version-$release"
    } > "$spec"
    rpmbuild -bb --quiet --define "_topdir $work/rpm" "$spec" || die "rpmbuild failed for $name"
    cp "$work/rpm/RPMS"/*/"$name-$version-$release".*.rpm "$out"/
    echo "   $(basename "$(ls "$out"/"$name-$version-$release".*.rpm | tail -1)")"
}

# arch: the payload plus .PKGINFO/.MTREE (and .INSTALL where dkms has to register itself), tarred and
# zstd'd. Writing the format directly instead of through makepkg is what lets one staged payload feed
# all three families; pacman installs the result like any other package.
mk_arch() {   # <name> <pkgver> <pkgdesc> <license> <root> [install-script]
    local name=$1 pkgver=$2 desc=$3 lic=$4 root=$5 inst=${6:-}
    if ! have bsdtar || ! have zstd; then die "bsdtar and zstd are needed for the arch family"; fi
    if [ -e "$root/.PKGINFO" ]; then die "$root already has a .PKGINFO"; fi
    {   echo "# Generated by mk-kernel-packages.sh - do not edit"
        echo "pkgname = $name"
        echo "pkgbase = $name"
        echo "pkgver = $pkgver"
        echo "pkgdesc = $desc"
        echo "url = https://github.com/1bit-MONSTER/engine"
        echo "builddate = $(date +%s)"
        echo "packager = $MAINT"
        echo "size = $(du -sb --exclude=.PKGINFO --exclude=.INSTALL --exclude=.MTREE "$root" | cut -f1)"
        echo "arch = x86_64"
        echo "license = $lic"
        [ -z "$inst" ] || echo "install = .INSTALL"
    } > "$root/.PKGINFO"
    if [ -n "$inst" ]; then printf '%s\n' "$inst" > "$root/.INSTALL"; fi
    ( cd "$root" && bsdtar -cf - --format=mtree \
        --options='!all,use-set,type,uid,gid,mode,time,size,md5,sha256,link' . | gzip -9 > .MTREE )
    local extra=()
    [ -z "$inst" ] || extra=(.INSTALL)
    ( cd "$root" && bsdtar --format=pax -cf - .PKGINFO .MTREE "${extra[@]}" * \
        | zstd -q -19 -T0 -o "$out/$name-$pkgver-x86_64.pkg.tar.zst" )
    echo "   $name-$pkgver-x86_64.pkg.tar.zst"
}

# ---- 1bit-npu-config ------------------------------------------------------------------------
cfg_desc="NPU runtime configuration for the 1bit NPU drivers (memlock, keep-awake, modules-load, MOK)"
case ",$families," in *,deb,*)
    mk_deb "$cfg" 1bit-npu-config "$CFG_VER-$ver" "$cfg_desc" \
        "The configuration the 1bit NPU packages need, and nothing else: unlimited memlock in
limits.d for the NPU runtime's locked command buffers, a udev rule that keeps the XDNA2 NPU out of
runtime autosuspend, modules-load.d for the AXB35 board's EC driver, and the public MOK certificate
the 1bit kernel is signed with (mokutil --import, see the README)." ;;
esac
case ",$families," in *,rpm,*)
    mk_rpm 1bit-npu-config "$CFG_VER" "$ver" "$cfg_desc" "Apache-2.0" "$cfg" ;;
esac
case ",$families," in *,arch,*)
    mk_arch 1bit-npu-config "$CFG_VER-$ver" "$cfg_desc" "Apache-2.0" "$cfg" ;;
esac

# ---- amdxdna-1bit-dkms ----------------------------------------------------------------------
# dkms registers the source on install and builds for the headers sets present; on removal it
# unregisters, which puts the in-tree amdxdna back. The post-install step must not fail when no
# headers are installed yet (a container, or a machine that adds headers later): dkms builds on the
# next headers install through its own triggers.
adx_root=$work/payload/amdxdna
adx_desc="amdxdna (XDNA2 NPU driver) $DRV_VER at $DRV_PIN plus the engine's series, as DKMS source"
adx_body="AMD's out-of-tree amdxdna driver at $DRV_PIN with the engine's patch series (the crash,
TDR-recovery and partition fixes this hardware needs; 0006/0007 are carried but not applied).
Installs as DKMS source and builds against YOUR kernel's headers, signing with your own MOK, so it
works on any 6.10+ kernel and not only on the 1bit kernel. The module installs to
/lib/modules/<ver>/updates/, which takes precedence over the in-tree amdxdna; uninstalling the
package removes it and the in-tree driver is used again."
adx_post="dkms add -m amdxdna-1bit -v $adx_ver >/dev/null 2>&1 || :
dkms autoinstall -m amdxdna-1bit -v $adx_ver >/dev/null 2>&1 || :
echo 'amdxdna-1bit: dkms build for the running kernel is best-effort here; check: dkms status'"
adx_preun="dkms remove -m amdxdna-1bit -v $adx_ver --all >/dev/null 2>&1 || :"
case ",$families," in *,deb,*)
    mkdir -p "$adx_root/DEBIAN"
    printf '%s\n' '#!/bin/sh' 'set -e' "$adx_post" > "$adx_root/DEBIAN/postinst"
    printf '%s\n' '#!/bin/sh' 'set -e' "$adx_preun" > "$adx_root/DEBIAN/prerm"
    chmod 0755 "$adx_root/DEBIAN/postinst" "$adx_root/DEBIAN/prerm"
    mk_deb "$adx_root" amdxdna-1bit-dkms "$adx_ver-$ver" "$adx_desc" "$adx_body" dkms ;;
esac
case ",$families," in *,rpm,*)
    mk_rpm amdxdna-1bit-dkms "$adx_ver" "$ver" "$adx_desc" "GPL-2.0 AND Apache-2.0" "$adx_root" "$adx_post" dkms "$adx_preun" ;;
esac
case ",$families," in *,arch,*)
    mk_arch amdxdna-1bit-dkms "$adx_ver-$ver" "$adx_desc" "GPL-2.0" "$adx_root" \
"post_install() {
    dkms add -m amdxdna-1bit -v $adx_ver >/dev/null 2>&1 || :
    dkms autoinstall -m amdxdna-1bit -v $adx_ver >/dev/null 2>&1 || :
    echo 'amdxdna-1bit: dkms status shows the build state; a machine without headers yet builds later'
}
post_upgrade() { post_install; }
pre_remove() { dkms remove -m amdxdna-1bit -v $adx_ver --all >/dev/null 2>&1 || :; }
post_remove() { pre_remove; }" ;;
esac

# ---- ec-su-axb35-dkms -----------------------------------------------------------------------
ec_root=$work/payload/ec
ec_desc="AXB35-02 embedded-controller driver (fans, EC temperature, power mode) as DKMS source"
ec_body="The Sixunited AXB35-02 board's EC driver, which the kernel has no driver for: fan curve and
tachometers, EC temperature, EC power mode. Installs as DKMS source and builds against YOUR kernel's
headers, signing with your own MOK. The driver asks the DMI board name at load and attaches only on an
AXB35 board, so installing it anywhere is harmless. Upstream
https://github.com/cmetz/ec-su_axb35-linux, GPL-2.0-only, source shipped in the package."
ec_post="dkms add -m ec-su-axb35 -v $EC_VER >/dev/null 2>&1 || :
dkms autoinstall -m ec-su-axb35 -v $EC_VER >/dev/null 2>&1 || :
echo 'ec-su-axb35: attaches only on an AXB35 board; dkms status shows the build state'"
ec_preun="dkms remove -m ec-su-axb35 -v $EC_VER --all >/dev/null 2>&1 || :"
case ",$families," in *,deb,*)
    mkdir -p "$ec_root/DEBIAN"
    printf '%s\n' '#!/bin/sh' 'set -e' "$ec_post" > "$ec_root/DEBIAN/postinst"
    printf '%s\n' '#!/bin/sh' 'set -e' "$ec_preun" > "$ec_root/DEBIAN/prerm"
    chmod 0755 "$ec_root/DEBIAN/postinst" "$ec_root/DEBIAN/prerm"
    mk_deb "$ec_root" ec-su-axb35-dkms "$EC_VER-$ver" "$ec_desc" "$ec_body" dkms ;;
esac
case ",$families," in *,rpm,*)
    mk_rpm ec-su-axb35-dkms "$EC_VER" "$ver" "$ec_desc" "GPL-2.0-only" "$ec_root" "$ec_post" dkms "$ec_preun" ;;
esac
case ",$families," in *,arch,*)
    mk_arch ec-su-axb35-dkms "$EC_VER-$ver" "$ec_desc" "GPL-2.0-only" "$ec_root" \
"post_install() {
    dkms add -m ec-su-axb35 -v $EC_VER >/dev/null 2>&1 || :
    dkms autoinstall -m ec-su-axb35 -v $EC_VER >/dev/null 2>&1 || :
    echo 'ec-su-axb35: attaches only on an AXB35 board; dkms status shows the build state'
}
post_upgrade() { post_install; }
pre_remove() { dkms remove -m ec-su-axb35 -v $EC_VER --all >/dev/null 2>&1 || :; }
post_remove() { pre_remove; }" ;;
esac

# ---- the kernel ------------------------------------------------------------------------------
if [ "$skip_kernel" = 0 ]; then
    # deb: the OS build's own packages, unchanged. They are what the stick and this box run and they
    # are already signed with the 1bit MOK; a second build of the same kernel would be a second thing
    # to keep in step. --skip-kernel is the only way not to ship them.
    case ",$families," in *,deb,*)
        cp "$kimg" "$kdevel" "$out"/
        echo "   $(basename "$kimg")  (as built)"
        echo "   $(basename "$kdevel")  (as built)"
        ;;
    esac
    arch_ver=$(printf '%s' "${drel%-1bit}" | sed 's/-rc/rc/')-1      # 7.4.0-rc1-1bit -> 7.4.0rc1-1
    rpm_ver=${dver%-*}; rpm_rel=${dver##*-}                          # 7.4.0~rc1-1 -> 7.4.0~rc1 / 1
    case ",$families," in *,rpm,*)
        if ! have rpmbuild; then die "rpmbuild is not installed: build the rpm family in a Fedora container"; fi
        rel=$work/kernel/rpm
        mkdir -p "$rel/usr/lib/modules" "$rel/usr/src/kernels"
        cp -a "$krel/usr/lib/modules/$drel" "$rel/usr/lib/modules/"
        cp -a "$krel/boot" "$rel/boot"
        cp -a "$work/kernel/headers/usr/src/linux-headers-$drel" "$rel/usr/src/kernels/$drel"
        ln -s "/usr/src/kernels/$drel" "$rel/usr/lib/modules/$drel/build"
        mkdir -p "$work/rpm/SPECS" "$work/rpm/BUILD" "$work/rpm/RPMS" "$work/rpm/SRPMS"
        kspec=$work/rpm/SPECS/1bit-kernel.spec
        {   echo "%global __strip /bin/true"
            echo "%global debug_package %{nil}"
            echo "%global _build_id_links none"
            echo "%global _binary_payload w19.zstdio"
            echo "Name: 1bit-kernel"
            echo "Version: $rpm_ver"
            echo "Release: $rpm_rel"
            echo "Summary: The 1bit kernel ($drel): Linux 7.4-rc1, our config, signed with the 1bit MOK"
            echo "License: GPL-2.0-only AND (GPL-2.0-only WITH Linux-syscall-note)"
            echo "URL: https://github.com/1bit-MONSTER/engine"
            echo "AutoReqProv: no"
            echo "AutoReq: no"
            echo "AutoProv: no"
            # No `Provides: kernel-uname-r`: rpm cannot express a version carrying a '-' the way our
            # localversion does (7.4.0-rc1-1bit is version 7.4.0, release rc1-1bit, and rpm calls the
            # second separator invalid). dkms finds the build tree by path — /lib/modules/<rel>/build —
            # not by this capability, so nothing that uses these packages needs it.
            echo "%description"
            echo "The kernel the 1bit OS boots: Linux 7.4-rc1 with this fleet's configuration (PerfOpt,"
            echo "BTF, the NPU and board firmware, the 1bit localversion), signed with the 1bit-MONSTER"
            echo "kernel MOK. Optional: a machine that stays on its distribution kernel needs only the"
            echo "driver packages. Run depmod/dracut and your bootloader's own tools after installing."
            echo "%package devel"
            echo "Summary: Headers and build files for the 1bit kernel, for dkms"
            echo "AutoReqProv: no"
            echo "AutoReq: no"
            echo "AutoProv: no"
            echo "Requires: 1bit-kernel = $rpm_ver-$rpm_rel"
            echo "%description devel"
            echo "The kernel build tree dkms uses to build amdxdna-1bit-dkms and ec-su-axb35-dkms"
            echo "against this kernel."
            echo "%install"
            echo "mkdir -p %{buildroot}"
            echo "cp -a $rel/. %{buildroot}/"
            echo "%files"
            ( cd "$rel" && find . -mindepth 1 -type d -printf '/%P\n' | awk -F/ 'NF>=5' \
                | grep -v "^/usr/src/kernels" | sed 's|^|%dir |' )
            ( cd "$rel" && find . -mindepth 1 \( -type f -o -type l \) -printf '/%P\n' \
                | grep -v '^/usr/src/' | grep -v "^/usr/lib/modules/$drel/build\$" )
            echo "%files devel"
            echo "%dir /usr/src/kernels"
            echo "/usr/src/kernels/$drel"
            echo "/usr/lib/modules/$drel/build"
            echo "%post"
            echo "if [ -x /usr/sbin/depmod ]; then /usr/sbin/depmod -a $drel || :; fi"
            # no depmod after removal: the tree it indexes is gone with the package (measured on fc45:
            # "depmod: ERROR: could not open directory /lib/modules/<rel>"); depmod is install-time only
            echo "%postun"
            echo "if [ -d /lib/modules/$drel ] && [ -x /usr/sbin/depmod ]; then /usr/sbin/depmod -a $drel || :; fi"
            echo "%changelog"
            echo "* $(date '+%a %b %d %Y') $MAINT - $rpm_ver-$rpm_rel"
            echo "- 1bit-kernel $rpm_ver-$rpm_rel ($drel)"
        } > "$kspec"
        rpmbuild -bb --quiet --define "_topdir $work/rpm" "$kspec" || die "rpmbuild failed for 1bit-kernel"
        cp "$work/rpm/RPMS"/*/1bit-kernel-*.rpm "$out"/
        echo "   1bit-kernel-${rpm_ver}-${rpm_rel}.x86_64.rpm 1bit-kernel-devel-${rpm_ver}-${rpm_rel}.x86_64.rpm"
        ;;
    esac
    case ",$families," in *,arch,*)
        rel=$work/kernel/arch
        mkdir -p "$rel/usr/lib/modules/$drel" "$rel/boot" "$rel/etc/mkinitcpio.d"
        cp -a "$krel/usr/lib/modules/$drel/." "$rel/usr/lib/modules/$drel/"
        # Arch names a kernel by suffix: vmlinuz-linux-<suffix>, and mkinitcpio needs a matching preset.
        cp -a "$krel/boot/vmlinuz-$drel" "$rel/boot/vmlinuz-1bit"
        cp -a "$krel/boot/config-$drel" "$rel/boot/config-1bit"
        cp -a "$krel/boot/System.map-$drel" "$rel/boot/System.map-1bit"
        cat > "$rel/etc/mkinitcpio.d/1bit.preset" <<EOF
# Generated by mk-kernel-packages.sh (engine#1280). The kernel package is linux-1bit.
ALL_kver="/boot/vmlinuz-1bit"
PRESETS=('default')
default_image="/boot/initramfs-1bit.img"
EOF
        mk_arch linux-1bit "$arch_ver" \
            "The 1bit kernel ($drel): Linux 7.4-rc1, our config, signed with the 1bit MOK" \
            "GPL-2.0-only" "$rel" \
"post_install() {
    if command -v mkinitcpio >/dev/null 2>&1; then
        echo 'linux-1bit: run \"mkinitcpio -P\" to build /boot/initramfs-1bit.img, then add a boot entry'
    fi
}"
        # the headers are a separate Arch package: a real directory at modules/<rel>/build, where Arch's
        # dkms and every out-of-tree build look
        arel=$work/kernel/arch-headers
        mkdir -p "$arel/usr/lib/modules/$drel"
        cp -a "$work/kernel/headers/usr/src/linux-headers-$drel/." "$arel/usr/lib/modules/$drel/build/"
        mk_arch linux-1bit-headers "$arch_ver" \
            "Headers and build files for the 1bit kernel, for dkms" "GPL-2.0-only" "$arel"
        ;;
    esac
fi

echo
echo "== $out"
( cd "$out" && sha256sum ./*.deb ./*.rpm ./*.pkg.tar.zst 2>/dev/null || : ) | sed 's/^/   /'
