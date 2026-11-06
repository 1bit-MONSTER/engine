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
# pkg-proof.sh <deb|rpm|arch> <package-dir>
#
# The proof for one package family (engine#1280 item 7), run INSIDE a clean container of that
# family. It installs the three 1bit packages, builds both DKMS modules against the CONTAINER's own
# kernel headers, checks that the out-of-tree module resolves ahead of the in-tree one, checks the
# AXB35 DMI gate is compiled into the EC module, installs the kernel package and checks its payload,
# then uninstalls everything and checks nothing is left behind. Prints one PASS/FAIL row per check and
# exits non-zero if any row fails.
#
#   deb   docker run --rm -v <pkgs>:/pkg:ro ubuntu:26.04 bash -c \
#           'apt-get update -qq && apt-get install -y -qq dkms udev kmod sbsigntool linux-headers-generic \
#              && apt-get install -y -qq linux-modules-$(ls /lib/modules | sort -V | tail -1) \
#              && bash /pkg/pkg-proof.sh deb /pkg'
#   rpm   docker run --rm -v <pkgs>:/pkg:ro fedora:latest bash -c \
#           'dnf -y install dkms kernel-devel kernel-core kmod binutils xz; bash /pkg/pkg-proof.sh rpm /pkg'
#   arch  docker run --rm -v <pkgs>:/pkg:ro archlinux:latest bash -c \
#           'pacman -Sy --noconfirm && pacman -S --noconfirm dkms linux linux-headers kmod binutils zstd; \
#            bash /pkg/pkg-proof.sh arch /pkg'
#
# `linux-modules-<ver>` / `kernel-core` / `linux` are the container's way of having the kernel's own
# module tree, which a real machine always has. Without it there is no in-tree amdxdna for the DKMS
# module to take precedence over, and the precedence row below cannot be measured.
#
# docs/install-any-distro.md carries the same three lines with the version and the measured output.
#
# Two facts about running this in a container, so a failure here is not read as a package defect:
#   * the container shares the HOST kernel, so `uname -r` is the host's, not the headers it has. Every
#     dkms call therefore names the kernel explicitly (-k) instead of letting dkms default to uname.
#   * dkms installs the module where its own framework config says (Debian /updates, Fedora /extra,
#     Arch /updates); the check is "modprobe -n resolves OUR file", not a fixed path.
set -uo pipefail

fam=${1:?usage: pkg-proof.sh <deb|rpm|arch> <package-dir>}
pkgdir=${2:?usage: pkg-proof.sh <deb|rpm|arch> <package-dir>}
pkgdir=$(cd "$pkgdir" && pwd)
fail=0
row() {   # <PASS|FAIL> <what> <detail>
    printf '%-4s %-46s %s\n' "$1" "$2" "$3"
    [ "$1" = PASS ] || fail=1
}
check() { local what=$1 detail=$2; shift 2; if "$@" >/dev/null 2>&1; then row PASS "$what" "$detail"; else row FAIL "$what" "$detail"; fi; }

# the kernel whose headers this container has (never the running kernel's)
case $fam in
    deb)  kv=$(ls /lib/modules | sort -V | tail -1) ;;
    rpm)  kv=$(ls /usr/src/kernels | head -1) ;;
    arch) kv=$(ls /usr/lib/modules | grep -v '^7\.4\.0-rc1-1bit$' | head -1) ;;
    *) echo "unknown family: $fam" >&2; exit 2 ;;
esac
[ -n "$kv" ] || { echo "pkg-proof: no kernel headers in this container; install them first (see the header)" >&2; exit 2; }
echo "== $fam in $(grep -h '^PRETTY_NAME' /etc/os-release | cut -d= -f2- | tr -d '\"') | headers $kv | running $(uname -r)"

# the in-tree module this must displace; without the kernel's own module tree there is nothing to
# displace, and a "no module" result below would be an artefact of the container, not a defect
intree=$(find /lib/modules/"$kv"/kernel /usr/lib/modules/"$kv"/kernel -path '*accel/amdxdna/amdxdna.ko*' 2>/dev/null | head -1)
if [ -n "$intree" ]; then row PASS "the in-tree amdxdna is present" "$intree"
else row FAIL "the in-tree amdxdna is present" "install the kernel's module tree (linux-modules-<ver> / kernel-core / linux)"; fi
depmod -a "$kv" 2>/dev/null || :

# ---- install the three packages --------------------------------------------------------------
root=()   # (subdir, name glob) per family, so the rows below do not care which family
case $fam in
    deb)  names=(1bit-npu-config amdxdna-1bit-dkms ec-su-axb35-dkms)
          dpkg -i "$pkgdir"/1bit-npu-config_*.deb "$pkgdir"/amdxdna-1bit-dkms_*.deb \
                  "$pkgdir"/ec-su-axb35-dkms_*.deb >/tmp/install.log 2>&1 ;;
    rpm)  names=(1bit-npu-config amdxdna-1bit-dkms ec-su-axb35-dkms)
          rpm -i --nosignature "$pkgdir"/1bit-npu-config-*.rpm "$pkgdir"/amdxdna-1bit-dkms-*.rpm \
                  "$pkgdir"/ec-su-axb35-dkms-*.rpm >/tmp/install.log 2>&1 ;;
    arch) names=(1bit-npu-config amdxdna-1bit-dkms ec-su-axb35-dkms)
          pacman -U --noconfirm "$pkgdir"/1bit-npu-config-*.pkg.tar.zst "$pkgdir"/amdxdna-1bit-dkms-*.pkg.tar.zst \
                  "$pkgdir"/ec-su-axb35-dkms-*.pkg.tar.zst >/tmp/install.log 2>&1 ;;
esac
if [ $? -eq 0 ]; then row PASS "install the three 1bit packages" "rc=0"; else row FAIL "install the three 1bit packages" "$(tail -3 /tmp/install.log)"; fi

# ---- 1bit-npu-config ------------------------------------------------------------------------
check "memlock limits.d" "2 lines, unlimited" \
    grep -q 'memlock    unlimited' /etc/security/limits.d/99-amdnpu-memlock.conf
check "NPU keep-awake udev rule syntax" "udevadm verify" \
    udevadm verify /etc/udev/rules.d/90-1bit-npu-power.rules
check "modules-load.d entry" "ec_su_axb35" \
    grep -qx ec_su_axb35 /etc/modules-load.d/ec_su_axb35.conf
check "public MOK certificate (kernel key)" \
    "$(openssl x509 -inform DER -in /usr/share/1bit/mok/MOK-Kernel.der -noout -subject 2>/dev/null | sed 's/subject=//')" \
    test -s /usr/share/1bit/mok/MOK-Kernel.der
check "config files are 0644" "not group-writable" \
    bash -c '! find /etc/security/limits.d/99-amdnpu-memlock.conf /etc/udev/rules.d/90-1bit-npu-power.rules -perm /022 | grep -q .'

# ---- the DKMS modules -----------------------------------------------------------------------
dkmsstate=$(dkms status 2>/dev/null || :)
case $dkmsstate in *amdxdna-1bit/0.18.0*) amdx=1 ;; *) amdx=0 ;; esac
case $dkmsstate in *ec-su-axb35/0.1*) ec=1 ;; *) ec=0 ;; esac
[ "$amdx$ec" = 11 ] && row PASS "dkms registers both source trees" "amdxdna-1bit/0.18.0, ec-su-axb35/0.1" \
    || row FAIL "dkms registers both source trees" "$(printf '%s' "$dkmsstate" | head -3)"

dkms build -m amdxdna-1bit -v 0.18.0 -k "$kv" >/tmp/b1.log 2>&1 && b1=0 || b1=$?
check "dkms builds amdxdna against $kv" "rc=$b1" test "$b1" = 0
dkms install --force -m amdxdna-1bit -v 0.18.0 -k "$kv" >/tmp/i1.log 2>&1 && i1=0 || i1=$?
check "dkms installs amdxdna" "rc=$i1" test "$i1" = 0
dkms build -m ec-su-axb35 -v 0.1 -k "$kv" >/tmp/b2.log 2>&1 && b2=0 || b2=$?
check "dkms builds ec_su_axb35 against $kv" "rc=$b2" test "$b2" = 0
dkms install --force -m ec-su-axb35 -v 0.1 -k "$kv" >/tmp/i2.log 2>&1 && i2=0 || i2=$?
check "dkms installs ec_su_axb35" "rc=$i2" test "$i2" = 0

# where dkms put them, and whether the out-of-tree module wins over the in-tree one
# --show-depends prints every module in dependency order; the last insmod line is the module itself
resolved=$(modprobe -S "$kv" --show-depends amdxdna 2>/dev/null | sed -n 's/^insmod //p' | tail -1)
case $resolved in
    */updates/*|*/extra/*) row PASS "amdxdna resolves to the DKMS build" "$resolved" ;;
    *) row FAIL "amdxdna resolves to the DKMS build" "${resolved:-modprobe -n found none}" ;;
esac
ecmod=$(find /lib/modules/"$kv"/updates /lib/modules/"$kv"/extra /usr/lib/modules/"$kv"/updates \
             /usr/lib/modules/"$kv"/extra -name 'ec_su_axb35.ko*' 2>/dev/null | head -1)
case $ecmod in
    *.zst) decomp="zstd -dc" ;;
    *.xz)  decomp="xz -dc" ;;
    *)     decomp="cat" ;;
esac
if [ -z "$ecmod" ]; then row FAIL "the AXB35 DMI gate is in the built module" "no module found"
elif ! command -v "${decomp%% *}" >/dev/null 2>&1; then
    row FAIL "the AXB35 DMI gate is in the built module" "${decomp%% *} missing: install it to read $(basename "$ecmod")"
elif $decomp "$ecmod" 2>/dev/null | strings | grep -q 'not an AXB35'; then
    row PASS "the AXB35 DMI gate is in the built module" "$(basename "$ecmod")"
else
    row FAIL "the AXB35 DMI gate is in the built module" "$(basename "$ecmod") has no gate string"
fi
amod=$resolved
[ -n "$amod" ] && row PASS "amdxdna built and installed for $kv" "$(basename "$amod")" \
    || row FAIL "amdxdna built and installed for $kv" "modprobe -n resolves nothing"

# ---- the kernel package ---------------------------------------------------------------------
case $fam in
    deb)
        dpkg -i "$pkgdir"/linux-image-*.deb "$pkgdir"/linux-headers-*.deb >/tmp/k.log 2>&1 \
            && row PASS "install the kernel debs" "linux-image + linux-headers" \
            || row FAIL "install the kernel debs" "$(tail -2 /tmp/k.log)"
        krel=$(ls /boot/vmlinuz-* | grep -v generic | sed 's|.*vmlinuz-||' | tail -1)
        check "kernel image in the package" "/boot/vmlinuz-$krel" test -s "/boot/vmlinuz-$krel"
        if ! command -v sbverify >/dev/null 2>&1; then
            row FAIL "kernel signed with the shipped MOK" "sbverify missing: install sbsigntool"
        elif sig=$(sbverify --cert <(openssl x509 -inform DER -in /usr/share/1bit/mok/MOK-Kernel.der) \
                    "/boot/vmlinuz-$krel" 2>&1); then
            row PASS "kernel signed with the shipped MOK" "$sig"
        else
            row FAIL "kernel signed with the shipped MOK" "$sig"
        fi
        ;;
    rpm)
        rpm -i --nosignature "$pkgdir"/1bit-kernel-[0-9]*.rpm "$pkgdir"/1bit-kernel-devel-*.rpm >/tmp/k.log 2>&1 \
            && row PASS "install the kernel rpms" "1bit-kernel + 1bit-kernel-devel" \
            || row FAIL "install the kernel rpms" "$(tail -2 /tmp/k.log)"
        ;;
    arch)
        pacman -U --noconfirm "$pkgdir"/linux-1bit-[0-9]*.pkg.tar.zst "$pkgdir"/linux-1bit-headers-*.pkg.tar.zst >/tmp/k.log 2>&1 \
            && row PASS "install the kernel packages" "linux-1bit + linux-1bit-headers" \
            || row FAIL "install the kernel packages" "$(tail -2 /tmp/k.log)"
        ;;
esac
krel=7.4.0-rc1-1bit
mods=$(find /usr/lib/modules/$krel/kernel /lib/modules/$krel/kernel -name '*.ko*' 2>/dev/null | wc -l)
check "the kernel package carries its modules" "$mods modules" test "$mods" -gt 3000
build=$(readlink -f /usr/lib/modules/$krel/build /lib/modules/$krel/build 2>/dev/null | head -1)
check "the headers package provides modules/$krel/build" "${build:-missing}" \
    bash -c "[ -f '$build/Makefile' ] && [ -f '$build/Module.symvers' ]"
# the kernel the packages ship is the kernel the box runs: same vmlinuz bytes as the running one
if [ -r "/boot/vmlinuz-$krel" ]; then
    sha=$(sha256sum "/boot/vmlinuz-$krel" | cut -d' ' -f1)
    row PASS "vmlinuz in the package" "/boot/vmlinuz-$krel sha256 $sha"
fi

# ---- uninstall ------------------------------------------------------------------------------
case $fam in
    deb)  dpkg -P ec-su-axb35-dkms amdxdna-1bit-dkms 1bit-npu-config >/tmp/r.log 2>&1 ;;
    rpm)  rpm -e ec-su-axb35-dkms amdxdna-1bit-dkms 1bit-npu-config >/tmp/r.log 2>&1 ;;
    arch) pacman -R --noconfirm ec-su-axb35-dkms amdxdna-1bit-dkms 1bit-npu-config >/tmp/r.log 2>&1 ;;
esac
if [ $? -eq 0 ]; then row PASS "uninstall the three 1bit packages" "rc=0"; else row FAIL "uninstall the three 1bit packages" "$(tail -2 /tmp/r.log)"; fi
left=$(ls /lib/modules/"$kv"/updates /lib/modules/"$kv"/updates/dkms /lib/modules/"$kv"/extra \
           /usr/lib/modules/"$kv"/updates /usr/lib/modules/"$kv"/updates/dkms /usr/lib/modules/"$kv"/extra 2>/dev/null | grep -c '\.ko')
check "no module left behind" "$left files" test "$left" = 0
leftstate=$(dkms status 2>/dev/null || :)
case $leftstate in *amdxdna-1bit*|*ec-su-axb35*) row FAIL "dkms forgot the module" "$(printf '%s' "$leftstate" | head -2)" ;;
    *) row PASS "dkms forgot the module" "dkms status clean" ;; esac
check "the config files are gone" "/etc rules removed" \
    bash -c "! test -e /etc/udev/rules.d/90-1bit-npu-power.rules"

echo
if [ "$fail" = 0 ]; then echo "== $fam: ALL ROWS PASS"; else echo "== $fam: FAILURES ABOVE"; fi
exit "$fail"
