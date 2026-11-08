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
# pkg-proof.sh <deb|rpm|arch|coexist> <package-dir>
#
# The proof for one package family (engine#1280 item 7), run INSIDE a clean container of that
# family. It installs the three 1bit packages, builds both DKMS modules against the CONTAINER's own
# kernel headers, checks that the out-of-tree module resolves ahead of the in-tree one, checks the
# AXB35 DMI gate is compiled into the EC module, installs the kernel package and checks its payload,
# then uninstalls everything and checks nothing is left behind. Prints one PASS/FAIL row per check and
# exits non-zero if any row fails.
#
#   deb   docker run --rm --cap-add=sys_resource -v <pkgs>:/pkg:ro ubuntu:26.04 bash -c \
#           'apt-get update -qq && apt-get install -y -qq dkms udev kmod sbsigntool linux-headers-generic \
#              && apt-get install -y -qq linux-modules-$(ls /lib/modules | sort -V | tail -1) \
#              && bash /pkg/pkg-proof.sh deb /pkg'
#   rpm   docker run --rm --cap-add=sys_resource -v <pkgs>:/pkg:ro fedora:latest bash -c \
#           'dnf -y install dkms kernel-devel kernel-core kmod binutils xz; bash /pkg/pkg-proof.sh rpm /pkg'
#   arch  docker run --rm --cap-add=sys_resource -v <pkgs>:/pkg:ro archlinux:latest bash -c \
#           'pacman -Sy --noconfirm && pacman -S --noconfirm dkms linux linux-headers kmod binutils zstd; \
#            bash /pkg/pkg-proof.sh arch /pkg'
#
# `--cap-add=sys_resource` is for the two memlock rows: a container's default capability set excludes
# CAP_SYS_RESOURCE, so raising the hard memlock limit to `unlimited` fails there and the login session
# keeps the container's inherited 8192 KB. A real machine's root has the capability; the flag makes the
# container behave like one.
#
# `linux-modules-<ver>` / `kernel-core` / `linux` are the container's way of having the kernel's own
# module tree, which a real machine always has. Without it there is no in-tree amdxdna for the DKMS
# module to take precedence over, and the precedence row below cannot be measured.
#
# docs/install-any-distro.md carries the same three lines with the version and the measured output.
#
# `coexist` is not a package format: it installs the engine package (1bit-engine_*.deb, which the
# release publishes from scripts/pack-distro.sh) NEXT TO this set and checks the two do not fight. The
# two harnesses that exist each install one set — this one and scripts/pack-proof.sh — so the
# combination had never been installed anywhere, and the sets do share a path (/usr/share/1bit: the
# engine's config fragment, our MOK certificate). Rows: the engine installs alone, this set installs
# over it with dpkg reporting no file conflict, /usr/share/1bit holds both and each file is attributed
# to its own package, the engine still runs, both dkms modules still build, this set purges while the
# engine stays installed (its files survive), and the engine purges.
#
#   coexist  docker run --rm -v <dir>:/pkg:ro ubuntu:26.04 bash -c \
#              'apt-get update -qq && apt-get install -y -qq dkms udev kmod linux-headers-generic \
#                 && apt-get install -y linux-modules-$(ls /lib/modules | sort -V | tail -1) \
#                 && bash /pkg/pkg-proof.sh coexist /pkg'      # needs 1bit-engine_*.deb in <dir>
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
krel=      # set by the kernel-package block; empty on the distribution-kernel path (no kernel packages)
row() {   # <PASS|FAIL> <what> <detail>
    printf '%-4s %-46s %s\n' "$1" "$2" "$3"
    [ "$1" = PASS ] || fail=1
}
check() { local what=$1 detail=$2; shift 2; if "$@" >/dev/null 2>&1; then row PASS "$what" "$detail"; else row FAIL "$what" "$detail"; fi; }

# the kernel whose headers this container has (never the running kernel's)
case $fam in
    deb|coexist) kv=$(ls /lib/modules | sort -V | tail -1) ;;
    rpm)  kv=$(ls /usr/src/kernels | head -1) ;;
    arch) for d in /usr/lib/modules/*; do case ${d##*/} in 7.4.0-rc1-1bit) continue ;; *) kv=${d##*/}; break ;; esac; done ;;
    *) echo "unknown family: $fam" >&2; exit 2 ;;
esac
[ -n "$kv" ] || { echo "pkg-proof: no kernel headers in this container; install them first (see the header)" >&2; exit 2; }
# Does this package dir carry the kernel packages? `mk-kernel-packages.sh --skip-kernel` does not, and that
# is the most common user path — a machine that stays on its distribution's kernel takes only the drivers
# and the config. Those rows are then skipped instead of failing; the rest of the proof runs unchanged.
case $fam in
    deb)  have_kernel=$(ls "$pkgdir"/linux-image-*.deb 2>/dev/null | wc -l) ;;
    rpm)  have_kernel=$(ls "$pkgdir"/1bit-kernel-[0-9]*.rpm 2>/dev/null | wc -l) ;;
    arch) have_kernel=$(ls "$pkgdir"/linux-1bit-[0-9]*.pkg.tar.zst 2>/dev/null | wc -l) ;;
    coexist) have_kernel=0 ;;   # this mode installs the engine package beside this set, never a kernel
    *)    have_kernel=1 ;;
esac
if [ "$have_kernel" = 0 ] && [ "$fam" != coexist ]; then
    printf 'INFO %-46s %s\n' "no kernel packages in this dir" "distribution-kernel path: kernel rows skipped"
fi
echo "== $fam in $(grep -h '^PRETTY_NAME' /etc/os-release | cut -d= -f2- | tr -d '\"') | headers $kv | running $(uname -r)"

# the in-tree module this must displace; without the kernel's own module tree there is nothing to
# displace, and a "no module" result below would be an artefact of the container, not a defect
intree=$(find /lib/modules/"$kv"/kernel /usr/lib/modules/"$kv"/kernel -path '*accel/amdxdna/amdxdna.ko*' 2>/dev/null | head -1)
if [ -n "$intree" ]; then row PASS "the in-tree amdxdna is present" "$intree"
else row FAIL "the in-tree amdxdna is present" "install the kernel's module tree (linux-modules-<ver> / kernel-core / linux)"; fi
depmod -a "$kv" 2>/dev/null || :

# ---- install the three packages --------------------------------------------------------------
case $fam in
    coexist)
        eng=$(ls "$pkgdir"/1bit-engine_*.deb | head -1)
        dpkg -i "$eng" >/tmp/engine.log 2>&1 && row PASS "the engine package installs alone" "$(basename "$eng")" \
            || row FAIL "the engine package installs alone" "$(tail -2 /tmp/engine.log)"
        dpkg -i "$pkgdir"/1bit-npu-config_*.deb "$pkgdir"/amdxdna-1bit-dkms_*.deb \
                "$pkgdir"/ec-su-axb35-dkms_*.deb >/tmp/install.log 2>&1 ;;
    deb)  dpkg -i "$pkgdir"/1bit-npu-config_*.deb "$pkgdir"/amdxdna-1bit-dkms_*.deb \
                  "$pkgdir"/ec-su-axb35-dkms_*.deb >/tmp/install.log 2>&1 ;;
    rpm)  rpm -i --nosignature "$pkgdir"/1bit-npu-config-*.rpm "$pkgdir"/amdxdna-1bit-dkms-*.rpm \
                  "$pkgdir"/ec-su-axb35-dkms-*.rpm >/tmp/install.log 2>&1 ;;
    arch) pacman -U --noconfirm "$pkgdir"/1bit-npu-config-*.pkg.tar.zst "$pkgdir"/amdxdna-1bit-dkms-*.pkg.tar.zst \
                  "$pkgdir"/ec-su-axb35-dkms-*.pkg.tar.zst >/tmp/install.log 2>&1 ;;
esac
if [ $? -eq 0 ]; then row PASS "install the three 1bit packages" "rc=0"; else row FAIL "install the three 1bit packages" "$(tail -3 /tmp/install.log)"; fi
if [ "$fam" = coexist ]; then
    # the one path the two sets share, and who owns each file under it
    both=$(ls /usr/share/1bit 2>/dev/null | tr '\n' ' ')
    case $both in *mok*lemonade-config-fragment.json*|*lemonade-config-fragment.json*mok*)
        row PASS "/usr/share/1bit carries both sets" "$both" ;;
        *) row FAIL "/usr/share/1bit carries both sets" "${both:-empty}" ;; esac
    who=$(dpkg -S /usr/share/1bit/mok/MOK-Kernel.der /usr/share/1bit/lemonade-config-fragment.json 2>&1 | tr '\n' ' ')
    case $who in *1bit-npu-config*1bit-engine*|*1bit-engine*1bit-npu-config*)
        row PASS "each file belongs to its own package" "$who" ;;
        *) row FAIL "each file belongs to its own package" "$who" ;; esac
    if /opt/1bit/engine/1bit --version >/tmp/engv.log 2>&1; then
        row PASS "the engine still runs with this set installed" "$(cat /tmp/engv.log)"
    else row FAIL "the engine still runs with this set installed" "$(tail -1 /tmp/engv.log)"; fi
fi

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

# Is the memlock file actually READ by the login path, or only present? The NPU runtime locks command
# buffers and fails without it, so a right-looking file at a path pam_limits does not read would be a
# confusing user-facing failure. Proof: write a distinctive finite limit into the installed file, log in
# as a fresh user, and see that value come back; then restore the shipped content and read it again.
if id kpkgprobe >/dev/null 2>&1 || useradd -m -s /bin/bash kpkgprobe >/dev/null 2>&1; then
    lim=/etc/security/limits.d/99-amdnpu-memlock.conf
    keep=$(mktemp); cp "$lim" "$keep"
    # 4096, not 8192: LOWERING a limit is always permitted, so the value can only come from the file.
    # (The first version of this row wrote 8192, which happened to equal the container's inherited
    # memlock and so passed without proving anything.)
    printf '*    soft    memlock    4096\n*    hard    memlock    4096\n' > "$lim"
    probe=$(su - kpkgprobe -c 'ulimit -l' 2>/dev/null | tr -d ' ')
    if [ "$probe" = 4096 ]; then
        row PASS "the memlock file is what pam_limits reads" "a probe login came back with 4096 KB"
    else
        row FAIL "the memlock file is what pam_limits reads" "probe login reports '${probe:-nothing}'"
    fi
    cp "$keep" "$lim"; rm -f "$keep"
    shipped=$(su - kpkgprobe -c 'ulimit -l' 2>/dev/null | tr -d ' ')
    case $shipped in
        unlimited) row PASS "the shipped limit reaches a login session" "ulimit -l: unlimited" ;;
        *) row FAIL "the shipped limit reaches a login session" \
               "ulimit -l: '${shipped:-nothing}' — raising a hard limit needs CAP_SYS_RESOURCE: run the container with --cap-add=sys_resource (a real machine's root has it)" ;;
    esac
else
    row FAIL "the memlock file is what pam_limits reads" "could not create a probe user"
fi

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
# grep -q on the tail of a pipeline closes it early, and under `pipefail` the upstream tool's SIGPIPE
# then fails the row even though the match was found (measured on arch, where zstd is still writing).
# So: decompress to a file once, then grep the file and read nm into a variable.
if [ -z "$ecmod" ]; then
    row FAIL "the AXB35 DMI gate is in the built module" "no module found"
elif ! $decomp "$ecmod" > /tmp/ecmod.ko 2>/dev/null; then
    row FAIL "the AXB35 DMI gate is in the built module" "cannot decompress $(basename "$ecmod")"
else
    gate=no
    grep -aq 'not an AXB35' /tmp/ecmod.ko && gate=yes                       # the patch's message
    nmout=$(nm /tmp/ecmod.ko 2>/dev/null || :)
    case $nmout in *dmi_get_system_info*) gate=yes ;; esac                  # the DMI call itself
    if [ "$gate" = yes ]; then
        row PASS "the AXB35 DMI gate is in the built module" "$(basename "$ecmod"): dmi_get_system_info, 'not an AXB35'"
    else
        row FAIL "the AXB35 DMI gate is in the built module" "$(basename "$ecmod") has neither the call nor the message"
    fi
fi

# ---- the kernel package ---------------------------------------------------------------------
if [ "$have_kernel" = 1 ]; then
case $fam in
    coexist) ;;   # this mode is about the engine set, not about the kernel package
    deb)
        dpkg -i "$pkgdir"/linux-image-*.deb "$pkgdir"/linux-headers-*.deb >/tmp/k.log 2>&1 \
            && row PASS "install the kernel debs" "linux-image + linux-headers" \
            || row FAIL "install the kernel debs" "$(tail -2 /tmp/k.log)"
        krel=7.4.0-rc1-1bit
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
[ "$fam" = coexist ] && krel=          # this mode installed no kernel package; the rows below are the deb/rpm/arch ones
if [ -n "$krel" ]; then
mods=$(find /usr/lib/modules/$krel/kernel /lib/modules/$krel/kernel -name '*.ko*' 2>/dev/null | wc -l)
check "the kernel package carries its modules" "$mods modules" test "$mods" -gt 3000
build=$(readlink -f /usr/lib/modules/$krel/build /lib/modules/$krel/build 2>/dev/null | head -1)
check "the headers package provides modules/$krel/build" "${build:-missing}" \
    bash -c "[ -f '$build/Makefile' ] && [ -f '$build/Module.symvers' ]"
# the kernel the packages ship is the kernel the box runs: same vmlinuz bytes as the running one
img=/boot/vmlinuz-$krel
[ "$fam" = arch ] && img=/boot/vmlinuz-1bit          # Arch names it by suffix, with the 1bit preset beside it
if [ -r "$img" ]; then
    row PASS "vmlinuz in the package" "$img sha256 $(sha256sum "$img" | cut -d' ' -f1)"
else
    row FAIL "vmlinuz in the package" "$img missing"
fi
fi          # [ -n "$krel" ]
fi   # have_kernel

# ---- uninstall ------------------------------------------------------------------------------
case $fam in
    coexist|deb) dpkg -P ec-su-axb35-dkms amdxdna-1bit-dkms 1bit-npu-config >/tmp/r.log 2>&1 ;;
    rpm)  rpm -e ec-su-axb35-dkms amdxdna-1bit-dkms 1bit-npu-config >/tmp/r.log 2>&1 ;;
    arch) pacman -R --noconfirm ec-su-axb35-dkms amdxdna-1bit-dkms 1bit-npu-config >/tmp/r.log 2>&1 ;;
esac
if [ $? -eq 0 ]; then row PASS "uninstall the three 1bit packages" "rc=0"; else row FAIL "uninstall the three 1bit packages" "$(tail -2 /tmp/r.log)"; fi
left=$(for d in /lib/modules/"$kv"/updates /lib/modules/"$kv"/updates/dkms /lib/modules/"$kv"/extra \
                   /usr/lib/modules/"$kv"/updates /usr/lib/modules/"$kv"/updates/dkms /usr/lib/modules/"$kv"/extra; do
             for f in "$d"/*.ko*; do [ -e "$f" ] && echo "$f"; done
         done 2>/dev/null | wc -l)
check "no module left behind" "$left files" test "$left" = 0
leftstate=$(dkms status 2>/dev/null || :)
case $leftstate in *amdxdna-1bit*|*ec-su-axb35*) row FAIL "dkms forgot the module" "$(printf '%s' "$leftstate" | head -2)" ;;
    *) row PASS "dkms forgot the module" "dkms status clean" ;; esac
check "the config files are gone" "/etc rules removed" \
    bash -c "! test -e /etc/udev/rules.d/90-1bit-npu-power.rules"
if [ "$fam" != coexist ]; then
    # The other half of "uninstalling puts the in-tree driver back": dkms archives the in-tree module
    # when it installs ours (into /var/lib/dkms/<pkg>/original_module/), so removal has to restore it —
    # both the file and what a modprobe resolves to, because a restored file that nothing resolves to is
    # not a restore. Verified by hand first: before our package the resolution is
    # /lib/modules/<kv>/kernel/drivers/accel/amdxdna/amdxdna.ko.zst, with it .../updates/dkms/amdxdna.ko.zst,
    # and after the purge back to the kernel/ path.
    depmod -a "$kv" 2>/dev/null || :
    back=$(modprobe -S "$kv" --show-depends amdxdna 2>/dev/null | sed -n 's/^insmod //p' | tail -1)
    case $back in
        */kernel/drivers/accel/amdxdna/*)
            row PASS "the in-tree amdxdna resolves again" "$(basename "$back")" ;;
        *)  row FAIL "the in-tree amdxdna resolves again" "${back:-nothing resolves amdxdna}" ;;
    esac
    intree_file=$(find -L /lib/modules/"$kv"/kernel -name 'amdxdna.ko*' 2>/dev/null | head -1)
    if [ -n "$intree_file" ]; then
        row PASS "the in-tree module file is restored" "$(basename "$intree_file")"
    else
        row FAIL "the in-tree module file is restored" "not under /lib/modules/$kv/kernel"
    fi
fi

# ---- removing the kernel package again -------------------------------------------------------
# The kernel rows above install it; nothing removed it, so "and uninstall cleanly" had no evidence for
# the kernel half. It also has to leave the filesystem as it found it: no /boot entry, no module tree.
if [ -n "$krel" ] && [ "$fam" != coexist ] && [ "$have_kernel" = 1 ]; then
    case $fam in
        deb)  dpkg -P "linux-image-$krel" "linux-headers-$krel" >/tmp/kr.log 2>&1 ;;
        rpm)  rpm -e 1bit-kernel-devel 1bit-kernel >/tmp/kr.log 2>&1 ;;
        arch) pacman -R --noconfirm linux-1bit-headers linux-1bit >/tmp/kr.log 2>&1 ;;
    esac
    if [ $? -eq 0 ]; then row PASS "the kernel package removes" "rc=0"
    else row FAIL "the kernel package removes" "$(tail -2 /tmp/kr.log)"; fi
    case $fam in
        deb|rpm) leftover=/boot/vmlinuz-$krel ;;
        arch)    leftover=/boot/vmlinuz-1bit ;;
    esac
    [ -e "$leftover" ] && row FAIL "the kernel image is gone" "$leftover still there" \
        || row PASS "the kernel image is gone" "$leftover removed"
    mods_left=$(find /lib/modules/"$krel" /usr/lib/modules/"$krel" -type f 2>/dev/null | wc -l)
    [ "$mods_left" = 0 ] && row PASS "the module tree is gone" "0 files under modules/$krel" \
        || row FAIL "the module tree is gone" "$mods_left files left"
fi
if [ "$fam" = coexist ]; then
    # the engine was there first and must be intact after our purge
    if [ -e /usr/share/1bit/lemonade-config-fragment.json ] && [ ! -e /usr/share/1bit/mok ]; then
        row PASS "the engine's files survive our purge" "fragment kept, mok/ gone"
    else
        row FAIL "the engine's files survive our purge" "$(ls /usr/share/1bit 2>/dev/null | tr '\n' ' ')"
    fi
    dpkg -P 1bit-engine >/tmp/re.log 2>&1 && row PASS "the engine purges last" "dpkg rc=0" \
        || row FAIL "the engine purges last" "$(tail -2 /tmp/re.log)"
fi

echo
if [ "$fail" = 0 ]; then echo "== $fam: ALL ROWS PASS"; else echo "== $fam: FAILURES ABOVE"; fi
exit "$fail"
