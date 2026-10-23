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
# mkimage.sh <out> [engine build dir]
#
# 1bit OS, the few-minutes version (os/README.md): a bootable image assembled from what this
# machine already has, no distro and no package builds. It boots from USB into RAM and never
# touches the internal disk.
#
#   <out>/1bit-os.efi   one EFI file: the kernel, its command line and the initramfs
#   <out>/1bit-os.img   a USB image: an EFI partition with 1bit-os.efi as the default boot
#                       entry, and an ext4 data partition (label 1BIT-DATA) for models
#
# 1bit OS is a Lemonade appliance: it boots Lemonade's server (lemond), with Lemonade's defaults,
# and the engine is one of Lemonade's backends (the onebit recipe). The initramfs holds BusyBox,
# the kernel modules for the GPU (amdgpu, which also provides /dev/kfd) and the NPU (amdxdna),
# Ethernet, Wi-Fi and USB storage, the firmware those load (the NPU runtime is compiled into 1bit:
# no XRT), Mesa's Radeon
# Vulkan driver (for Lemonade's Vulkan backends), Lemonade (lemond, the lemonade CLI and the web
# app, built from the engine's pin third_party/lemonade), and the engine: 1bit and its HRX
# llama-server, with TheRock's HSA runtime that HRX loads (ONEBIT_HRX_LIBHSA from the build's
# CMake cache, kept at its path, which 1bit has compiled in), each with exactly the shared
# libraries it links. HRX compiles its Loom kernels on first use and caches them
# (GGML_HRX_JIT_CACHE_DIR); init keeps that cache on the stick's data partition, so only the first
# boot pays the compile.
#
# Lemonade's own backends (llama.cpp, whisper.cpp, stable-diffusion.cpp, Kokoro, ...) are
# downloaded by Lemonade on first use. For a stick that works with no network, this script
# installs them ahead of time with Lemonade's own installer (lemond's /api/v1/install, run on this
# machine) into Lemonade's cache directory on the data partition, at the paths and versions
# Lemonade expects: LEMONADE_BACKENDS lists them as recipe:backend (empty: none). Their system
# libraries go into the initramfs. ROCm backends are left to Lemonade's download on the stick:
# their runtime is per GPU architecture and keeps absolute paths (os/README.md).
#
# Licences (os/LICENSES.md): a backend ships only when os/licenses/<recipe>/SOURCE says
# "Ship: yes"; that directory goes onto the stick as /data/lemonade/licenses/<recipe>. The
# initramfs carries /usr/share/licenses/1bit-os (LICENSES.md, the engine's LICENSE and NOTICE, the
# closed binaries' LICENSE-BINARY and NOTICE-BINARY,
# Lemonade's LICENSE, packages.txt) and the Ubuntu copyright file of every package it takes files
# from, in /usr/share/doc/<package>/copyright, with /usr/share/common-licenses.
#
#   LEMONADE=<dir>        a build-lemonade.sh prefix with bin/lemond (default <engine build>/lemonade,
#                         built with the web app when missing)
#   LEMONADE_CONFIG=<f>   Lemonade's config.json for the stick (default: models from /data/models,
#                         downloads on /data; everything else Lemonade's defaults)
#   LEMONADE_ENV=<f>      Lemonade's environment file for the stick (LEMONADE_API_KEY, ...)
#
# Test images only (vmtest.sh on a machine without TheRock): LLAMA_SERVER=<a CPU llama-server>
# stands in for the HRX build.
#
# Needs no root: the EFI stub and mtools come from their Debian packages, unpacked locally.
set -euo pipefail
# modprobe and mkfs live in /usr/sbin, which cron and the weekly wrapper leave off PATH
export PATH="$PATH:/usr/sbin:/sbin"
out=${1:?usage: mkimage.sh <out> [engine build dir]}
build=${2:-$HOME/.cache/1bit-os/src/build}
build=$(cd "$build" && pwd)   # absolute: binaries keep their build paths inside the image
kver=${KVER:-$(uname -r)}
# the kernel image: /boot/vmlinuz-<kver>, or VMLINUZ= (e.g. unpacked from its .deb where /boot is root-only);
# the engine builds no kernel: the Strix Halo kernel packages come from our separate kernel build
vmlinuz=${VMLINUZ:-/boot/vmlinuz-$kver}
here=$(cd "$(dirname "$0")" && pwd)
tools=$HOME/.cache/1bit-os/tools/x
# Dropbear, mtools and the EFI stub come from their Debian packages, unpacked here on first use. A
# cleaned ~/.cache used to stop the weekly release at "SSH (Dropbear)" (2026-10-01 rehearsal).
tool_debs="dropbear-bin libtomcrypt1 libtommath1 mtools systemd-boot-efi"
if [ ! -x "$tools/usr/sbin/dropbear" ] || [ ! -x "$tools/usr/bin/mtools" ] ||
   [ ! -e "$tools/usr/lib/systemd/boot/efi/linuxx64.efi.stub" ]; then
    echo "== unpacking $tool_debs into $tools"
    mkdir -p "$tools"
    (cd "$(dirname "$tools")" && apt-get download $tool_debs)
    for d in "$(dirname "$tools")"/*.deb; do dpkg -x "$d" "$tools"; done
fi
mkdir -p "$out"; out=$(cd "$out" && pwd)
root=$out/root
rm -rf "$root"; mkdir -p "$root"/{bin,sbin,etc,proc,sys,dev,tmp,run,data,lib/firmware,root,usr/share/vulkan/icd.d}

# --- a file and every shared library it links, at the same paths -------------------------
copy_libs() {   # the shared libraries these files link, at their paths (not the files themselves)
    local f
    for f in "$@"; do
        ldd "$f" 2>/dev/null | grep -o '/[^ ]*' | while read -r lib; do
            [ -e "$root$lib" ] || { mkdir -p "$root$(dirname "$lib")"; cp -L "$lib" "$root$lib"; }
        done
    done
}
copy_elf() {
    local f
    for f in "$@"; do
        [ -e "$f" ] || { echo "missing: $f"; exit 1; }
        mkdir -p "$root$(dirname "$f")"; cp -L "$f" "$root$f"
        copy_libs "$f"
    done
}

echo "== BusyBox"
cp /usr/bin/busybox "$root/bin/busybox"
for a in $("$root/bin/busybox" --list); do [ -e "$root/bin/$a" ] || ln -s busybox "$root/bin/$a"; done

echo "== kernel modules ($kver)"
# plus virtio, so 1bit OS also runs (and is tested) in a VM; together a few hundred KB
# USB4 v2 / Thunderbolt on Strix Halo: the USB4 host router (1022:158d) is driven by thunderbolt.ko and
# its Type-C side by typec + ucsi_acpi (alt modes, display); without them the USB4 ports and any dock
# behind them do nothing, and with them a stick on a USB4 port tunnels like any other.
# nvme and ahci are for the install target, not for the live stick: 1bit-install writes an internal
# disk, and without these modules the live system cannot even see one (os/mini/install.sh). nls_cp437 and
# nls_ascii are for mounting its ESP: the image mounts no vfat until the installer does, and this kernel's
# CONFIG_NLS_DEFAULT is ascii ("codepage cp437 not found" / "IO charset ascii not found" in the VM).
mods="nvme ahci amd_pmf amdgpu amdxdna r8169 realtek mt7925e xhci_pci usb_storage uas sd_mod vfat ext4 nls_iso8859_1 nls_cp437 nls_ascii virtio_net virtio_blk thunderbolt thunderbolt_net typec ucsi_acpi typec_displayport"
for m in $mods; do
    modprobe -S "$kver" --show-depends "$m" 2>/dev/null | awk '$1 == "insmod" {print $2}'
done | sort -u | while read -r ko; do mkdir -p "$root$(dirname "$ko")"; cp "$ko" "$root$ko"; done
cp /lib/modules/"$kver"/modules.{order,builtin,builtin.modinfo} "$root/lib/modules/$kver/" 2>/dev/null || true
# The NPU needs the PRIVATE XDNA driver build, not the stock module in /lib/modules: with the stock one
# the engine's NPU route resets the machine. Same vermagic, so it is a straight drop-in.
xdna=${AMDXDNA_KO:-$HOME/wt/perfopt-7.3-obj/drivers/accel/amdxdna/amdxdna.ko}
if [ -f "$xdna" ]; then
    dst=$(find "$root/lib/modules" -name 'amdxdna.ko*' | head -1)
    if [ -n "$dst" ]; then
        rm -f "$dst"; mkdir -p "$(dirname "$dst")"; cp "$xdna" "$dst"
        echo "   amdxdna: private build installed over the stock module ($(stat -c %s "$dst") bytes)"
    fi
else
    echo "   amdxdna: private build not found at $xdna - the stock module ships"
fi
depmod -b "$root" "$kver"

echo "== firmware (only what these chips load)"
fw() {   # a pattern that matches nothing on this machine is skipped
    (cd /lib/firmware && for p in "$@"; do ls -d $p 2>/dev/null || true; done) | while read -r f; do
        mkdir -p "$root/lib/firmware/$(dirname "$f")"; cp -aL "/lib/firmware/$f" "$root/lib/firmware/$f"; done
    echo "   $(find "$root/lib/firmware" -type f | wc -l) files, $(du -sh "$root/lib/firmware" | cut -f1)"; }
# the GPU's IP blocks (ip_discovery): GC 11.5.1, SDMA 6.1.1, PSP/SMU (MP0/MP1) 14.0.1, VPE 6.1.1,
# display DCN 3.5.1; the NPU (1022:17f0 rev 11); the RTL8125 and MT7925 network chips
fw "amdgpu/gc_11_5_1_*" "amdgpu/sdma_6_1_1*" "amdgpu/psp_14_0_1*" "amdgpu/smu_14_0_1*" \
   "amdgpu/vpe_6_1_1*" "amdgpu/dcn_3_5_1*" "amdgpu/vcn_4_0_6*" "amdnpu/17f0_11" \
   "rtl_nic/rtl8125*" "mediatek/mt7925" "mediatek/WIFI_*7925*" "mediatek/BT_*7925*" "regulatory.db*"

echo "== the engine"
bin=$build/onebit; [ -x "$bin" ] || bin=$build/1bit
[ -x "$bin" ] || { echo "no engine binary in $build (build with -DONEBIT_NPU=ON -DONEBIT_HRX=ON)"; exit 1; }
copy_elf "$bin"
ln -s "$bin" "$root/bin/1bit"
# The sealed NPU kernel sets travel BESIDE the engine binary (docs/releases.md, "The NPU engine"):
# in a package that is <pkg>/npu/kernels next to <pkg>/1bit, and here the binary keeps its build
# path, so the sets go to the same place relative to it. Without them the image ships the NPU
# route and nothing it can load - the route then refuses rather than falling back.


echo "== Lemonade (lemond, the lemonade CLI, the web app)"
src=$(cd "$here/../.." && pwd)
lem=${LEMONADE:-$build/lemonade}
if [ ! -x "$lem/bin/lemond" ]; then
    echo "   building it from third_party/lemonade into $lem"
    LEMONADE_WEB_APP=1 "$src/scripts/build-lemonade.sh" "$lem" > "$out/lemonade-build.log" 2>&1 ||
        { echo "build-lemonade.sh failed: $out/lemonade-build.log"; exit 1; }
fi
lem=$(cd "$lem" && pwd)
[ -d "$lem/bin/resources/web-app" ] || echo "   warning: no web app in $lem (build it with LEMONADE_WEB_APP=1 scripts/build-lemonade.sh)"
# lemond reads resources/ beside itself
mkdir -p "$root/opt/lemonade"; cp -a "$lem/bin" "$root/opt/lemonade/bin"
# Lemonade's build links the web app's KaTeX fonts to the build machine's fonts-katex
# (/usr/share/fonts/truetype/katex), which the image does not have: every link out of the tree
# becomes a copy of its target (scripts/pack-elf.sh), the links inside it stay
# shellcheck source=scripts/pack-elf.sh
. "$src/scripts/pack-elf.sh"
pack_localize_links "$root/opt/lemonade/bin"
n=$(pack_outside_links "$root/opt/lemonade/bin" | wc -l)
[ "$n" = 0 ] || { echo "symlinks out of /opt/lemonade/bin in the image:"; pack_outside_links "$root/opt/lemonade/bin"; exit 1; }
copy_libs "$lem/bin/lemond" "$lem/bin/lemonade"
# HTTPS for Lemonade's downloads (models, backends) when the stick has a network
mkdir -p "$root/etc/ssl/certs"; cp -L /etc/ssl/certs/ca-certificates.crt "$root/etc/ssl/certs/"
echo "   lemond $(git -C "$src/third_party/lemonade" rev-parse --short=12 HEAD 2>/dev/null || echo '(pin unknown)'), $(du -sh "$root/opt/lemonade" | cut -f1)"

echo "== Vulkan (Mesa RADV, for Lemonade's Vulkan backends)"
radv=/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so
copy_elf /usr/lib/x86_64-linux-gnu/libvulkan.so.1 "$radv"
printf '{"file_format_version":"1.0.1","ICD":{"library_path":"%s","api_version":"1.4"}}\n' "$radv" \
    > "$root/usr/share/vulkan/icd.d/radeon_icd.json"

echo "== Lemonade's backends, installed ahead of time (offline first)"
# Lemonade's Linux x86_64 backends that run on Strix Halo without ROCm (the list lemond's
# /api/v1/system-info gives there); the onebit recipe is the engine itself, already in the image
# Not by default (os/LICENSES.md): thinksound (no licence), flm (closed NPU binaries linked with GPL FFTW)
backends=${LEMONADE_BACKENDS-llamacpp:vulkan llamacpp:cpu llamacpp-hrx:hrx whispercpp:vulkan whispercpp:cpu sd-cpp:vulkan sd-cpp:cpu kokoro:cpu moonshine:cpu onnxruntime:cpu acestep:vulkan trellis:vulkan openmoss:vulkan}
licenses=$here/../licenses
for b in $backends; do
    grep -q '^Ship: yes' "$licenses/${b%%:*}/SOURCE" 2>/dev/null ||
        { echo "   $b: not cleared to ship (os/licenses/${b%%:*}/SOURCE, os/LICENSES.md)"; exit 1; }
done
# installed once per Lemonade pin (its backend_versions.json) and reused by later builds
seed=$HOME/.cache/1bit-os/lemonade-backends/$(sha256sum "$lem/bin/resources/backend_versions.json" | cut -c1-16)
if [ -n "$backends" ]; then
    mkdir -p "$seed/cache" "$seed/config"
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    "$lem/bin/lemond" --port "$port" --host 127.0.0.1 --no-broadcast "$seed/cache" "$seed/config" > "$out/lemond-seed.log" 2>&1 &
    lemond_pid=$!
    for i in $(seq 60); do curl -sf "127.0.0.1:$port/api/v1/health" >/dev/null && break; sleep 1; done
    for b in $backends; do
        case $b in *:rocm*) echo "   $b: skipped (ROCm backends download on the stick, os/README.md)"; continue ;; esac
        # force: this machine's GPU may not be the one the backend is for
        r=$(curl -s --max-time 3600 "127.0.0.1:$port/api/v1/install" -H 'Content-Type: application/json' \
            -d "{\"recipe\":\"${b%%:*}\",\"backend\":\"${b#*:}\",\"force\":true}")
        case $r in *'"success"'*) echo "   $b: $(cat "$seed/cache/bin/${b%%:*}/${b#*:}/version.txt" 2>/dev/null)" ;;
                   *) kill $lemond_pid; echo "   $b: install failed: $r"; exit 1 ;; esac
    done
    kill $lemond_pid; wait $lemond_pid 2>/dev/null || true
    # the system libraries they link (everything else ships inside each backend's directory); the
    # seed may hold more backends than this list, from earlier builds, and only these ship
    for b in $backends; do echo "$seed/cache/bin/${b%%:*}/${b#*:}"; done | xargs -d '\n' -I{} find {} -type f \( -perm -u+x -o -name '*.so*' \) | while read -r f; do
        case $(head -c 4 "$f" | od -An -c | tr -d ' ') in '177ELF') ;; *) continue ;; esac
        d=$(dirname "$f")
        LD_LIBRARY_PATH="$d:$d/lib:$d/../lib" ldd "$f" 2>/dev/null | grep -o '=> /[^ ]*' | cut -c4- | grep -v "^$seed/" || true
    done | sort -u | while read -r lib; do
        [ -e "$root$lib" ] || { mkdir -p "$root$(dirname "$lib")"; cp -L "$lib" "$root$lib"; }
    done
fi

echo "== SSH (Dropbear, key logins only)"
lib=$tools/usr/lib/x86_64-linux-gnu
for f in usr/sbin/dropbear usr/bin/dropbearkey; do mkdir -p "$root/$(dirname "$f")"; cp "$tools/$f" "$root/$f"; done
LD_LIBRARY_PATH=$lib ldd "$tools/usr/sbin/dropbear" | grep -o '/[^ ]*' | while read -r l; do
    d=${l#"$tools"}; [ -e "$root$d" ] || { mkdir -p "$root$(dirname "$d")"; cp -L "$l" "$root$d"; }
done

echo "== init"
if [ -n "${BOOT_STUB:-}" ]; then
    cp "$here/stub-init.sh" "$root/init"; chmod +x "$root/init"
    echo "   init: STUB (diagnostic: proves userspace and dumps dmesg to the stick)"
else
    cp "$here/init.sh" "$root/init"; chmod +x "$root/init"
fi
# one command that proves the engine answers from this boot (console or SSH)
cp "$here/selftest.sh" "$root/bin/1bit-selftest"; chmod +x "$root/bin/1bit-selftest"
# the installer (os/README.md): from a live boot onto one internal disk. It lays the target out with
# the same sfdisk script this build uses for the stick, so an installed disk and a stick are the same
# layout - so the image carries sfdisk (busybox fdisk is MBR-only and cannot make the GPT the stick has)
cp "$here/install.sh" "$root/bin/1bit-install"; chmod +x "$root/bin/1bit-install"
cp /usr/sbin/sfdisk "$root/bin/sfdisk"; chmod +x "$root/bin/sfdisk"
copy_libs /usr/sbin/sfdisk
# and the real mke2fs, for the same reason: busybox's is ext2-only and takes no -t, so the installer's
# data partition was never made ("1bit-install: mke2fs failed", with busybox's usage text above it, in
# the VM). /etc/mke2fs.conf comes along or mke2fs silently falls back to its built-in defaults.
# rm first: bin/mke2fs is a symlink to busybox, and cp would write the new binary THROUGH it, into busybox
rm -f "$root/bin/mke2fs"
cp /usr/sbin/mke2fs "$root/bin/mke2fs"; chmod +x "$root/bin/mke2fs"
copy_libs /usr/sbin/mke2fs
cp /etc/mke2fs.conf "$root/etc/mke2fs.conf"
cp "$here/release-check.sh" "$root/bin/1bit-release-check"; chmod +x "$root/bin/1bit-release-check"
cp "$here/udhcpc.sh" "$root/etc/udhcpc.script"; chmod +x "$root/etc/udhcpc.script"
echo "1bit-os" > "$root/etc/hostname"
printf 'root:x:0:0:root:/root:/bin/sh\n' > "$root/etc/passwd"
printf 'root:x:0:\n' > "$root/etc/group"
printf '/bin/sh\n' > "$root/etc/shells"
# Lemonade's default host is "localhost"
printf '127.0.0.1 localhost\n::1 localhost\n' > "$root/etc/hosts"

echo "== licences (os/LICENSES.md)"
doc=$root/usr/share/licenses/1bit-os
mkdir -p "$doc"
cp "$here/../LICENSES.md" "$src/LICENSE" "$src/NOTICE" "$src/LICENSE-BINARY" "$src/NOTICE-BINARY" "$doc/"
cp "$src/third_party/lemonade/LICENSE" "$doc/lemonade-LICENSE" ||
    { echo "no third_party/lemonade/LICENSE (git submodule update --init third_party/lemonade)"; exit 1; }
cp "$licenses/llamacpp-hrx/ROCR-Runtime-LICENSE" "$doc/ROCR-Runtime-LICENSE"   # TheRock's libhsa for HRX
cp -rL /usr/share/common-licenses "$root/usr/share/"
# the Ubuntu package behind each file in the image, found at its path or its merged-/usr alias
(cd "$root" && find . -type f | cut -c2-) > "$out/image-files.txt"
debs() { xargs -r -d '\n' dpkg -S 2>/dev/null | grep -v '^diversion' | sed 's/: \/.*//; s/, /\n/g' | sed 's/:.*//' || true; }
{ debs < "$out/image-files.txt"
  sed -n 's|^/usr/|/|p' "$out/image-files.txt" | debs
  { grep -v '^/usr/' "$out/image-files.txt" || true; } | sed 's|^|/usr|' | debs
  # the NPU firmware is not packaged on every builder; its licence is linux-firmware's amdnpu one
  if [ -d "$root/lib/firmware/amdnpu" ]; then echo linux-firmware-amd-misc; fi
  # generated by update-ca-certificates, so no package owns it: Mozilla's CA bundle, MPL-2.0
  if [ -e "$root/etc/ssl/certs/ca-certificates.crt" ]; then echo ca-certificates; fi
} | sort -u > "$out/image-debs.txt"
while read -r p; do
    [ -e "/usr/share/doc/$p/copyright" ] || { echo "   warning: no copyright file for $p"; continue; }
    mkdir -p "$root/usr/share/doc/$p"; cp -L "/usr/share/doc/$p/copyright" "$root/usr/share/doc/$p/"
done < "$out/image-debs.txt"
{ xargs -r dpkg-query -W -f='${Package} ${Version} ${Architecture}\n' < "$out/image-debs.txt" 2>/dev/null || true; } | awk '$3 == "amd64" || $3 == "all" {print $1, $2}' > "$doc/packages.txt"
# Dropbear and its libraries come from .debs unpacked into $tools, not installed
for p in dropbear-bin libtomcrypt1 libtommath1; do
    [ -e "$tools/usr/share/doc/$p/copyright" ] || continue
    mkdir -p "$root/usr/share/doc/$p"; cp -L "$tools/usr/share/doc/$p/copyright" "$root/usr/share/doc/$p/"
done
echo "   $(wc -l < "$out/image-debs.txt") packages, $(du -sh "$root/usr/share/doc" "$doc" "$root/usr/share/common-licenses" | cut -f1 | tr '\n' ' ')"

echo "== initramfs"
# the builder's umask must not reach the image: nothing group- or world-writable, /root private
# (Dropbear, like sshd, refuses keys when the home directory is group-writable)
chmod -R go-w "$root"; chmod 1777 "$root/tmp"; chmod 700 "$root/root"
# every file owned by root (Dropbear, like sshd, refuses keys in files the user does not own)
(cd "$root" && find . -print0 | cpio --null -o -H newc -R 0:0 --quiet) | gzip -9 -c > "$out/initramfs.img"   # gzip, not zstd: the machine booted its own 37M gzip initrd through grub and never our 106M zstd one

echo "== 1bit-os.efi (unified kernel image)"
stub=$tools/usr/lib/systemd/boot/efi/linuxx64.efi.stub
# the GPU may use up to 120 GiB of RAM (the same limits as the development machine)
cmdline="console=tty0 console=ttyS0,115200 ignore_loglevel loglevel=7 panic=600 oops=panic fbcon=nodefer vt.global_cursor_default=1 rdinit=/init"
printf '%s' "$cmdline" > "$out/cmdline.txt"
# section layout after the stub's own sections, each aligned to 64 KiB
align() { echo $(( ($1 + 65535) / 65536 * 65536 )); }
end=$(objdump -h "$stub" | python3 -c 'import sys; print(max(int(f[3], 16) + int(f[2], 16) for f in (l.split() for l in sys.stdin) if len(f) == 7 and f[0].isdigit()))')
o_cmd=$(align "$end");                 o_linux=$(align $((o_cmd + $(stat -c %s "$out/cmdline.txt"))))
o_initrd=$(align $((o_linux + $(stat -c %s "$vmlinuz"))))
objcopy --add-section .cmdline="$out/cmdline.txt" --change-section-vma .cmdline=$o_cmd \
        --add-section .linux="$vmlinuz" --change-section-vma .linux=$o_linux \
        --add-section .initrd="$out/initramfs.img" --change-section-vma .initrd=$o_initrd \
        "$stub" "$out/1bit-os.efi"

# Secure Boot: sign the unified kernel image with the machine's own enrolled MOK when its key is
# named (MOK_KEY + MOK_CERT), so a signed 1bit OS stick boots with Secure Boot left on rather than
# asking the owner to switch it off. sbsign appends a signature over the whole PE, which is what
# firmware verifies for a UKI; sbverify then proves the result against the same certificate.
if [ -n "${MOK_KEY:-}${MOK_CERT:-}" ]; then
    [ -n "${MOK_KEY:-}" ] && [ -n "${MOK_CERT:-}" ] ||
        { echo "MOK_KEY and MOK_CERT must be given together: sbsign --key <priv> --cert <pem>"; exit 1; }
    command -v sbsign > /dev/null ||
        { echo "sbsign is needed to sign the image: apt install sbsigntool"; exit 1; }
    sbsign --key "$MOK_KEY" --cert "$MOK_CERT" --output "$out/1bit-os.efi.signed" "$out/1bit-os.efi" > /dev/null ||
        { echo "sbsign failed"; exit 1; }
    mv -f "$out/1bit-os.efi.signed" "$out/1bit-os.efi"
    if command -v sbverify > /dev/null; then
        sbverify --cert "$MOK_CERT" "$out/1bit-os.efi" > /dev/null 2>&1 ||
            { echo "the signature does not verify against $MOK_CERT"; exit 1; }
    fi
    echo "   signed with $(basename "$MOK_CERT")"
else
    echo "   unsigned (set MOK_KEY and MOK_CERT to sign it for Secure Boot)"
fi

echo "== 1bit-os.img (USB: EFI partition + 1BIT-DATA)"
efi_mb=$(( $(stat -c %s "$out/1bit-os.efi") / 1048576 + 64 ))
mk() { "$tools/usr/bin/$1" "${@:2}"; }
rm -f "$out/efi.part"; mkfs.fat -F 32 -n 1BIT-OS -C "$out/efi.part" $((efi_mb * 1024)) >/dev/null
MTOOLS_SKIP_CHECK=1 mk mmd -i "$out/efi.part" ::/EFI ::/EFI/BOOT
# Secure Boot again: firmware verifies EFI/BOOT/BOOTX64.EFI against db, and a MOK-signed image is not
# in db - only shim consults MokList. So a signed image is booted the way every Secure Boot machine
# boots: a shim signed by a db key (Canonical's, Microsoft UEFI CA) as BOOTX64.EFI, which then
# verifies our MOK-signed image as its second stage, grubx64.efi. Without signing, the image is
# BOOTX64.EFI itself and only boots with Secure Boot off.
# Two boot layouts. "grub" is the one this machine is proven to boot: the shim (Microsoft-signed,
# in db) hands over to Canonical's own signed grub, which reads grub.cfg and loads a separate kernel
# and initrd. "uki" is the one-EFI-file layout we used before; it boots under QEMU but dies on this
# hardware before userspace, so it stays available behind BOOT_LAYOUT=uki.
BOOT_LAYOUT=${BOOT_LAYOUT:-grub}
if [ "$BOOT_LAYOUT" = grub ]; then
    shim=${SHIM:-/usr/lib/shim/shimx64.efi.signed}
    [ -f "$shim" ] || shim=/boot/efi/EFI/ubuntu/shimx64.efi
    [ -f "$shim" ] || { echo "grub layout needs a Microsoft-signed shim: set SHIM=<shimx64.efi.signed>"; exit 1; }
    grub=${GRUB:-/usr/lib/grub/x86_64-efi-signed/grubx64.efi.signed}
    [ -f "$grub" ] || grub=/boot/efi/EFI/ubuntu/grubx64.efi
    [ -f "$grub" ] || { echo "grub layout needs a signed grub: set GRUB=<grubx64.efi.signed>"; exit 1; }
    mkdir -p "$out/esp/EFI/BOOT" "$out/esp/1bit"
    cp -L "$shim" "$out/esp/EFI/BOOT/BOOTX64.EFI"
    cp -L "$grub" "$out/esp/EFI/BOOT/grubx64.efi"
    [ -f "${MOK_MANAGER:-/usr/lib/shim/mmx64.efi}" ] && cp -L "${MOK_MANAGER:-/usr/lib/shim/mmx64.efi}" "$out/esp/EFI/BOOT/mmx64.efi"
    # grub under Secure Boot only loads a signed kernel, so the kernel is signed with the same enrolled
    # MOK (it already is on this box, but sign our copy rather than trusting that).
    cp -L "$vmlinuz" "$out/esp/1bit/vmlinuz"
    if [ -n "${MOK_KEY:-}${MOK_CERT:-}" ] && command -v sbsign > /dev/null; then
        sbsign --key "$MOK_KEY" --cert "$MOK_CERT" --output "$out/esp/1bit/vmlinuz.signed" "$out/esp/1bit/vmlinuz" > /dev/null &&
            mv -f "$out/esp/1bit/vmlinuz.signed" "$out/esp/1bit/vmlinuz"
    fi
    cp -L "$out/initramfs.img" "$out/esp/1bit/initramfs.img"
    cat > "$out/esp/EFI/BOOT/grub.cfg" <<GRUBCFG
set timeout=12
set default=0
menuentry "1bit OS v2026.43 - LIVE (runs from RAM, nothing installed)" {
    linux /1bit/vmlinuz $cmdline
    initrd /1bit/initramfs.img
}
menuentry "1bit OS - LIVE, verbose, no GPU driver" {
    linux /1bit/vmlinuz ${cmdline} 1bit.nogpu=1
    initrd /1bit/initramfs.img
}
menuentry "1bit OS - root shell only" {
    linux /1bit/vmlinuz $cmdline 1bit.shell=1
    initrd /1bit/initramfs.img
}
GRUBCFG
    # Bisect entry: the machine's own signed kernel + initrd, on the same stick, through the same
    # shim->grub chain. If this reaches userspace and ours does not, the boot path is proven and the
    # fault is in our kernel/initramfs payload.
    if [ -n "${BOOT_TEST_KERNEL:-}" ] && [ -f "${BOOT_TEST_KERNEL}" ]; then
        mkdir -p "$out/esp/test"
        cp -L "${BOOT_TEST_KERNEL}" "$out/esp/test/vmlinuz"
        [ -n "${BOOT_TEST_INITRD:-}" ] && [ -f "${BOOT_TEST_INITRD}" ] && cp -L "${BOOT_TEST_INITRD}" "$out/esp/test/initrd.img"
        cat >> "$out/esp/EFI/BOOT/grub.cfg" <<TESTCFG
menuentry "TEST: this machine's own kernel + initrd (bisect)" {
    linux /test/vmlinuz root=/dev/ram0 rw console=tty0 console=ttyS0,115200
    initrd /test/initrd.img
}
TESTCFG
        echo "   ESP: bisect entry added ($(basename "${BOOT_TEST_KERNEL}"))"
    fi
    echo "   ESP: shim + signed grub + separate vmlinuz/initramfs + grub.cfg (layout=grub)"
else
    MTOOLS_SKIP_CHECK=1 mk mcopy -i "$out/efi.part" "$out/1bit-os.efi" ::/EFI/BOOT/BOOTX64.EFI
    echo "   ESP: unified kernel image as BOOTX64.EFI (layout=uki)"
fi
if [ "$BOOT_LAYOUT" = grub ]; then
    MTOOLS_SKIP_CHECK=1 mk mcopy -s -i "$out/efi.part" "$out/esp"/* ::/ ||
        { echo "copying the ESP tree into the FAT failed"; exit 1; }
fi
rm -rf "$out/data.dir"; mkdir -p "$out/data.dir/models" "$out/data.dir/ssh" "$out/data.dir/lemonade/cache" "$out/data.dir/lemonade/config"
cat > "$out/data.dir/models/README" <<'TXT'
Put GGUF files and NPU model directories here. Lemonade lists every GGUF in this folder
(extra_models_dir in lemonade/config/config.json; its llamacpp recipe), and 1bit OS also
registers each one, and each NPU model directory, for the onebit recipe (the 1bit engine)
in lemonade/config/user_models.json, as <name>-1bit. Models Lemonade downloads go to
lemonade/huggingface.
TXT
# Lemonade's backends, installed above, in Lemonade's cache directory on the stick
for b in $backends; do
    mkdir -p "$out/data.dir/lemonade/cache/bin/${b%%:*}"
    cp -a "$seed/cache/bin/${b%%:*}/${b#*:}" "$out/data.dir/lemonade/cache/bin/${b%%:*}/"
done
# their licences (os/LICENSES.md): what the release archives leave out, beside what they hold
for r in $(for b in $backends; do echo "${b%%:*}"; done | sort -u); do
    mkdir -p "$out/data.dir/lemonade/licenses"; cp -a "$licenses/$r" "$out/data.dir/lemonade/licenses/$r"
done
if [ -d "$out/data.dir/lemonade/licenses/llamacpp-hrx" ]; then
    # the TheRock system libraries in its lib/rocm_sysdeps: the Ubuntu copyright files of the same projects
    d=$out/data.dir/lemonade/licenses/llamacpp-hrx/rocm_sysdeps; mkdir -p "$d"
    for p in libelf1t64 libdw1t64 libasm1t64 libnl-3-200 libnuma1 libmnl0 libcap2 libdrm2 libbz2-1.0 liblzma5 libsqlite3-0; do
        if [ -e "/usr/share/doc/$p/copyright" ]; then cp -L "/usr/share/doc/$p/copyright" "$d/$p.copyright"; fi
    done
fi
if [ -d "$out/data.dir/lemonade/licenses" ]; then cp "$here/../LICENSES.md" "$out/data.dir/lemonade/licenses/"; fi
# Lemonade's own config file: where the models are, and downloads kept on the stick; host, port
# and everything else are Lemonade's defaults (localhost:13305)
if [ -n "${LEMONADE_CONFIG:-}" ]; then cp "$LEMONADE_CONFIG" "$out/data.dir/lemonade/config/config.json"
else printf '{\n    "extra_models_dir": "/data/models",\n    "models_dir": "/data/lemonade/huggingface",\n    "host": "0.0.0.0",\n    "port": 13305\n}\n' > "$out/data.dir/lemonade/config/config.json"; fi
# Lemonade's environment file (its systemd unit's EnvironmentFile, /etc/default/lemond)
if [ -n "${LEMONADE_ENV:-}" ]; then cp "$LEMONADE_ENV" "$out/data.dir/lemonade/lemond.conf"
else
    { echo "# Lemonade's environment, read at boot (the file Lemonade's systemd unit loads from"
      echo "# /etc/default/lemond). LEMONADE_API_KEY makes every API request present it as a Bearer token;"
      echo "# set it before opening Lemonade to the network (\"host\" in config/config.json)."
      grep -v '^# Installed as\|^# EnvironmentFile\|^# You can also' "$src/third_party/lemonade/data/secrets.conf" 2>/dev/null ||
          printf '#HF_TOKEN=\n#LEMONADE_API_KEY=\n#LEMONADE_ADMIN_API_KEY=\n'
    } > "$out/data.dir/lemonade/lemond.conf"
fi
chmod 600 "$out/data.dir/lemonade/lemond.conf"
echo "Put your SSH public key(s) in authorized_keys here; 1bit OS takes key logins only." > "$out/data.dir/ssh/README"
# optional contents of the data partition (tests, or a ready-to-go stick)
[ -n "${AUTHORIZED_KEYS:-}" ] && cp "$AUTHORIZED_KEYS" "$out/data.dir/ssh/authorized_keys"
for m in ${MODELS:-}; do
    if [ -d "$m" ]; then cp -aL "$m" "$out/data.dir/models/"      # an NPU model set (.q4nx); -L: FLM ships symlinks
    else cp -L "$m" "$out/data.dir/models/"; fi
done
data_mb=${DATA_MB:-$(( $(du -sm "$out/data.dir" | cut -f1) + 1024 ))}   # the contents plus 1 GiB
rm -f "$out/data.part"; if [ -d "$build/npu/kernels" ]; then
    # The sealed sets go on the STICK'S DATA PARTITION, not inside the RAM image. They are ~124 MB and
    # were making the initramfs 103 MB compressed, which is what grub was failing to load on real
    # hardware (the machine's own 37 MB initrd loads fine). init.sh links them to the engine's
    # expected path at boot, so the engine still finds npu/kernels beside itself.
    # Only the artifact surface goes on the stick: <set>/layout.json and <set>/layer1_*/layer1.elf, the
    # allowlist scripts/package-linux.sh packs. A set directory also holds layer1.work/ (plaintext
    # MLIR, object files, intermediate ELFs) because that is where it was built, so copying the tree
    # would put the kernel method on the stick; the guard that caught that used `find ... | head -3`,
    # which exits 141 (SIGPIPE) under `set -o pipefail` and aborted the whole image instead.
    mkdir -p "$out/data.dir/npu/kernels"
    for set in "$build"/npu/kernels/*/; do
        setname=$(basename "$set")
        [ -f "$set/layout.json" ] || continue
        mkdir -p "$out/data.dir/npu/kernels/$setname"
        cp -a "$set/layout.json" "$out/data.dir/npu/kernels/$setname/"
        for layer in "$set"/layer1_*/; do
            [ -f "$layer/layer1.elf" ] || continue
            mkdir -p "$out/data.dir/npu/kernels/$setname/$(basename "$layer")"
            cp -a "$layer/layer1.elf" "$out/data.dir/npu/kernels/$setname/$(basename "$layer")/"
        done
    done
    sealed=$(find "$out/data.dir/npu/kernels" -name layer1.elf -exec \
        sh -c 'head -c 8 "$1" | grep -q 1BPKCRYP' _ {} \; -print | wc -l)
    total=$(find "$out/data.dir/npu/kernels" -name layer1.elf | wc -l)
    [ "$total" -gt 0 ] || { echo "npu/kernels in $build carries no layer1.elf"; exit 1; }
    [ "$sealed" = "$total" ] || { echo "npu/kernels: $sealed of $total kernels are sealed (plaintext would leak the method)"; exit 1; }
    echo "   npu/kernels: $(ls "$out/data.dir/npu/kernels" | wc -l) sealed sets on the data partition ($sealed kernels)"
fi
mkdir -p "$root/opt/1bit-llama" "$root/etc/1bit"
hrx=$build/hrx/llama/bin
cache_get() { sed -n "s/^$1:[A-Z]*=//p" "$build/CMakeCache.txt" 2>/dev/null; }
if [ -x "$hrx/llama-server" ]; then
    # at its build path: 1bit serve runs it from there (ONEBIT_HRX_SERVER), for HRX0 and the CPU
    copy_elf "$hrx/llama-server" $(ls "$hrx"/lib*.so* 2>/dev/null)
    ln -s "$hrx/llama-server" "$root/opt/1bit-llama/llama-server"
    # TheRock's HSA runtime: the distro's rejects gfx1151's PM4-emulation probe (docs/hrx.md)
    hsa=$(cache_get ONEBIT_HRX_LIBHSA)
    [ -n "$hsa" ] && [ -e "$hsa" ] || { echo "no ONEBIT_HRX_LIBHSA in $build/CMakeCache.txt"; exit 1; }
    copy_elf "$hsa"
    echo "$hsa" > "$root/etc/1bit/libhsa"
    # libdrm's GPU name table, where TheRock has one (names only; HSA works without it)
    for ids in "$(dirname "$hsa")"/../share/libdrm/amdgpu.ids "$(cache_get ONEBIT_HRX_TOOLCHAIN)"/share/libdrm/amdgpu.ids; do
        if [ -e "$ids" ]; then ids=$(realpath "$ids"); mkdir -p "$root$(dirname "$ids")"; cp "$ids" "$root$ids"; break; fi
    done
    echo "   HRX llama-server ($hrx), HSA runtime $hsa"
elif [ -n "${LLAMA_SERVER:-}" ]; then
    # test images only: a CPU llama-server, found on PATH by a 1bit built without HRX
    ls=$(realpath "$LLAMA_SERVER")
    copy_elf "$ls" $(ls "$(dirname "$ls")"/lib*.so* 2>/dev/null)
    ln -s "$ls" "$root/opt/1bit-llama/llama-server"
    echo "   CPU llama-server ($ls): a test image, no HRX"
else
    echo "no HRX llama-server in $build (build with -DONEBIT_HRX=ON; LLAMA_SERVER=<path> for a CPU-only test image)"; exit 1
fi
mkfs.ext4 -q -L 1BIT-DATA -d "$out/data.dir" "$out/data.part" $((data_mb))M
img=$out/1bit-os.img; rm -f "$img"
truncate -s $(( (efi_mb + data_mb + 2) * 1048576 )) "$img"
printf 'label: gpt\nstart=2048, size=%s, type=U, name="EFI"\ntype=L, name="1BIT-DATA"\n' $((efi_mb * 2048)) | sfdisk -q "$img"
dd if="$out/efi.part" of="$img" bs=1M seek=1 conv=notrunc status=none
dd if="$out/data.part" of="$img" bs=1M seek=$((efi_mb + 1)) conv=notrunc status=none
rm -f "$out/efi.part" "$out/data.part"
ls -la "$out"/1bit-os.efi "$out"/1bit-os.img "$out"/initramfs.img
echo "write it:  sudo dd if=$img of=/dev/sdX bs=4M conv=fsync   (then grow 1BIT-DATA to fill the stick)"
