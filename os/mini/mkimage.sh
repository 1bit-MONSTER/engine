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
# Ethernet, Wi-Fi and USB storage, the firmware those load, XRT with the XDNA plugin, Mesa's Radeon
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
#   LEMONADE=<dir>        a build-lemonade.sh prefix with bin/lemond (default <engine build>/lemonade,
#                         built with the web app when missing)
#   LEMONADE_CONFIG=<f>   Lemonade's config.json for the stick (default: models from /data/models,
#                         downloads on /data; everything else Lemonade's defaults)
#   LEMONADE_ENV=<f>      Lemonade's environment file for the stick (LEMONADE_API_KEY, ...)
#
# Test images only (vmtest.sh on a machine without TheRock or XRT): LLAMA_SERVER=<a CPU
# llama-server> stands in for the HRX build, and NO_XRT=1 leaves XRT out.
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
# the engine builds no kernel: the Strix Halo kernel packages come from github.com/1bit-MONSTER/kernel
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
mods="amdgpu amdxdna r8169 mt7925e xhci_pci usb_storage uas sd_mod vfat ext4 nls_iso8859_1 nls_cp437 virtio_net virtio_blk"
for m in $mods; do
    modprobe -S "$kver" --show-depends "$m" 2>/dev/null | awk '$1 == "insmod" {print $2}'
done | sort -u | while read -r ko; do mkdir -p "$root$(dirname "$ko")"; cp "$ko" "$root$ko"; done
cp /lib/modules/"$kver"/modules.{order,builtin,builtin.modinfo} "$root/lib/modules/$kver/" 2>/dev/null || true
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

echo "== XRT"
if [ -n "${NO_XRT:-}" ]; then
    echo "   left out (NO_XRT=1: a test image, no NPU)"
    mkdir -p "$root/etc/1bit"; touch "$root/etc/1bit/no_xrt"
else
    copy_elf /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 /opt/xilinx/xrt/lib/libxrt_core.so.2 \
             /opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2
    (cd "$root/opt/xilinx/xrt/lib" && for l in libxrt_coreutil libxrt_core libxrt_driver_xdna; do ln -sf $l.so.2 $l.so; done)
fi

echo "== the engine"
bin=$build/onebit; [ -x "$bin" ] || bin=$build/1bit
[ -x "$bin" ] || { echo "no engine binary in $build (build with -DONEBIT_NPU=ON -DONEBIT_HRX=ON)"; exit 1; }
copy_elf "$bin"
ln -s "$bin" "$root/bin/1bit"
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
backends=${LEMONADE_BACKENDS-llamacpp:vulkan llamacpp:cpu llamacpp-hrx:hrx whispercpp:vulkan whispercpp:cpu sd-cpp:vulkan sd-cpp:cpu kokoro:cpu moonshine:cpu onnxruntime:cpu acestep:vulkan thinksound:vulkan trellis:vulkan openmoss:vulkan flm:npu}
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
    # the system libraries they link (everything else ships inside each backend's directory)
    find "$seed/cache/bin" -type f \( -perm -u+x -o -name '*.so*' \) | while read -r f; do
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
cp "$here/init.sh" "$root/init"; chmod +x "$root/init"
cp "$here/udhcpc.sh" "$root/etc/udhcpc.script"; chmod +x "$root/etc/udhcpc.script"
echo "1bit-os" > "$root/etc/hostname"
printf 'root:x:0:0:root:/root:/bin/sh\n' > "$root/etc/passwd"
printf 'root:x:0:\n' > "$root/etc/group"
printf '/bin/sh\n' > "$root/etc/shells"
# Lemonade's default host is "localhost"
printf '127.0.0.1 localhost\n::1 localhost\n' > "$root/etc/hosts"

echo "== initramfs"
# the builder's umask must not reach the image: nothing group- or world-writable, /root private
# (Dropbear, like sshd, refuses keys when the home directory is group-writable)
chmod -R go-w "$root"; chmod 1777 "$root/tmp"; chmod 700 "$root/root"
# every file owned by root (Dropbear, like sshd, refuses keys in files the user does not own)
(cd "$root" && find . -print0 | cpio --null -o -H newc -R 0:0 --quiet) | zstd -q -19 -T4 > "$out/initramfs.img"

echo "== 1bit-os.efi (unified kernel image)"
stub=$tools/usr/lib/systemd/boot/efi/linuxx64.efi.stub
# the GPU may use up to 120 GiB of RAM (the same limits as the development machine)
cmdline="console=tty0 console=ttyS0,115200 quiet rdinit=/init ttm.pages_limit=31457280 ttm.page_pool_size=31457280"
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

echo "== 1bit-os.img (USB: EFI partition + 1BIT-DATA)"
efi_mb=$(( $(stat -c %s "$out/1bit-os.efi") / 1048576 + 64 ))
mk() { "$tools/usr/bin/$1" "${@:2}"; }
rm -f "$out/efi.part"; mkfs.fat -F 32 -n 1BIT-OS -C "$out/efi.part" $((efi_mb * 1024)) >/dev/null
MTOOLS_SKIP_CHECK=1 mk mmd -i "$out/efi.part" ::/EFI ::/EFI/BOOT
MTOOLS_SKIP_CHECK=1 mk mcopy -i "$out/efi.part" "$out/1bit-os.efi" ::/EFI/BOOT/BOOTX64.EFI
rm -rf "$out/data.dir"; mkdir -p "$out/data.dir/models" "$out/data.dir/ssh" "$out/data.dir/lemonade/cache" "$out/data.dir/lemonade/config"
cat > "$out/data.dir/models/README" <<'TXT'
Put GGUF files and NPU model directories here. Lemonade lists every GGUF in this folder
(extra_models_dir in lemonade/config/config.json; its llamacpp recipe), and 1bit OS also
registers each one, and each NPU model directory, for the onebit recipe (the 1bit engine)
in lemonade/config/user_models.json, as <name>-1bit. Models Lemonade downloads go to
lemonade/huggingface.
TXT
# Lemonade's backends, installed above, in Lemonade's cache directory on the stick
[ -n "$backends" ] && cp -a "$seed/cache/bin" "$out/data.dir/lemonade/cache/bin"
# Lemonade's own config file: where the models are, and downloads kept on the stick; host, port
# and everything else are Lemonade's defaults (localhost:13305)
if [ -n "${LEMONADE_CONFIG:-}" ]; then cp "$LEMONADE_CONFIG" "$out/data.dir/lemonade/config/config.json"
else printf '{\n    "extra_models_dir": "/data/models",\n    "models_dir": "/data/lemonade/huggingface"\n}\n' > "$out/data.dir/lemonade/config/config.json"; fi
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
for m in ${MODELS:-}; do cp -L "$m" "$out/data.dir/models/"; done
data_mb=${DATA_MB:-$(( $(du -sm "$out/data.dir" | cut -f1) + 1024 ))}   # the contents plus 1 GiB
rm -f "$out/data.part"; mkfs.ext4 -q -L 1BIT-DATA -d "$out/data.dir" "$out/data.part" $((data_mb))M
img=$out/1bit-os.img; rm -f "$img"
truncate -s $(( (efi_mb + data_mb + 2) * 1048576 )) "$img"
printf 'label: gpt\nstart=2048, size=%s, type=U, name="EFI"\ntype=L, name="1BIT-DATA"\n' $((efi_mb * 2048)) | sfdisk -q "$img"
dd if="$out/efi.part" of="$img" bs=1M seek=1 conv=notrunc status=none
dd if="$out/data.part" of="$img" bs=1M seek=$((efi_mb + 1)) conv=notrunc status=none
rm -f "$out/efi.part" "$out/data.part"
ls -la "$out"/1bit-os.efi "$out"/1bit-os.img "$out"/initramfs.img
echo "write it:  sudo dd if=$img of=/dev/sdX bs=4M conv=fsync   (then grow 1BIT-DATA to fill the stick)"
