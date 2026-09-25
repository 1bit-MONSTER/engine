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
# The initramfs holds only what 1bit needs: BusyBox, the kernel modules for the GPU (amdgpu),
# the NPU (amdxdna), Ethernet, Wi-Fi and USB storage, the firmware those load, Mesa's Radeon
# Vulkan driver, XRT with the XDNA plugin, and the engine (1bit, and its Vulkan llama-server),
# each with exactly the shared libraries it links.
#
# Needs no root: the EFI stub and mtools come from their Debian packages, unpacked locally.
set -euo pipefail
out=${1:?usage: mkimage.sh <out> [engine build dir]}
build=${2:-$HOME/.cache/1bit-os/src/build}
build=$(cd "$build" && pwd)   # absolute: binaries keep their build paths inside the image
kver=${KVER:-$(uname -r)}
here=$(cd "$(dirname "$0")" && pwd)
tools=$HOME/.cache/1bit-os/tools/x
mkdir -p "$out"; out=$(cd "$out" && pwd)
root=$out/root
rm -rf "$root"; mkdir -p "$root"/{bin,sbin,etc,proc,sys,dev,tmp,run,data,lib/firmware,usr/share/vulkan/icd.d,root}

# --- a file and every shared library it links, at the same paths -------------------------
copy_elf() {
    local f
    for f in "$@"; do
        [ -e "$f" ] || { echo "missing: $f"; exit 1; }
        mkdir -p "$root$(dirname "$f")"; cp -L "$f" "$root$f"
        ldd "$f" 2>/dev/null | grep -o '/[^ ]*' | while read -r lib; do
            [ -e "$root$lib" ] || { mkdir -p "$root$(dirname "$lib")"; cp -L "$lib" "$root$lib"; }
        done
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

echo "== Vulkan (Mesa RADV) and XRT"
radv=$(ls /usr/lib/x86_64-linux-gnu/libvulkan_radeon.so)
copy_elf /usr/lib/x86_64-linux-gnu/libvulkan.so.1 "$radv"
printf '{"file_format_version":"1.0.1","ICD":{"library_path":"%s","api_version":"1.4"}}\n' "$radv" \
    > "$root/usr/share/vulkan/icd.d/radeon_icd.json"
if [ -n "${LAVAPIPE:-}" ]; then   # test images only: Vulkan on the CPU, for a VM with no Radeon GPU
    lvp=/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so
    copy_elf "$lvp"
    printf '{"file_format_version":"1.0.1","ICD":{"library_path":"%s","api_version":"1.4"}}\n' "$lvp" \
        > "$root/usr/share/vulkan/icd.d/lvp_icd.json"
fi
copy_elf /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 /opt/xilinx/xrt/lib/libxrt_core.so.2 \
         /opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2
(cd "$root/opt/xilinx/xrt/lib" && for l in libxrt_coreutil libxrt_core libxrt_driver_xdna; do ln -sf $l.so.2 $l.so; done)

echo "== the engine"
bin=$build/onebit; [ -x "$bin" ] || bin=$build/1bit
[ -x "$bin" ] || { echo "no engine binary in $build (build with -DONEBIT_NPU=ON -DONEBIT_VULKAN=ON)"; exit 1; }
copy_elf "$bin"
ln -s "$bin" "$root/bin/1bit"
vk=$build/vulkan/llama/bin
if [ -x "$vk/llama-server" ]; then
    copy_elf "$vk/llama-server" $(ls "$vk"/lib*.so* 2>/dev/null)   # at its build path: 1bit serve runs it from there
    mkdir -p "$root/opt/1bit-llama"; ln -s "$vk/llama-server" "$root/opt/1bit-llama/llama-server"
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

echo "== initramfs"
(cd "$root" && find . -print0 | cpio --null -o -H newc --quiet) | zstd -q -19 -T0 > "$out/initramfs.img"

echo "== 1bit-os.efi (unified kernel image)"
stub=$tools/usr/lib/systemd/boot/efi/linuxx64.efi.stub
# the GPU may use up to 120 GiB of RAM (the same limits as the development machine)
cmdline="console=tty0 console=ttyS0,115200 quiet rdinit=/init ttm.pages_limit=31457280 ttm.page_pool_size=31457280"
printf '%s' "$cmdline" > "$out/cmdline.txt"
# section layout after the stub's own sections, each aligned to 64 KiB
align() { echo $(( ($1 + 65535) / 65536 * 65536 )); }
end=$(objdump -h "$stub" | python3 -c 'import sys; print(max(int(f[3], 16) + int(f[2], 16) for f in (l.split() for l in sys.stdin) if len(f) == 7 and f[0].isdigit()))')
o_cmd=$(align "$end");                 o_linux=$(align $((o_cmd + $(stat -c %s "$out/cmdline.txt"))))
o_initrd=$(align $((o_linux + $(stat -c %s /boot/vmlinuz-"$kver"))))
objcopy --add-section .cmdline="$out/cmdline.txt" --change-section-vma .cmdline=$o_cmd \
        --add-section .linux=/boot/vmlinuz-"$kver" --change-section-vma .linux=$o_linux \
        --add-section .initrd="$out/initramfs.img" --change-section-vma .initrd=$o_initrd \
        "$stub" "$out/1bit-os.efi"

echo "== 1bit-os.img (USB: EFI partition + 1BIT-DATA)"
efi_mb=$(( $(stat -c %s "$out/1bit-os.efi") / 1048576 + 64 ))
mk() { "$tools/usr/bin/$1" "${@:2}"; }
rm -f "$out/efi.part"; mkfs.fat -F 32 -n 1BIT-OS -C "$out/efi.part" $((efi_mb * 1024)) >/dev/null
MTOOLS_SKIP_CHECK=1 mk mmd -i "$out/efi.part" ::/EFI ::/EFI/BOOT
MTOOLS_SKIP_CHECK=1 mk mcopy -i "$out/efi.part" "$out/1bit-os.efi" ::/EFI/BOOT/BOOTX64.EFI
rm -rf "$out/data.dir"; mkdir -p "$out/data.dir/models" "$out/data.dir/ssh"
echo "Put GGUF files and NPU model directories here." > "$out/data.dir/models/README"
echo "Put your SSH public key(s) in authorized_keys here; 1bit OS takes key logins only." > "$out/data.dir/ssh/README"
# optional contents of the data partition (tests, or a ready-to-go stick)
[ -n "${AUTHORIZED_KEYS:-}" ] && cp "$AUTHORIZED_KEYS" "$out/data.dir/ssh/authorized_keys"
for m in ${MODELS:-}; do cp -L "$m" "$out/data.dir/models/"; done
[ -n "${CONF:-}" ] && cp "$CONF" "$out/data.dir/1bit.conf"
cat > "$out/data.dir/1bit.conf.example" <<'CONF'
# 1bit OS starts `1bit serve` at boot when 1bit.conf (this file, renamed) names a model.
MODEL=/data/models/Qwen3-32B-UD-Q4_K_XL.gguf
DEVICE=vulkan
PORT=8000
# anything else for 1bit serve, e.g. --mtp or --parallel 4
ARGS=
CONF
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
