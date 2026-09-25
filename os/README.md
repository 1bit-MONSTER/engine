<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# 1bit OS

The 1bit engine as its own operating system: it boots a Strix Halo machine from a USB stick,
runs entirely in RAM, and never touches the internal disk. Nothing but what the engine needs
is in it: the kernel with the GPU (`amdgpu`) and NPU (`amdxdna`) drivers, the firmware those
chips load, BusyBox, Mesa's Radeon Vulkan driver, XRT with the XDNA plugin, and the engine
(`1bit`, and the Vulkan `llama-server` it runs). Models live on the stick's `1BIT-DATA`
partition.

## Build it (a few minutes, no root)

On a machine that has built the engine with the NPU lane and Vulkan:

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DONEBIT_NPU=ON -DONEBIT_VULKAN=ON
cmake --build build
os/mini/mkimage.sh out build
```

`mkimage.sh` takes the running kernel (or `KVER=`), copies only the modules, firmware and
shared libraries the stack uses, and writes:

| File | What |
|---|---|
| `out/1bit-os.efi` | the OS in one EFI file (kernel, command line, initramfs), about 100 MB |
| `out/1bit-os.img` | a USB image: an EFI partition that boots `1bit-os.efi`, and `1BIT-DATA` for models |

It needs `systemd-boot-efi` (the EFI stub) and `mtools`; unpack both without installing:
`cd ~/.cache/1bit-os/tools && apt-get download systemd-boot-efi mtools && for d in *.deb; do dpkg -x $d x; done`.

## Put it on a stick and boot

```sh
sudo dd if=out/1bit-os.img of=/dev/sdX bs=4M conv=fsync   # the whole stick is overwritten
sudo growpart /dev/sdX 2 && sudo resize2fs /dev/sdX2         # 1BIT-DATA fills the stick
```

Copy models into `models/` on `1BIT-DATA`, then boot the Strix Halo machine from the stick
(its firmware's boot menu, Secure Boot off). It comes up with a shell, DHCP on Ethernet, and:

```sh
1bit serve -m /data/models/<file.gguf> --device vulkan --host 0.0.0.0
```

## Check it without the hardware

```sh
qemu-system-x86_64 -enable-kvm -cpu host -m 4G -kernel /boot/vmlinuz-$(uname -r) \
  -initrd out/initramfs.img -append "console=ttyS0 rdinit=/init 1bit.check" -nographic -no-reboot
```

`1bit.check` makes init check that the engine, `llama-server`, the XRT plugin and the NPU
firmware are in place and start, print `1BIT-OS CHECK PASS` or `FAIL`, and power off.

## Next

- HRX (the HSA runtime and the HRX llama.cpp build) alongside Vulkan and the NPU.
- Lemonade in the image, with its web UI on the network; SSH.
- Mesa without LLVM (RADV compiles with ACO): the image drops by about 130 MB.
- Every piece pinned upstream and bumped by a workflow, like the rest of the engine.
