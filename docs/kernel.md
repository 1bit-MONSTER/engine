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
# Linux kernel

The engine's NPU driver (`amdxdna`) and GPU driver (`amdgpu`) come from the
kernel, so the kernel is pinned like every other dependency:
`third_party/linux` is upstream `torvalds/linux` at a release tag, built into
Debian packages by `scripts/build-kernel.sh`.

## The pin

| | |
|---|---|
| Source | `torvalds/linux` **v7.3-rc4** (`93f51579`) |
| Config | `config/kernel/strixhalo.config`: the config of the kernel Strix Halo ran on 2026-09-23 (`7.2.0-next-20260821-unstable-ogc`), plus `config/kernel/strixhalo.fragment` |
| Fragment | `DRM_ACCEL=y`, **`DRM_ACCEL_AMDXDNA=m`**, `DRM_AMDGPU=m`, `HSA_AMD=y`, `LOCALVERSION=-1bit`, our own module-signing key |
| Kept current by | `.github/workflows/bump-linux.yml`: daily, moves to the newest upstream tag (release or -rc) |

The base config had `amdxdna` switched off, and the running system loaded a
separately built `amdxdna.ko`. In the pinned kernel it is built in-tree
from the same source.

## Build

```sh
# Debian/Ubuntu build deps:
#   clang lld llvm flex bison bc dwarves libelf-dev libdw-dev libssl-dev dpkg-dev debhelper rsync kmod cpio zstd
scripts/build-kernel.sh <out>          # uses third_party/linux (fetched at the pin, depth 1)
LINUX_SRC=<tree> scripts/build-kernel.sh <out>   # or an existing checkout of the pinned commit
```

It builds with LLVM (clang + lld), like the base config's kernel, out of tree
in `<out>/obj`. It refuses to continue if `olddefconfig` dropped
`DRM_ACCEL_AMDXDNA=m`, and writes `linux-image`, `linux-headers`,
`linux-libc-dev` (and `-dbg`) packages into `<out>`. It installs nothing.

Verified 2026-09-23 on a 256-core build box (Ubuntu 24.04, clang 18): the
source was the `v7.3-rc4` tarball from git.kernel.org, whose embedded commit id
was checked to be `93f51579` before building. The image package contains
`vmlinuz-7.3.0-rc4-1bit`, `amdxdna.ko.xz` and `amdgpu.ko.xz`.

## Installing (deliberate, needs a reboot)

A reboot stops every NPU and GPU job on the machine, so installation is a
separate step, never part of a build:

```sh
sudo apt install ./linux-image-7.3.0-rc4-1bit_*.deb ./linux-headers-7.3.0-rc4-1bit_*.deb
# keep the previous kernel as a GRUB entry to fall back to, then reboot
```

After the reboot, rerun the hardware tests: `amdxdna` loads,
`tests/npu_lane_e2e.sh` (24/24) and `tests/serve_e2e.sh` on every device.
