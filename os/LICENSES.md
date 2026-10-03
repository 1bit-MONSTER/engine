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
# 1bit OS: third-party software on the image

The 1bit OS image (`os/mini/mkimage.sh`) carries other projects' programs next to the engine:
Lemonade and its backends, Mesa's Radeon Vulkan driver, LLVM, BusyBox, Dropbear, the Linux kernel
and firmware, and the Ubuntu system libraries those programs link. Each keeps its own license.
This file lists them, with the release on the image, the license and where its source is. It was
written from the licenses inside each release archive, or the upstream repository at that tag
where the archive has none (reviewed 2026-10-03, Lemonade fork pin b333eb0, its
`backend_versions.json` sha256 13bab8dbf87b12e2...).

On the image:

| Where | What |
|---|---|
| `/usr/share/licenses/1bit-os/` | this file, the engine's `LICENSE` and `NOTICE`, Lemonade's license (`lemonade-LICENSE`), the ROCr HSA runtime license, and `packages.txt`: every Ubuntu package the image takes files from, with its exact version |
| `/usr/share/doc/<package>/copyright` | the Ubuntu copyright file of each of those packages |
| `/usr/share/common-licenses/` | the GPL, LGPL, Apache, MPL and other texts those copyright files refer to |
| `/data/lemonade/licenses/<recipe>/` | per Lemonade backend: `SOURCE` (project, release, licenses, source) and the license texts its release archive leaves out (from `os/licenses/<recipe>/`) |
| `/data/lemonade/cache/bin/<recipe>/<backend>/` | the backends themselves, with the license files their archives hold |
| `/opt/lemonade/bin/resources/web-app/renderer.bundle.js.LICENSE.txt` | the licenses of the JavaScript bundled into Lemonade's web app |

A backend ships only when `os/licenses/<recipe>/SOURCE` says `Ship: yes`; mkimage.sh stops on
any other backend in `LEMONADE_BACKENDS`.

## Lemonade's backends

Installed ahead of time by Lemonade's own installer (lemond `/api/v1/install`), from the release
assets its `backend_versions.json` pins. None of these archives holds model weights except
moonshine (its English tiny model, MIT); the others download models when used, under each
model's own license.

| Recipe:backend | Project (release asset source) | Release | License | Status |
|---|---|---|---|---|
| llamacpp:vulkan, llamacpp:cpu | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | b10825 | MIT (the ggml authors); vendored nlohmann/json and cpp-httplib MIT | ships: license notice |
| llamacpp-hrx:hrx | [ROCm/ggml-staging-automation](https://github.com/ROCm/ggml-staging-automation) | hrx-b69 | MIT (AMD); llama.cpp MIT; hrx-system Apache-2.0; ROCr NCSA; aqlprofile MIT; TheRock sysdeps: elfutils LGPL-3.0+/GPL-2.0+, libnl LGPL-2.1, numactl LGPL-2.1, libmnl LGPL-2.1+, libcap BSD-3-Clause/GPL-2.0, libdrm MIT, bzip2, xz 0BSD, SQLite public domain | ships: notices, LGPL source |
| whispercpp:vulkan, whispercpp:cpu | [lemonade-sdk/whisper.cpp-rocm](https://github.com/lemonade-sdk/whisper.cpp-rocm) (whisper.cpp) | v1.8.4 | MIT (the ggml authors) | ships: license notice |
| sd-cpp:vulkan, sd-cpp:cpu | [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | master-843-462d675 | MIT (leejet; ggml); bundled libwebp and libwebm BSD-3-Clause (Google) | ships: license notices |
| kokoro:cpu | [lemonade-sdk/Kokoros](https://github.com/lemonade-sdk/Kokoros) (fork of lucasjinreal/Kokoros) | b17 | Apache-2.0 by README statement (no LICENSE file); statically links eSpeak NG, GPL-3.0-or-later, so the binary as a whole is GPL-3.0-or-later; espeak-ng-data GPL-3.0-or-later; ONNX Runtime MIT | ships: GPL-3.0 source |
| moonshine:cpu | [lemonade-sdk/moonshine-server-rocm](https://github.com/lemonade-sdk/moonshine-server-rocm) (moonshine-voice 0.0.62 + Lemonade's main.py, PyInstaller) | moonshine0.0.62 | moonshine-voice MIT (code and English tiny model); Lemonade Apache-2.0; Python PSF; numpy BSD-3-Clause with OpenBLAS BSD, libgfortran GPL-3.0+ with the GCC runtime exception, libquadmath LGPL-2.1+; websockets BSD-3-Clause; tqdm MPL-2.0 and MIT; certifi MPL-2.0; charset_normalizer MIT | ships: notices, LGPL/MPL source |
| onnxruntime:cpu | [lemonade-sdk/ort-server](https://github.com/lemonade-sdk/ort-server) | 0.3.7 | Apache-2.0, with ONNX Runtime MIT and the third-party licenses in its archive | ships: license notices |
| acestep:vulkan | [pwilkin/acestep.cpp](https://github.com/pwilkin/acestep.cpp) | v0.1.1 | MIT (the acestep.cpp authors); ggml MIT | ships: license notice |
| trellis:vulkan | [pwilkin/trellis.cpp](https://github.com/pwilkin/trellis.cpp) | v0.4.3 | MIT (Piotr Wilkin), added after the tag "as intended" (commit 6409621c); ggml MIT | ships: license notice |
| openmoss:vulkan | [pwilkin/openmoss](https://github.com/pwilkin/openmoss) | v0.3.0 | Apache-2.0; ggml and llama.cpp MIT | ships: license notice |
| thinksound:vulkan | [pwilkin/thinksound.cpp](https://github.com/pwilkin/thinksound.cpp) | v0.1.2 | none: no LICENSE in the repository at the tag or on main | **not shipped** |
| flm:npu | [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM) | v1.0.6 | CLI MIT (AMD); NPU binaries closed, see below; bundled FFTW 3.3.10 GPL-2.0-or-later, XRT 2.25 Apache-2.0, Boost 1.83 BSL-1.0, zlib | **not shipped** |

Not shipped, and why:

- **thinksound** has no license. Without one, copying its binary is not permitted. It can ship
  once the author adds a license.
- **flm** (FastFlowLM's runtime) ships 26 prebuilt NPU libraries (`lib/lib*_npu.so`,
  `libq4_npu_eXpress.so`, ...) and 231 xclbins that have no source in the repository. At the
  v1.0.6 tag its README says these binaries are "completely free for any use, including commercial
  use", while `TERMS.md` at the same tag still calls them proprietary, patent-pending and free only
  below USD 10M annual revenue, and neither text grants redistribution in so many words. Its
  `flm-real` links those closed libraries and FFTW (GPL-2.0-or-later) into one program, which the
  GPL does not allow to be distributed. The engine's NOTICE also says the engine does not ship
  FastFlowLM's kernel binaries. Lemonade still downloads flm on the stick when a user asks for it.

`LEMONADE_BACKENDS` can still name them, but mkimage.sh refuses them until their `SOURCE` file
says `Ship: yes`.

## Lemonade and the engine

| Component | Release | License | Source |
|---|---|---|---|
| 1bit engine (`1bit`, its HRX llama-server) | this repository | Apache-2.0; third-party parts in [NOTICE](../NOTICE) | https://github.com/1bit-MONSTER/engine |
| Lemonade (lemond, the lemonade CLI, the web app) | third_party/lemonade pin | Apache-2.0, Advanced Micro Devices, Inc.; web app bundle licenses in `renderer.bundle.js.LICENSE.txt` | https://github.com/1bit-MONSTER/lemonade |
| XRT with the XDNA plugin (`/opt/xilinx/xrt`) | the builder's XRT | Apache-2.0 | https://github.com/Xilinx/XRT, https://github.com/amd/xdna-driver |
| ROCr HSA runtime (TheRock, for HRX) | the build's ONEBIT_HRX_LIBHSA | University of Illinois/NCSA | https://github.com/ROCm/ROCR-Runtime, https://github.com/ROCm/TheRock |

## System software from Ubuntu

mkimage.sh copies these from the build machine (Ubuntu 26.04), unmodified, and records the exact
package versions in `/usr/share/licenses/1bit-os/packages.txt`. The versions below are the ones on
the machine this list was made on.

| Component | Ubuntu package | Version | License |
|---|---|---|---|
| Mesa Radeon Vulkan driver (RADV) | mesa-vulkan-drivers | 26.0.8-1ubuntu0.3 | MIT (some files BSD, Khronos, SGI) |
| LLVM (libLLVM, used by RADV) | libllvm21 | 21.1.8-6ubuntu1 | Apache-2.0 WITH LLVM-exception |
| Vulkan loader | libvulkan1 | 1.4.341.0-1 | Apache-2.0 |
| BusyBox | busybox-static | 1.37.0-7ubuntu1 | GPL-2.0-only |
| Dropbear SSH, libtomcrypt, libtommath | dropbear-bin, libtomcrypt1, libtommath1 | the Ubuntu archive's | MIT; public domain / WTFPL; Unlicense |
| GNU C library | libc6 | 2.43-2ubuntu2.4 | LGPL-2.1-or-later |
| GCC runtime (libstdc++, libgcc_s, libgomp) | libstdc++6, libgcc-s1, libgomp1 | 16-20260322-1ubuntu1 | GPL-3.0-or-later WITH GCC-exception-3.1 |
| libquadmath | libquadmath0 | 16-20260322-1ubuntu1 | LGPL-2.1-or-later |
| GNU Readline (pulled in for moonshine's Python) | libreadline8t64 | 8.3-4 | GPL-3.0-or-later |
| GnuTLS, Nettle, GMP, libidn2, libunistring, libtasn1 (curl's TLS stack) | libgnutls30t64, libnettle8t64, libhogweed6t64, libgmp10, libidn2-0, libunistring5, libtasn1-6 | as in packages.txt | LGPL-2.1+ / LGPL-3.0+ or GPL-2.0+ |
| elfutils libelf | libelf1t64 | 0.194-4 | LGPL-3.0-or-later or GPL-2.0-or-later |
| librtmp, keyutils, libsystemd, libcrypt | librtmp1, libkeyutils1, libsystemd0, libcrypt1 | as in packages.txt | LGPL-2.1-or-later |
| everything else (OpenSSL, zlib, zstd, xz, bzip2, curl, Kerberos, OpenLDAP, Cyrus SASL, libdrm, expat, libxml2, libffi, PCRE2, protobuf, SQLite, libwebsockets, ncurses, libedit, libbsd, libmd, brotli, nghttp2, libssh2, libpsl, p11-kit, util-linux libuuid, libcap, Wayland, X11 xcb libraries, libdisplay-info) | as in packages.txt | as in packages.txt | permissive (Apache-2.0, MIT, BSD, zlib, public domain and similar); each package's copyright file is on the image |
| Linux kernel and modules | the build's linux-image (scripts/build-kernel.sh) | third_party/linux pin | GPL-2.0-only WITH Linux-syscall-note |
| Firmware: amdgpu, amdnpu, MediaTek MT7925, Realtek RTL8125, wireless-regdb | linux-firmware-amd-graphics, linux-firmware-amd-misc, linux-firmware-mediatek, linux-firmware-realtek, wireless-regdb | as in packages.txt | redistributable binary-only licenses (LICENSE.amdgpu, LICENSE.amdnpu, LICENCE.mediatek, the r8169 notice) and ISC (regulatory.db), in each package's copyright file |
| Mozilla CA certificates | ca-certificates | as in packages.txt | MPL-2.0 |

## Source for the GPL, LGPL and MPL components

The GPL and LGPL require that whoever distributes these binaries also makes their corresponding
source available; the MPL requires the same for its covered files. For each:

| Component | Corresponding source |
|---|---|
| Linux kernel | https://github.com/torvalds/linux at the commit third_party/linux pins, built with scripts/build-kernel.sh |
| BusyBox, glibc, GCC runtime, Readline, GnuTLS, Nettle, GMP, libidn2, libunistring, libtasn1, elfutils, librtmp, keyutils, systemd, libxcrypt, ca-certificates | the Ubuntu source packages at the versions in packages.txt (`apt-get source <package>=<version>`, https://launchpad.net/ubuntu/+source/<package>) |
| eSpeak NG in kokoro's `koko`, and Kokoros | https://crates.io/api/v1/crates/espeak-rs-sys/0.1.9/download (eSpeak NG as built into koko); https://github.com/lemonade-sdk/Kokoros/tree/b17 |
| TheRock sysdeps in llamacpp-hrx (elfutils, libnl, numactl, libmnl) | the upstream tarballs TheRock builds, https://github.com/ROCm/TheRock/tree/main/third-party/sysdeps/linux (elfutils 0.195, libnl 3.12.0, numactl 2.0.19, libmnl 1.0.5 at the time of this review) |
| libquadmath, libgfortran in moonshine | GCC, https://gcc.gnu.org |
| certifi, tqdm in moonshine (MPL-2.0) | the covered files are on the image as shipped (Python source and cacert.pem); https://pypi.org/project/certifi/, https://pypi.org/project/tqdm/ |
