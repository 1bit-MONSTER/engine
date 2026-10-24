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

Lemonade on a USB stick, with the 1bit engine inside. 1bit OS boots a Strix Halo machine from
the stick, runs entirely in RAM, never touches the internal disk, and starts
[Lemonade](https://github.com/lemonade-sdk/lemonade)'s server, `lemond`, with Lemonade's own
defaults. Lemonade leads: its web app, its API, its model catalog and downloads, its routing and its
settings. The engine is one of Lemonade's backends, the `onebit` recipe (HRX0 and the NPU,
[docs/lemonade.md](../docs/lemonade.md)), next to Lemonade's own (llama.cpp, whisper.cpp,
stable-diffusion.cpp, Kokoro and the rest).

What is in it: the kernel with the GPU (`amdgpu`, which also gives HSA its `/dev/kfd`) and NPU
(`amdxdna`) drivers, the firmware those chips load, BusyBox, the NPU runtime compiled into `1bit`,
Mesa's Radeon Vulkan driver (for Lemonade's Vulkan backends), Lemonade (`lemond`, the `lemonade` CLI
and the web app, built from the engine's pin `third_party/lemonade`), and the engine: `1bit` with its
HRX backend (the `llama-server` it runs and the HSA runtime that `llama-server` loads are on the data
partition - `1bit-llama/` below - because the initramfs has to stay under the size grub will load).

Everything that has to last lives on the stick's `1BIT-DATA` partition:

| Path on `1BIT-DATA` | What |
|---|---|
| `models/` | your GGUF files and NPU model directories |
| `lemonade/config/` | Lemonade's config directory: `config.json`, `user_models.json`, `recipe_options.json` |
| `lemonade/lemond.conf` | Lemonade's environment file: `LEMONADE_API_KEY`, `LEMONADE_ADMIN_API_KEY`, `HF_TOKEN` |
| `lemonade/cache/` | Lemonade's cache directory, with its backends in `bin/`, installed when the image was built |
| `lemonade/licenses/` | the backends' licenses, one directory per recipe ([LICENSES.md](LICENSES.md)) |
| `lemonade/huggingface/` | models Lemonade downloads (`models_dir`) |
| `1bit-llama/` | the engine's own HRX `llama-server` and the libraries it links, put on the stick instead of in the RAM image; `init` links `/opt/1bit-llama` to it at boot, so `1bit serve` finds it on `PATH` whether the boot is LIVE or INSTALLED |
| `cache/hrx-jit/` | HRX's Loom kernel cache (`GGML_HRX_JIT_CACHE_DIR`): the first use of a kernel compiles it, later boots reuse it |
| `ssh/` | `authorized_keys` and the SSH host key |

## Build it (a few minutes, no root)

On a machine that has built the engine with the NPU lane and HRX (TheRock in `/opt/rocm-therock`):

```sh
# the private components the release compiles in (docs/releases.md, "Private components")
npurt=$(scripts/fetch-private.sh config/npu-runtime.config ~/.cache/npu-runtime)
scripts/fetch-private.sh config/gpu-kernels.config ~/.cache/gpu-kernels
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DONEBIT_NPU=ON -DONEBIT_NPU_RT_DIR="$npurt" \
      -DONEBIT_HRX=ON -DONEBIT_GPU_PRIVATE="$HOME/.cache/gpu-kernels"
cmake --build build
os/mini/mkimage.sh out build
```

`mkimage.sh` puts the HRX `llama-server` and every library it links on the data partition under
`1bit-llama/`, with each ELF's `RUNPATH` rewritten to `$ORIGIN` (`patchelf`) so the set runs from there -
the build tree's paths do not exist on a stick. The HSA runtime HRX loads is taken from the Lemonade
`llamacpp-hrx` package that is already on the stick, and `init` writes its on-stick path into
`/etc/1bit/libhsa`, which is where the engine reads `IREE_HAL_AMDGPU_LIBHSA_PATH` from. It takes `lemond` from `build/lemonade` (or `LEMONADE=`, a
`scripts/build-lemonade.sh` prefix), and when there is none it builds it there from
`third_party/lemonade` with the web app (`LEMONADE_WEB_APP=1`, which needs `node` and `npm`).

**Offline first.** Lemonade downloads a backend the first time a model needs it. `mkimage.sh` does
that ahead of time instead: it runs `lemond` on the build machine and asks Lemonade's own installer
(`POST /api/v1/install`) for each backend in `LEMONADE_BACKENDS`, then copies Lemonade's `bin/`
directory into `lemonade/cache/` on the stick, at the paths and versions Lemonade expects. The
system libraries those backends link go into the image. The default list is every backend Lemonade
offers on Linux x86_64 for Strix Halo that needs no ROCm runtime and whose license allows it on the
stick:

`llamacpp:vulkan llamacpp:cpu llamacpp-hrx:hrx whispercpp:vulkan whispercpp:cpu sd-cpp:vulkan
sd-cpp:cpu kokoro:cpu moonshine:cpu onnxruntime:cpu acestep:vulkan trellis:vulkan openmoss:vulkan`

`thinksound:vulkan` (no license) and `flm:npu` (closed NPU binaries linked with GPL FFTW) are left
to Lemonade's download on the stick. [LICENSES.md](LICENSES.md) lists every third-party piece of
the image, its license and its source; a backend ships only when `os/licenses/<recipe>/SOURCE`
says `Ship: yes`, and its license texts go with it.

The installs are kept in `~/.cache/1bit-os/lemonade-backends/`, one set per Lemonade pin, so the
next build downloads nothing. `LEMONADE_BACKENDS=` (empty) leaves them all to Lemonade's download.

**Network only: the ROCm backends.** `llamacpp:rocm`, `whispercpp:rocm`, `sd-cpp:rocm`,
`acestep:rocm`, `thinksound:rocm`, `trellis:rocm`, `thenoise:rocm`, `vllm:rocm` and `ds4:rocm` run
on Lemonade's ROCm runtime, which Lemonade installs per GPU architecture (`gfx1151` on Strix Halo,
about 3 GB) as a Python environment with absolute paths. It cannot be installed on another machine
and copied, so on a stick with a network Lemonade downloads these on first use as it does
everywhere; without one they are unavailable. So are models that are not on the stick.

`mkimage.sh` takes the running kernel (or `KVER=`, with `VMLINUZ=` for an image outside `/boot`),
copies only the modules, firmware and shared libraries the stack uses. The engine no
longer builds a kernel: a Strix Halo kernel with `amdxdna` comes from
our separate kernel build, installed on the build machine
(or unpacked from its packages for `KVER=`/`VMLINUZ=`). It writes:

| File | What |
|---|---|
| `out/1bit-os.efi` | the OS in one EFI file (kernel, command line, initramfs), about 150 MB |
| `out/1bit-os.img` | a USB image: an EFI partition that boots `1bit-os.efi`, and `1BIT-DATA` with Lemonade's backends (about 1.4 GB) |

It needs `systemd-boot-efi` (the EFI stub) and `mtools`; unpack both without installing:
`cd ~/.cache/1bit-os/tools && apt-get download systemd-boot-efi mtools && for d in *.deb; do dpkg -x $d x; done`.

## Put it on a stick and boot

```sh
sudo dd if=out/1bit-os.img of=/dev/sdX bs=4M conv=fsync   # the whole stick is overwritten
sudo growpart /dev/sdX 2 && sudo resize2fs /dev/sdX2         # 1BIT-DATA fills the stick
```

Copy models into `models/` on `1BIT-DATA`, then boot the Strix Halo machine from the stick
(its firmware's boot menu, Secure Boot off). It comes up with a shell, DHCP on Ethernet, and
Lemonade on `http://localhost:13305`: its web app, its OpenAI-compatible API under `/api/v1`
(`/v1` too), and `lemonade` on the shell.

**Models.** Lemonade lists every GGUF in `models/` itself (`extra_models_dir` in its
`config.json`), for its `llamacpp` recipe. At boot, 1bit OS also registers each GGUF there (not
`mmproj` files; a split model by its first part) and each NPU model directory for the `onebit`
recipe, as `user.<name>-1bit` in Lemonade's model registry (`lemonade/config/user_models.json`;
entries it already has are left alone). Lemonade shows those only where one of the recipe's
backends runs (gfx1151 for HRX, an XDNA2 NPU), and decides everything else: which backend, context
size, which models stay loaded. Models pulled from Lemonade's catalog (`lemonade pull`, or the web
app) are stored in `lemonade/huggingface/` and survive a reboot.

**Network exposure and the API key are Lemonade's settings.** Lemonade listens on `localhost`, so
from another machine reach it over SSH (`ssh -L 13305:localhost:13305 root@<IP>`). To open it to
the network, set Lemonade's own `"host"` in `lemonade/config/config.json` (`"0.0.0.0"`, or
`lemonade config set host=0.0.0.0` on the running server) and, first, its API key in
`lemonade/lemond.conf`:

```sh
LEMONADE_API_KEY=<a long random string>      # every /api, /v0 and /v1 request must send it as a Bearer token
#LEMONADE_ADMIN_API_KEY=<another one>        # optional: a separate key for /internal (settings, shutdown)
```

`lemond.conf` is the file Lemonade's systemd unit reads as `/etc/default/lemond`; init reads it
for `lemond` only. The rest of `config.json` follows Lemonade's documentation
(`third_party/lemonade/docs/guide/configuration/README.md`); 1bit OS sets only `extra_models_dir`
and `models_dir`. `/data/1bit.conf`, which earlier images read to start `1bit serve` at boot, is
no longer read.

**The engine alone.** `1bit serve` is on the shell for debugging, as on any machine:

```sh
1bit serve -m /data/models/<file.gguf>                    # --device auto: HRX0, on 127.0.0.1:8000
```

It has no authentication of its own (SECURITY.md); keep it on 127.0.0.1.

## Check the release

One command says PASS or FAIL for the image that is booted, on the live stick and again after the
install to disk. Every check prints its own PASS/FAIL/SKIP line, a transcript lands in
`/data/release-check.log`, and the exit status is non-zero on any failure:

```sh
# on the live stick (console, or over SSH from another machine)
1bit-release-check

# after installing to the disk and booting from it
1bit-release-check
```

It checks the boot mode (LIVE or INSTALLED), the kernel and the Secure Boot state, the GPU and NPU
devices, that Lemonade answers with its API key and refuses without it, that `/api/v1/models` lists
the onebit entries the build shipped (22 on HRX, plus the NPU entries whose signed kernel set is on
the data partition) and the stick's own GGUF, one chat per backend the machine can run, one
`lemonade bench` chat-short row on the default model, that every file under `/data/npu/kernels` is
listed in its set's signed manifest and matches its sha256, that `LICENSE-BINARY`/`NOTICE-BINARY`
are present, and the image's tag + sha256.
The rows that need the hardware - `gpu`, `npu`, `models`, `chat-hrx`, `chat-npu` - report
`SKIP(reason)` in a VM.

## Install it on the machine's own disk

The stick is the installer. Boot it as above, then from its shell (console or SSH):

```sh
1bit-selftest                       # the live boot works: RESULT: PASS, and the NPU where there is one
1bit-install /dev/nvme0n1 --yes     # ONE internal disk, named explicitly; this destroys it
```

`1bit-install` lays the disk out exactly as the stick is laid out, with the same tool the image is
built with: a 512 MB EFI system partition carrying the signed chain this boot came from (shim, grub,
`vmlinuz`, `initramfs`, `grub.cfg` - byte for byte, so the machine's MOK enrolment needs no change),
and a `1BIT-DATA` ext4 partition filling the rest, carrying `models/`, `lemonade/` (its backends,
config, cache and the API key), the NPU kernel sets and `ssh/` from the stick's `/data`.

It refuses, before writing anything: to run without `--yes`; to be given a partition instead of a
whole disk; to point at a removable device; to point at the disk this live boot is running from; to
point at a disk with anything mounted on it. Then it prints the disk's model, size and current
partition table - what it is about to destroy - and waits 10 seconds before the first write.

Power off, remove the stick, boot the machine from the disk. It boots the same init, which finds
`/data/.installed` and says so:

```
1bit OS  v2026.43  -  INSTALLED on this machine. /data is on its disk; ...
MODE: INSTALLED - 1bit OS runs from this machine's disk; /data is /dev/nvme0n1p2.
```

and `/etc/1bit/installed` exists instead of `/etc/1bit/live` (the menu a stick shows is skipped -
there is nothing to choose). Then `1bit-release-check` again: the same PASS/FAIL list as on the stick,
now reporting the INSTALLED mode. `1bit-selftest` works too and must report `mode : INSTALLED`,
`RESULT: PASS` and, on hardware with an NPU, `RESULT-NPU: PASS` with the placement line in
`/data/selftest.log`.

**Secure Boot.** The install copies the ESP as it is, so the kernel and initramfs on the disk are the
same MOK-signed files the stick booted; the machine's MOK enrolment (`mokutil --list-enrolled`) is
not touched by an install. If the firmware does not offer the disk on its own, add an entry pointing
at `\EFI\BOOT\BOOTX64.EFI` on its partition 1 (on a Linux box: `efibootmgr -c -d /dev/nvme0n1
-p 1 -L "1bit OS" -l '\EFI\BOOT\BOOTX64.EFI'`). Booting the stick again is unchanged and does not
touch the installed disk: that is how you get back to a live system (or to reinstall).

What the VM proves (`os/mini/vmtest.sh`, which now also installs onto a virtual NVMe and boots from
it with no stick attached): the installer's refusals, the install itself, the INSTALLED banner and
`/etc/1bit/installed`, `/data` coming from the installed disk, Lemonade up with its key and answering
a chat, and `1bit-selftest` reporting INSTALLED. What only the real machine can prove: that this
firmware boots the disk on its own, and the NPU on it.

## Check it without the hardware

```sh
qemu-system-x86_64 -enable-kvm -cpu host -m 4G -kernel /boot/vmlinuz-$(uname -r) \
  -initrd out/initramfs.img -append "console=ttyS0 rdinit=/init 1bit.check" -nographic -no-reboot
```

`1bit.check` makes init check that the engine, `llama-server`, Lemonade, the HSA runtime and the
NPU firmware are in place and start, that the image's runtime dependencies are self-contained, print
`1BIT-OS CHECK PASS` or `FAIL`, and power off.

`os/mini/vmtest.sh out build [model.gguf]` boots the USB image in QEMU/KVM with UEFI, installs it
onto an empty virtual NVMe with `1bit-install` (checking every refusal first), boots from that
disk with no stick attached, and checks all of it; the network is QEMU's restricted one (nothing
outside the VM is reachable, so the stick has to work offline), with
Lemonade's `"host": "0.0.0.0"` and a `LEMONADE_API_KEY` on the stick. It checks over SSH and HTTP
that Lemonade's web app loads, that Lemonade refuses a request without the key, that
`/api/v1/models` lists the stick's model, and that Lemonade answers a chat request with it. The VM
has no Radeon GPU or NPU: Lemonade serves that chat with its own `llamacpp` backend on the CPU, and
hides the `onebit` entry (its backends need gfx1151 or the NPU), so the test checks only that
the entry is in Lemonade's registry on the stick. On a machine without TheRock,
`LLAMA_SERVER=<a CPU llama-server>` makes a CPU-only test image.

What only the hardware can check: `amdgpu` bringing up `/dev/kfd` and HRX0 from the image (TheRock's
HSA runtime and its libraries complete, the PM4-emulation probe answered), Lemonade offering and
running `onebit` models on HRX0 and the NPU, HRX kernels compiling into `/data/cache/hrx-jit` and
being reused after a reboot, Lemonade's Vulkan backends on RADV, `llamacpp-hrx`, Lemonade's ROCm downloads for gfx1151, the Wi-Fi chip, and the speed of all of it.

## Next

- A Loom kernel cache built ahead of time, so the first boot needs no compile.
- The image split: a base (the EFI file and an empty data partition) and Lemonade's backends as a
  separate bundle for the data partition.
- An install that can also grow `1BIT-DATA` over a disk whose firmware needs a partition table it did
  not write, and an upgrade path that keeps `/data` while replacing the ESP.
- Every piece pinned upstream and bumped by a workflow, like the rest of the engine.
