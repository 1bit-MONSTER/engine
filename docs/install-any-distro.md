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

# Install the 1bit engine on any Linux, with Lemonade from its own package

The engine does not need a patched Lemonade. A **stock** Lemonade discovers it through the
`llamacpp-hrx` recipe it already has: that recipe resolves the name `llama-server` inside the
directory named by its `hrx_bin` setting, so pointing `hrx_bin` at the engine package's
`lemonade-bin/` directory is the whole plug-in contract. Nothing is added upstream and nothing is
patched locally.

Two packages, from two places:

| piece | where it comes from | what it is |
|---|---|---|
| Lemonade server + CLI | Lemonade's own package (Ubuntu: its PPA, weekly) | `lemonade-server`, `lemond` |
| the engine | `1bit-engine_<version>_amd64.deb`, or the `.tar.zst` on a non-deb distro | `1bit`, its HRX llama.cpp build (`hrx/llama/bin/llama-server`), the sealed NPU kernel sets, the HSA runtime it dlopens, and the engine's `lemonade-hrx` wrapper, which is what the recipe's `llama-server` name points at |

The engine package is userspace only: no kernel module, no Python at runtime.

One indirection carries the whole contract, and it matters: Lemonade's recipe hands a backend exactly one
environment variable (`GGML_DISABLE_VULKAN=1`), so pointing `hrx_bin` at the engine's raw `llama-server`
would start the HRX kernels without the environment `1bit serve` gives them. `lemonade-bin/llama-server`
is therefore the engine's own `lemonade-hrx` wrapper: it asks `1bit serve --print-backend-env` for that
environment and for serve's architecture and memory checks, then `exec`s the engine on Lemonade's argv,
filling in serve's defaults for what the argv left out (`-ngl 99`, `--cache-reuse 256`, `-kvu` above one
slot).

## Ubuntu 26.04 (resolute)

**The engine package needs glibc 2.43, so it is a 26.04-or-newer package.** Measured on the shipping
package (`readelf -V` on the extracted `1bit`, 2026-11-05): its highest requirement is
`GLIBC_2.43`, and it is **one symbol** — `sqrtf@GLIBC_2.43`. On Ubuntu 24.04 (noble, glibc 2.39) the
deb installs and `ldd` is happy, and then the binary refuses to load with
`version 'GLIBC_2.43' not found`. Lemonade's own PPA does publish noble builds, so the engine half is
the one that pins the distribution: on noble, build `1bit` from source, or run it on a 26.04 box.
The tarball below carries the same binary and therefore the same floor.

```bash
# 1. Lemonade's own package. The archive is signed by key
#    881CF4B4 0B0BFA28 8D6776E8 3BF36CFA 0BD50AEC  (~lemonade-team).
sudo install -d /etc/apt/keyrings
curl -fsSL 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x881CF4B40B0BFA288D6776E83BF36CFA0BD50AEC' \
  | sudo gpg --dearmor -o /etc/apt/keyrings/lemonade.gpg
echo "deb [signed-by=/etc/apt/keyrings/lemonade.gpg] https://ppa.launchpadcontent.net/lemonade-team/stable/ubuntu $(. /etc/os-release; echo $VERSION_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/lemonade.list
sudo apt-get update
sudo apt-get install -y lemonade-server

# 2. the engine
sudo apt-get install -y ./1bit-engine_<version>_amd64.deb

# 3. point Lemonade at it. The package already ships this drop-in as
#    /usr/lib/systemd/system/lemond.service.d/10-1bit-engine.conf, so on a stock systemd install there is
#    nothing to do here; a local override is only for a different path. Both forms are shipped, and
#    /usr/share/1bit/lemonade-config-fragment.json is the config-key form.
sudo mkdir -p /etc/systemd/system/lemond.service.d
sudo tee /etc/systemd/system/lemond.service.d/10-1bit-engine.conf >/dev/null <<'EOF'
[Service]
Environment=LEMONADE_HRX_HRX_BIN=/opt/1bit/engine/lemonade-bin
EOF
#   ... or, without systemd:  {"hrx": {"hrx_bin": "/opt/1bit/engine/lemonade-bin"}}
#                             in Lemonade's config.json
sudo systemctl daemon-reload && sudo systemctl restart lemond
```

**Do not install `amdxdna-dkms` from that archive.** It is in the same PPA but nothing pulls it, and it
builds AMD's *stock* `amdxdna` for the running kernel. Our boards reset with the stock driver (see
`scripts/mkimage.sh`), which is why every 1bit OS image and this box run a patched build. If you want
the driver on another distribution, take our package instead: it is the same driver with the fleet's
patch series, and the next section installs it.

Verify the plug-in (no model needed):

```bash
curl -s localhost:13305/api/v1/system-info | grep -A2 '"llamacpp-hrx"'
#   -> the "hrx" backend should read state=<installed>; on a machine without an AMD GPU it reads
#      "unsupported: Unsupported GPU" before it ever looks at the binary, which is expected.
```

Then one real chat per device is in "Verify" below.

## 1bit OS (Puppy appliance)

Nothing to do: the appliance already ships the engine and Lemonade. This page is the recipe for
everything else, and the appliance's own check is `1bit-release-check` / `puppy-check`.

## Other distributions

Same floor as above: the tarball carries the same `1bit`, so the distribution needs glibc 2.43 or
newer as well (measured: one symbol, `sqrtf@GLIBC_2.43`). Unpack it at `/opt/1bit/engine` and install
Lemonade your distro's way (its `.deb`/`.rpm`, its container, or a source build — anything that reports
`llamacpp-hrx` in `system-info`):

```bash
sudo tar -C / --zstd -xf 1bit-engine-<version>-linux-x86_64.tar.zst opt
sudo ln -sf /opt/1bit/engine/1bit /usr/local/bin/1bit-engine
# then the same two lines as above, in whatever runs lemond: the env var, or the config key.
```

If your Lemonade runs under systemd, the drop-in from the deb applies verbatim; in a container or a
hand-started `lemond`, export `LEMONADE_HRX_HRX_BIN=/opt/1bit/engine/lemonade-bin` in its environment.

## Kernel and NPU driver packages (optional)

The engine package is userspace only. The NPU and the board's ECU need three more pieces, and a machine
that stays on its distribution's kernel only needs the second and third:

| package | what it is | needed for |
|---|---|---|
| `linux-image-<rel>` / `linux-headers-<rel>` (deb), `1bit-kernel` + `1bit-kernel-devel` (rpm), `linux-1bit` + `linux-1bit-headers` (Arch) | the kernel 1bit OS boots: Linux 7.4-rc1 with this fleet's config, signed with the 1bit MOK | only if you want OUR kernel; the drivers build against yours as well |
| `amdxdna-1bit-dkms` | `amd/xdna-driver` at `526c6081` plus the engine's patch series (`third_party/patches/amdxdna/0001-0005`), as DKMS source | the NPU. Any 6.10+ kernel, signed with **your** MOK |
| `ec-su-axb35-dkms` | the AXB35-02 board's embedded-controller driver (fans, EC temperature, power mode), GPL-2.0, upstream `cmetz/ec-su_axb35-linux` at `f62c2c22`, plus a DMI gate so it attaches only on an AXB35 | the fans on a Bosgame AXB35; inert on any other board |
| `1bit-npu-config` | `limits.d/99-amdnpu-memlock.conf` (unlimited memlock), `udev/rules.d/90-1bit-npu-power.rules` (NPU keep-awake), `modules-load.d/ec_su_axb35.conf`, and the public MOK certificate at `/usr/share/1bit/mok/MOK-Kernel.der` | the NPU runtime's locked command buffers, and engine#1051's suspend/resume hang |

The three driver/config packages are the same set in all three families. They install to
`/lib/modules/<ver>/updates/` (Debian, Arch) or `/extra/` (Fedora) through dkms, which is in front of
the in-tree `amdxdna`, and uninstalling the package puts the in-tree driver back.

```bash
# Debian / Ubuntu: the kernel comes from 1bit OS's own build, the drivers build against your headers.
sudo apt-get install -y ./amdxdna-1bit-dkms_<ver>_all.deb ./ec-su-axb35-dkms_<ver>_all.deb ./1bit-npu-config_<ver>_all.deb
sudo apt-get install -y ./linux-image-7.4.0-rc1-1bit_<ver>_amd64.deb ./linux-headers-7.4.0-rc1-1bit_<ver>_amd64.deb  # optional

# Fedora / openSUSE: the local files are unsigned until the release pipeline signs them, hence --nogpgcheck.
sudo dnf install --nogpgcheck ./amdxdna-1bit-dkms-<ver>.x86_64.rpm ./ec-su-axb35-dkms-<ver>.x86_64.rpm ./1bit-npu-config-<ver>.x86_64.rpm
sudo dnf install --nogpgcheck ./1bit-kernel-<ver>.x86_64.rpm ./1bit-kernel-devel-<ver>.x86_64.rpm   # optional
sudo depmod -a 7.4.0-rc1-1bit && sudo dracut --kver 7.4.0-rc1-1bit   # if you installed the kernel
# openSUSE: zypper install --allow-unsigned-rpm ./<file>.rpm, same files

# Arch: pacman warns that a local package is unsigned and installs it.
sudo pacman -U ./amdxdna-1bit-dkms-<ver>-x86_64.pkg.tar.zst ./ec-su-axb35-dkms-<ver>-x86_64.pkg.tar.zst ./1bit-npu-config-<ver>-x86_64.pkg.tar.zst
sudo pacman -U ./linux-1bit-<ver>-x86_64.pkg.tar.zst ./linux-1bit-headers-<ver>-x86_64.pkg.tar.zst   # optional
sudo mkinitcpio -P                                                  # if you installed the kernel
```

Every one of them needs the kernel's own module tree next to the headers (`linux-modules-<ver>` on
Ubuntu, `kernel-core` on Fedora, `linux` on Arch) — dkms looks for
`/lib/modules/<ver>/build`, which those packages provide and a headers-only container does not.

**Secure Boot.** Our kernel and its modules are signed with the `1bit-MONSTER` kernel key, which no
machine has enrolled. Enrol the public half once, from the config package:

```bash
sudo mokutil --test-key /usr/share/1bit/mok/MOK-Kernel.der          # "not enrolled" on a new machine
openssl x509 -inform DER -in /usr/share/1bit/mok/MOK-Kernel.der -noout -subject
#   subject=CN=strixhalo Secure Boot Kernel, O=1bit-MONSTER
sudo mokutil --import /usr/share/1bit/mok/MOK-Kernel.der            # sets a password; reboot -> MokManager
```

The **private** half never leaves the machine that signs the release and is part of no package. The
dkms packages are the other way round: dkms signs with *your* key (`/var/lib/shim-signed/mok` on
Debian/Ubuntu, `/etc/dkms/` elsewhere), so a module built on your machine chains to the MOK you
enrolled for it — on Fedora and Arch, dkms generates that key pair on first use.

Verify (no model needed):

```bash
dkms status                                  # amdxdna-1bit/0.18.0 and ec-su-axb35/0.1, installed for your kernel
modinfo -n amdxdna                           # .../updates/dkms/amdxdna.ko (Debian, Arch) or .../extra/amdxdna.ko.xz (Fedora)
modinfo -F srcversion amdxdna                # EE1F36A2B9BA294CE11AD80 is the series this release ships
cat /etc/security/limits.d/99-amdnpu-memlock.conf
```

Uninstall puts the in-tree driver back:

```bash
sudo apt-get remove amdxdna-1bit-dkms ec-su-axb35-dkms 1bit-npu-config   # or dnf remove / pacman -R
```

**Source.** Each driver package ships the source it builds (`/usr/src/amdxdna-1bit-0.18.0/`,
`/usr/src/ec-su-axb35-0.1/`) with its LICENSE; the amdxdna patch series is
`third_party/patches/amdxdna/` in the engine repository, and 0006/0007 are carried there but **not
applied** (loading those two is a separate decision — engine#812). The kernel packages carry
`/boot/config-7.4.0-rc1-1bit`, and the tree they were built from is `torvalds/linux` v7.4-rc1 plus that
config, built by `scripts/build-kernel.sh` / `os/mini/mkimage.sh`.

The packages are built by `scripts/mk-kernel-packages.sh` and proven per family by
`scripts/pkg-proof.sh`, which installs them in a clean container of that family, builds both dkms
modules against the container's own kernel headers, checks that `amdxdna` resolves to the dkms build
rather than the in-tree one, and uninstalls everything again.

## Host prerequisites, per device (measured on the Strix Halo development box)

**GPU / HRX.** No ROCm install, no HIP user-space: the package carries the HSA runtime the HRX
backend dlopens, and it is found because the recipe's libraries and ours share `$ORIGIN`.

* `amdgpu` loaded, with `/dev/kfd` and `/dev/dri/renderD*` both `root:render`, mode `0660`; the user
  running `lemond` must be in the `render` group (a container needs `--group-add <render gid>`, and
  that group must exist inside the image or the GID is dropped).
* `amdgpu.cwsr_enable=0` is what the shipping hardware reports. In that configuration the engine
  sets `HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1`; without it the kernels take
  the ROCr SVM path and a nightly `libhsa` page-faults (engine#715). Those come from the engine's own
  wrapper (`1bit serve --print-backend-env`), so Lemonade's child gets them even though Lemonade
  itself passes only `GGML_DISABLE_VULKAN=1`.
* Kernels: an HRX build needs a KFD that supports the device (gfx1151 here on
  `7.3.0-rc4-1bit-perfopt+`); it does not need a patched kernel.

**NPU.** `amdxdna` bound to the device, plus the board firmware files under
`/lib/firmware/amdnpu/` (`17f0_10`, `17f0_11` on Strix Halo) and `memlock unlimited` — the NPU
runtime is compiled into `1bit` and locks command buffers.

* Which `amdxdna` matters. The in-tree driver from your distro's kernel is the default; this board
  resets the machine with the stock module and is shipped with a private build of the same vermagic
  dropped over it (`scripts/mkimage.sh`). On a distro kernel, use that kernel's own `amdxdna` first
  and only replace it with a build for the exact kernel you run.
* Secure Boot: the engine package contains no module, so it is unaffected. An out-of-tree `amdxdna`
  replaces a module the kernel trusts, so it must be signed by an enrolled MOK
  (`/var/lib/shim-signed/mok/MOK.priv` + `.der` on 1bit OS); the kernel side of that is
  `CONFIG_LOCK_DOWN_IN_EFI_SECURE_BOOT` in the distro kernel, which the 1bit OS kernel leaves unset.

## Verify

One chat per device, through Lemonade, on a model the recipe knows (`*-HRX` in Lemonade's catalog):

```bash
# GPU
curl -s -X POST localhost:13305/api/v1/load -H 'Content-Type: application/json' \
  -d '{"model_name":"Qwen3-4B-Instruct-2507-HRX"}'
curl -s -X POST localhost:13305/api/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-4B-Instruct-2507-HRX","messages":[{"role":"user","content":"2+3?"}],"max_tokens":32}'
```

The engine alone, which is the fastest way to separate "Lemonade" from "the engine" when something
is wrong:

```bash
1bit-engine serve -m <model>.gguf --device hrx          # GPU, prints the HRX0 backend it starts
1bit-engine serve -m <model>.gguf --device npu          # NPU
```

What this route does **not** do, compared with the fork's `onebit` recipe: the model list is Lemonade's
own catalog, not our 27 rows — ours can be added on the side through the config directory's
`user_models.json`, below — and there are no NPU catalog rows, because Lemonade has no recipe for our
NPU route at all (NPU stays on the engine: `1bit serve --device npu`). The engine's own checks do run:
the wrapper is `1bit serve --print-backend-env`, so a model that does not fit is refused by serve
(exit 3) with Lemonade reporting the load failure, not silently started.

Our own catalog rows can ride along through the config directory's `user_models.json`, which Lemonade
reads at start and lists like its own (`recipe` must be `llamacpp-hrx`, and the checkpoint names a
registry model, since this Lemonade version has no local-file registration):

```json
{"Qwen3.6-35B-A3B-NPU-1bit": {"checkpoint": "owner/repo:VARIANT", "recipe": "llamacpp-hrx", "labels": ["chat"]}}
```

## What has been measured where

| claim | where | result |
|---|---|---|
| a stock Lemonade takes a backend binary from `hrx_bin` / `LEMONADE_HRX_HRX_BIN` | extracted PPA deb `2026.44.1~26.04` on the dev box, isolated cache/config | `llamacpp-hrx` backend `installable` → `installed`, both forms, with no fork and no patch |
| the same recipe, both packages side by side | clean `ubuntu:26.04` container: PPA `lemonade-server` + this `.deb` | `lemond 2026.44.1` starts, `1bit-engine --version` and the shim run; the recipe's backend state is hardware-gated there (`unsupported: Unsupported GPU`), so the end-to-end chat is the hardware row below |
| the shim is the engine's own wrapper | `1bit/lemonade-hrx` from this repo's package (`scripts/package-linux.sh`) | it runs `1bit serve --print-backend-env` and then `exec`s the engine with serve's environment and defaults; `lemonade-bin/llama-server` is a symlink to it |
| the recipe's env for its backend is one variable | this tree, `src/cpp/server/backends/hrx/hrx_server.cpp` `build_server_environment()` | Lemonade passes its backend `GGML_DISABLE_VULKAN=1` only — which is why the wrapper, not our binary, has to be what the recipe starts |
| `amdxdna-dkms` from the PPA drags kernel bits | clean container, `apt-get -s install amdxdna-dkms` | `dkms`, `build-essential`, `linux-headers-7.0.0-42-generic`, `linux-headers-generic`, `amdxdna-dkms` |
| the driver packages install, build and uninstall on Debian/Ubuntu | clean `ubuntu:26.04` container, `scripts/pkg-proof.sh deb` | 22/22 rows PASS: dkms builds both modules against `7.0.0-43-generic`, `modprobe --show-depends amdxdna` resolves to `/lib/modules/7.0.0-43-generic/updates/dkms/amdxdna.ko.zst` (in front of the in-tree one), the kernel debs' `vmlinuz` is `sbverify`-clean against the certificate in `1bit-npu-config`, purge removes the /etc files |
| the same on Fedora | clean `fedora:latest` (fc45) container, `scripts/pkg-proof.sh rpm` | 22/22 rows PASS: modules under `/lib/modules/7.2.15-300.fc45.x86_64/extra/`, kernel rpms install `1bit-kernel` + `1bit-kernel-devel` with the build symlink at `/usr/src/kernels/7.4.0-rc1-1bit`, uninstall leaves nothing |
| the same on Arch | clean `archlinux:latest` container, `scripts/pkg-proof.sh arch` | 22/22 rows PASS: the pacman dkms hook builds both modules at install time, `linux-1bit` + `linux-1bit-headers` install with the build tree at `/usr/lib/modules/7.4.0-rc1-1bit/build` and `/etc/mkinitcpio.d/1bit.preset` |
| the dkms source reproduces the driver this box runs | `scripts/mk-kernel-packages.sh` payload built in a container against `7.0.0-43-generic` | `srcversion EE1F36A2B9BA294CE11AD80`, the srcversion of the `amdxdna.ko` installed at `/lib/modules/7.4.0-rc1-1bit/updates/` — the package's source *is* the fleet's series |
| the kernel the packages ship is the kernel this box runs | `linux-image-7.4.0-rc1-1bit_7.4.0~rc1-1_amd64.deb` payload vs `/boot/vmlinuz-7.4.0-rc1-1bit` | `sha256 6f3c6764…` in both, and `sbverify --cert MOK-Kernel.der` reports `Signature verification OK` |
| the shipped driver answers a known answer on the NPU | this box, `npu-hold`, `1bit serve --device npu` on Qwen3-0.6B | `GET /health` ok, greedy `The capital of France is` → the model's expected continuation, 0 `amdxdna` errors in the window (no install, no module or udev change) |
