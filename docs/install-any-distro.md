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
| the engine | its own package for your distribution: `.deb` (Debian, Ubuntu), `.rpm` (Fedora, openSUSE Tumbleweed), `.pkg.tar.zst` (Arch), `.tar.zst` (anything else) | `1bit`, its HRX llama.cpp build (`hrx/llama/bin/llama-server`), the sealed NPU kernel sets, the HSA runtime it dlopens, and the engine's `lemonade-hrx` wrapper, which is what the recipe's `llama-server` name points at |

The engine package is userspace only: no kernel module, no Python at runtime. The four packages are
one payload in four wrappings (`scripts/pack-distro.sh`): they install the same files at the same
paths, so everything below about the drop-in, the config key and the shim is the same on every family,
and only the install command differs.

| your distribution | the engine |
|---|---|
| Debian, Ubuntu 26.04 or newer | `sudo apt-get install -y ./1bit-engine_<version>_amd64.deb` ([walkthrough below](#ubuntu-2604-resolute)) |
| Fedora (the current release) | `sudo dnf install ./1bit-engine-<version>-1.x86_64.rpm` |
| openSUSE Tumbleweed | `sudo zypper install ./1bit-engine-<version>-1.x86_64.rpm` (unsigned local file: [below](#fedora-and-opensuse-tumbleweed)) |
| Arch Linux | `sudo pacman -U 1bit-engine-<version>-1-x86_64.pkg.tar.zst` |
| anything else | the `.tar.zst`, [below](#any-other-distribution) |

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
`GLIBC_2.43`, and it is **one symbol** — `sqrtf@GLIBC_2.43`. The package declares that floor
(`Depends: libc6 (>= 2.43)`), so on Ubuntu 24.04 (noble, glibc 2.39) apt refuses it outright:
`1bit-engine : Depends: libc6 (>= 2.43) but 2.39-0ubuntu8.9 is to be installed`. Before that floor
was declared, the deb installed, `ldd` was happy, and the binary then refused to load with
`version 'GLIBC_2.43' not found`. Lemonade's own PPA does publish noble builds, so the engine half is
the one that pins the distribution: on noble, build `1bit` from source, or run it on a 26.04 box.
The `.rpm` says the same thing as `libc.so.6(GLIBC_2.43)` and the Arch package as `glibc>=2.43`; the
tarball carries the same binary and the same floor and is the one route that cannot refuse.

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

## Fedora and openSUSE Tumbleweed

One `.rpm` for both families, built without a distribution tag:

```bash
sudo dnf install ./1bit-engine-<version>-1.x86_64.rpm         # Fedora
sudo zypper install ./1bit-engine-<version>-1.x86_64.rpm      # openSUSE Tumbleweed
```

**Upgrading.** `sudo dnf upgrade ./1bit-engine-<new>.rpm` (zypper: `sudo zypper update ./…rpm`). Use
`upgrade`, not `install`: for a local file, `dnf install ./…rpm` performs whatever the file says — an
older file **downgrades** the package silently (measured), while `dnf upgrade` of an older file refuses
with `already installed. Nothing to do.`. On Debian and Ubuntu `apt-get install ./…deb` refuses a
downgrade outright, and on Arch `pacman -U` warns and does it. The version scheme makes all of them an
upgrade in the ordinary case: a weekly release is `2026.45`, the release before it `2026.44`, and a
candidate of it `2026.45~c32` / `2026.45c32`, which sit between the two.

The recipe (`packaging/rpm/1bit-engine.spec`) declares no dependencies: rpm reads the ELF files the
package ships and requires the sonames that come from outside it (`libcrypto.so.3`, `libgomp.so.1`,
`libzstd.so.1`, `libpcre2-8.so.0`, `libstdc++.so.6`, `libm.so.6`), which dnf resolves to `openssl-libs`, `libgomp`,
`libzstd`, `pcre2` and `libstdc++`, and zypper to the same packages under their own names. That also
encodes the floor above as `libc.so.6(GLIBC_2.43)`, so on an older release the install is refused
with `nothing provides libc.so.6(GLIBC_2.43)` instead of succeeding and then failing to load.

zypper prints `Package header is not signed!` for the local file — the release signs assets with
`SHA256SUMS`, not with an RPM key — so in a script pass `--allow-unsigned-rpm`, after
`sha256sum -c SHA256SUMS --ignore-missing`.

## Arch Linux

```bash
sudo pacman -U 1bit-engine-<version>-1-x86_64.pkg.tar.zst
```

Built from `packaging/arch/PKGBUILD` in this repository (an AUR entry can come later), with the
family's own dependency names: `glibc>=2.43`, `gcc-libs` (which carries `libstdc++`, `libgcc_s` and
`libgomp`), `openssl`, `zstd`, `pcre2`.
## Any other distribution

The same floor as above: the tarball carries the same `1bit`, so the distribution needs glibc 2.43 or
newer as well (measured: one symbol, `sqrtf@GLIBC_2.43`); the native packages say so and refuse to
install below it, and the tarball cannot. Unpack it at `/opt/1bit/engine` and install
Lemonade your distro's way (its `.deb`/`.rpm`, its container, or a source build — anything that reports
`llamacpp-hrx` in `system-info`):

```bash
sudo tar -C / --zstd --no-same-owner --no-overwrite-dir -xf 1bit-engine-<version>-linux-x86_64.tar.zst
sudo ln -sf /opt/1bit/engine/1bit /usr/local/bin/1bit-engine
# then the same two lines as above, in whatever runs lemond: the env var, or the config key.
```

The tarball's members are `opt/1bit/…`, `usr/bin/1bit-engine`, `usr/share/1bit/…`,
`usr/share/doc/1bit-engine/…` and `usr/lib/systemd/system/lemond.service.d/…` — the same paths the
`.deb` and the `.rpm` install, and no other, so `-C /` puts every file where they put it. Every member
is owned by `root:root` in the archive (`tar --owner=0 --group=0 --numeric-owner`), and the
filesystem's own directories (`/`, `/usr`, `/usr/bin`, `/usr/lib`, `/opt`, …) are not members at all:
they belong to the distribution's `filesystem` package. The two flags above are belt and braces from
the other side — a tarball of ours that ever carried a wrong owner, or a `./` member, would otherwise
rewrite the ownership of directories that are not ours when extracted as root, which is a local
root escalation on the machine (`#1292` review); `scripts/pack-distro.sh` also fails the build if a
member is not `0/0` or a `./` member appears.

A tarball also carries no dependency metadata, so nothing resolves what the native packages'
`Depends`/`Requires` name. If the engine reports one missing, it is the set the `.deb` declares
(`libgomp1`, `libssl3`, `libzstd1`, `libpcre2-8-0`; measured: a bare `ubuntu:26.04` container
answers `libgomp.so.1: cannot open shared object file` until `libgomp1` is installed).

A tarball has no post-install either, so on a systemd machine copy the drop-in
in by hand instead (the packaged families install it for you):

```bash
sudo install -D -m 0644 /dev/stdin /usr/lib/systemd/system/lemond.service.d/10-1bit-engine.conf <<'EOF'
[Service]
Environment=LEMONADE_HRX_HRX_BIN=/opt/1bit/engine/lemonade-bin
EOF
```

If your Lemonade runs under systemd, the drop-in from the deb applies verbatim; in a container or a
hand-started `lemond`, export `LEMONADE_HRX_HRX_BIN=/opt/1bit/engine/lemonade-bin` in its environment.

What every family installs, and where: `/opt/1bit/engine` (the engine, its backends, the sealed NPU
kernel sets, the HSA runtime) with `/usr/bin/1bit-engine` pointing at it,
`/usr/lib/systemd/system/lemond.service.d/10-1bit-engine.conf` (the env form),
`/usr/share/1bit/lemonade-config-fragment.json` (the config-key form) and
`/usr/share/doc/1bit-engine/` (README and licences).

## Kernel and NPU driver packages (optional)

The engine package is userspace only. The NPU and the board's ECU need three more pieces, and a machine
that stays on its distribution's kernel only needs the second and third:

| package | what it is | needed for |
|---|---|---|
| `linux-image-<rel>` / `linux-headers-<rel>` (deb), `1bit-kernel` + `1bit-kernel-devel` (rpm), `linux-1bit` + `linux-1bit-headers` (Arch) | the kernel 1bit OS boots: Linux 7.4-rc1 with this fleet's config, signed with the 1bit MOK | only if you want OUR kernel; the drivers build against yours as well |
| `amdxdna-1bit-dkms` | `amd/xdna-driver` at `526c6081` plus the engine's patch series (`third_party/patches/amdxdna/0001-0005`), as DKMS source | the NPU. **6.10+ kernels only** (measured, not inherited — see below), signed with **your** MOK |
| `ec-su-axb35-dkms` | the AXB35-02 board's embedded-controller driver (fans, EC temperature, power mode), GPL-2.0, upstream `cmetz/ec-su_axb35-linux` at `f62c2c22`, plus a DMI gate so it attaches only on an AXB35 | the fans on a Bosgame AXB35; inert on any other board |
| `1bit-npu-config` | `limits.d/99-amdnpu-memlock.conf` (unlimited memlock), `udev/rules.d/90-1bit-npu-power.rules` (NPU keep-awake), `modules-load.d/ec_su_axb35.conf`, and the public MOK certificate at `/usr/share/1bit/mok/MOK-Kernel.der` | the NPU runtime's locked command buffers, and engine#1051's suspend/resume hang |

The three driver/config packages are the same set in all three families, and that set alone is a
complete install for a machine that keeps its distribution's kernel — proved as its own path
(`mk-kernel-packages.sh --skip-kernel`, then `pkg-proof.sh`): 22 rows PASS with the kernel rows marked
skipped, ending with the distribution's own `amdxdna` restored. They install to
`/lib/modules/<ver>/updates/` (Debian, Arch) or `/extra/` (Fedora) through dkms, which is in front of
the in-tree `amdxdna`, and uninstalling the package puts the in-tree driver back — measured, not assumed:
dkms archives the distribution's own module to `/var/lib/dkms/<pkg>/original_module/` when ours goes in,
and on removal the `kernel/` file and what `modprobe` resolves to are both back.

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

**Why 6.10, measured.** The package declares `BUILD_EXCLUSIVE_KERNEL_MIN=6.10` (AMD's own floor, which
the engine inherits rather than invents), and that is not caution: the driver's trace-event header calls
the one-argument form of `__assign_str()`, which the kernel gained in 6.10 — before that the macro takes
two arguments. On **Ubuntu 24.04 GA (6.8.0-149-generic)**:

    $ dkms build -m amdxdna-1bit -v 0.18.0 -k 6.8.0-149-generic
    Error! … includes a BUILD_EXCLUSIVE directive which does not match this kernel/arch/config.
    This indicates that it should not be built.                                    (rc=77)

and with the floor bypassed the compiler stops at
`include/trace/events/amdxdna.h:32: error: macro "__assign_str" requires 2 arguments, but only 1 given`.
So on 24.04 the package refuses cleanly instead of leaving you a failed build to debug. The options
there are the distro kernel's own in-tree `amdxdna` (the warning above explains why ours exists) or a
6.10+ kernel. `ec-su-axb35-dkms` has no such floor in practice: it builds and installs on the same 6.8
headers (measured, `ec_su_axb35.ko` 380 440 B), consistent with its declared 6.1.

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

**One release, three families, one version.** `scripts/mk-kernel-packages.sh <out> <version>` takes the
release tag as the *release* field of the three 1bit packages (the kernel keeps the version it was built
with), and that tag carries into all three families — measured, because "the weekly calls it with a date"
is exactly the case a package can silently refuse to upgrade:

| family | package a weekly tag produces | the family's tool says | upgrade over the previous release |
|---|---|---|---|
| deb | `amdxdna-1bit-dkms_0.18.0-2026.45.0_all.deb` | `dpkg --compare-versions 0.18.0-1 lt 0.18.0-2026.45.0` true | `dpkg -i` rc=0, version after `0.18.0-2026.45.0`, one `/usr/src/amdxdna-1bit-0.18.0` left (no stale duplicate) |
| rpm | `amdxdna-1bit-dkms-0.18.0-2026.45.0.x86_64.rpm` | `rpmdev-vercmp` → `0.18.0-1 < 0.18.0-2026.45.0` | `dnf install` rc=0, version after `0.18.0-2026.45.0` |
| arch | `amdxdna-1bit-dkms-0.18.0-2026.45.0-x86_64.pkg.tar.zst` | `vercmp 0.18.0-1 0.18.0-2026.45.0` → `-1` | `pacman -U` rc=0, version after `0.18.0-2026.45.0` |

One convention difference worth knowing rather than discovering: this set puts the tag in the **release**
field, so the Arch package's `pkgrel` becomes `2026.45.0` instead of an integer. pacman accepts it and
orders it correctly (above), but it is not Arch's usual shape; the engine packages (`scripts/pack-version.sh`)
map the tag into the *version* instead (`2026.45c32`). Both work; neither is wrong; do not mix them in one
comparison.

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

If that answers `HRX strict placement refused the model: "embd" is GET_ROWS f32 … and is not placed
on HRX`, the model's graph has a node HRX cannot take under [serve's strict placement](serve.md)
(measured 2026-11-06: `Qwen3-0.6B-Q4_K_M` has one, `1 of 986 nodes off HRX`, and
`Qwen3-4B-Q4_K_M` has one too, `1 of 1266`). Add `--no-strict-device` to run those nodes on the CPU
— the row below was measured that way — or use a model whose graph places fully (Lemonade's
`*-HRX` rows are the ones we have measured end to end).

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
| **each family installs, runs and uninstalls** | clean containers, 2026-11-06, from the `rc-2026-11-06-c31` payload: `fedora:latest` (`dnf`, rpm 6.1.0, glibc 2.44), `opensuse/tumbleweed:latest` (`zypper`), `archlinux:latest` (`pacman -U`, pacman 7.1.0), `ubuntu:26.04` (`apt-get`) | all four: install, `1bit-engine --version` prints `1bit 0.0.1`, the shim `/opt/1bit/engine/lemonade-bin/llama-server --version` prints `version: 1 (32559003)` / `built with Clang 24.0.0 for Linux x86_64`, then remove and `/opt/1bit/engine/1bit` is gone. Fedora refused the first build and that was the payload's fault, not dnf's: the package listed `/usr/bin` and `/usr/lib` at 0755 while Fedora's `filesystem` owns them at 0555 (`file /usr/bin from install of 1bit-engine-2026.45-1.x86_64 conflicts with file from package filesystem-3.18-59.fc45.x86_64`); `scripts/pack-distro.sh` now normalises the payload's modes and leaves the distributions' own skeleton out of the file list |
| **the four packages are one payload** | the `.deb`, `.rpm`, `.pkg.tar.zst` and `.tar.zst` of the same run, unpacked as root in one `ubuntu:26.04` container (so each archive's own owner and mode are what lands) | 80 files each, 60 MB: identical paths, modes, **owner (`0:0`)**, sizes and sha256 across the four. The owner column was the gap in the first version of this check — `tar -tvf` said `bcloud/bcloud` for the tarball while the other three said root, and only the owner comparison showed it |
| the tarball cannot hand a system directory to uid 1000 | clean `ubuntu:26.04`, the documented `sudo tar -C / --no-same-owner --no-overwrite-dir -xf …` | `/`, `/usr`, `/usr/bin`, `/opt` read `0:0 755` before and after; the payload lands `0:0`, `0` of 75 files not owned by root, and the shim runs (`version: 1 (32559003)`). Before the fix the same extraction as root left uid 1000 owning `/`, `/usr`, `/usr/bin` and `/opt` — a local root escalation on every "any other distribution" install (#1292 review) |
| the glibc floor is enforced, not just documented | clean `ubuntu:24.04` container (glibc 2.39), `apt-get install ./1bit-engine_<version>_amd64.deb` | refused: `1bit-engine : Depends: libc6 (>= 2.43) but 2.39-0ubuntu8.9 is to be installed`. Before the floor was declared, the same deb installed, `ldd` was happy and the binary then failed to load |
| **the package's payload serves on the GPU** | strixhalo, the Fedora `.rpm` unpacked with `rpm2cpio` (so exactly the files the package installs), `1bit serve -m Qwen3-0.6B-Q4_K_M.gguf --device hrx --no-strict-device`, two chats | `1bit serve: HRX: HSA runtime …/_rocm_sdk_core/lib/libhsa-runtime64.so.1 sha256 21a66824…` — the runtime the package ships, not the box's — then `Qwen3-0.6B-Q4_K_M on hrx`, health `200`, and `request /v1/chat/completions -> hrx status 200 in 202.5 ms | prompt 16 completion 16 79.0 tok/s`; a second chat with `max_tokens: 200` came back `content: "5"` (109 completion tokens, 69.1 tok/s). The first one's content was empty because Qwen3 spends the opening tokens on its thinking trace, so a 16-token budget never reaches the answer. Both this model and `Qwen3-4B-Q4_K_M` are refused first by strict placement (`"embd" is GET_ROWS f32 … 1 of 986 nodes off HRX`), which is why the runs pass `--no-strict-device` |
| **the release proves its own packages** | `scripts/pack-proof.sh <out-dir> <version>`, wired into `scripts/weekly.sh`'s package stage, 2026-11-07, on the `rc-2026-11-06-c31` artifacts | 6/6 rows PASS: the `.deb` installed in `ubuntu:26.04`, the `.rpm` in both `fedora:latest` and `opensuse/tumbleweed`, the pacman package in `archlinux:latest` (each asked `1bit-engine --version` = `1bit 0.0.1`, made to run the shim = `version: 1 (32559003)`, then removed with nothing left), the tarball unpacked as root the way this page says, and the four compared as root (paths, modes, owner `0:0`, sizes, sha256). Handed the pre-fix tarball — uid 1000, a `./` member, the filesystem directories — the same harness fails **4/6**, naming the 151 members that are not root's and the payload difference, so the row is a check and not a claim |
| **a candidate package installs over the previous release** | `scripts/pack-version.sh` + `tests/pack_version_test.sh` (ctest `pack_version`), measured with each family's own comparator, 2026-11-07 | a candidate tag has to map to a version that sorts *above* the previous release and *below* the release it precedes, or installing it is a downgrade that apt and rpm refuse. `rc-2026-11-07-c32` packs as `2026.45~c32` (deb, rpm, tarball) and `2026.45c32` (pacman): `2026.44 < 2026.45~c32 < 2026.45` with `dpkg --compare-versions` and with rpm's `labelCompare` in `fedora:latest`, and `2026.44 < 2026.45c32 < 2026.45` with pacman's `vercmp` in `archlinux:latest`. The mapping this replaced read the tag as `2026.11.07.c32`, which is *below* every release since week 11 (`dpkg: 2026.11.07.c32 < 2026.44`) — the bug the #1368 review found, kept as a row in the test |
| **a newer package upgrades an installed one, in all three families** | 2026-11-08, clean containers, the `rc-2026-11-07-c32` payload built at three versions (`2026.44`, the candidate, `2026.45`) | `2026.44 → 2026.45`: `apt-get install` rc 0, `dnf` rc 0, `pacman -U` rc 0 — and the payload is replaced in place, 75 files before and after, in each. And the candidate path the #1368 review was about: `2026.44 → 2026.45~c32 → 2026.45` (deb, rpm) and `2026.44 → 2026.45c32 → 2026.45` (pacman), so a candidate installs over the previous release and its release installs over the candidate. Downgrades are the tool's business and they differ, measured on the same three containers: **apt refuses** (it needs `--allow-downgrades`), **`dnf install ./old.rpm` downgrades silently** while **`dnf upgrade ./old.rpm` refuses** (`already installed. Nothing to do.`), and **pacman warns and downgrades** |
| a stock Lemonade takes a backend binary from `hrx_bin` / `LEMONADE_HRX_HRX_BIN` | extracted PPA deb `2026.44.1~26.04` on the dev box, isolated cache/config | `llamacpp-hrx` backend `installable` → `installed`, both forms, with no fork and no patch |
| the same recipe, both packages side by side | clean `ubuntu:26.04` container: PPA `lemonade-server` + this `.deb` | `lemond 2026.44.1` starts, `1bit-engine --version` and the shim run; the recipe's backend state is hardware-gated there (`unsupported: Unsupported GPU`), so the end-to-end chat is the hardware row below |
| the shim is the engine's own wrapper | `1bit/lemonade-hrx` from this repo's package (`scripts/package-linux.sh`) | it runs `1bit serve --print-backend-env` and then `exec`s the engine with serve's environment and defaults; `lemonade-bin/llama-server` is a symlink to it |
| the recipe's env for its backend is one variable | this tree, `src/cpp/server/backends/hrx/hrx_server.cpp` `build_server_environment()` | Lemonade passes its backend `GGML_DISABLE_VULKAN=1` only — which is why the wrapper, not our binary, has to be what the recipe starts |
| `amdxdna-dkms` from the PPA drags kernel bits | clean container, `apt-get -s install amdxdna-dkms` | `dkms`, `build-essential`, `linux-headers-7.0.0-42-generic`, `linux-headers-generic`, `amdxdna-dkms` |
| the driver packages install, build and uninstall on Debian/Ubuntu | clean `ubuntu:26.04` container, `scripts/pkg-proof.sh deb` | all 31 rows PASS: dkms builds both modules against `7.0.0-43-generic`, `modprobe --show-depends amdxdna` resolves to `/lib/modules/7.0.0-43-generic/updates/dkms/amdxdna.ko.zst` (in front of the in-tree one), the kernel debs' `vmlinuz` is `sbverify`-clean against the certificate in `1bit-npu-config`, purge removes the /etc files |
| the same on Fedora | clean `fedora:latest` (fc45) container, `scripts/pkg-proof.sh rpm` | all 29 rows PASS: modules under `/lib/modules/7.2.15-300.fc45.x86_64/extra/`, kernel rpms install `1bit-kernel` + `1bit-kernel-devel` with the build symlink at `/usr/src/kernels/7.4.0-rc1-1bit`, uninstall leaves nothing |
| the same on Arch | clean `archlinux:latest` container, `scripts/pkg-proof.sh arch` | all 29 rows PASS: the pacman dkms hook builds both modules at install time, `linux-1bit` + `linux-1bit-headers` install with the build tree at `/usr/lib/modules/7.4.0-rc1-1bit/build` and `/etc/mkinitcpio.d/1bit.preset` |
| the dkms source reproduces the driver this box runs | `scripts/mk-kernel-packages.sh` payload built in a container against `7.0.0-43-generic` | `srcversion EE1F36A2B9BA294CE11AD80`, the srcversion of the `amdxdna.ko` installed at `/lib/modules/7.4.0-rc1-1bit/updates/` — the package's source *is* the fleet's series |
| the kernel the packages ship is the kernel this box runs | `linux-image-7.4.0-rc1-1bit_7.4.0~rc1-1_amd64.deb` payload vs `/boot/vmlinuz-7.4.0-rc1-1bit` | `sha256 6f3c6764…` in both, and `sbverify --cert MOK-Kernel.der` reports `Signature verification OK` |
| the amdxdna floor is really 6.10 | clean `ubuntu:24.04` container, headers `6.8.0-149-generic`, this page's packages | `dkms build` refuses with the declared exclusive rule (`rc=77`, "should not be built"), and with the floor bypassed the driver does not compile: `amdxdna.h:32: macro "__assign_str" requires 2 arguments, but only 1 given` — the 6.10 trace-event API. `ec-su-axb35-dkms` builds on the same headers (`380 440 B`), so only the NPU driver has the floor |
| a weekly release tag works in all three families, and upgrades | `scripts/mk-kernel-packages.sh <out> 2026.45.0 --skip-kernel`, each family's own tool, in `fedora:latest` / `ubuntu:26.04` / `archlinux:latest` | deb `0.18.0-1 < 0.18.0-2026.45.0` (dpkg), rpm `0.18.0-1 < 0.18.0-2026.45.0` (rpmdev-vercmp), arch `vercmp` `-1`; each installs over the previous package (`rc=0`) and reports `0.18.0-2026.45.0` afterwards, with one `/usr/src/amdxdna-1bit-0.18.0` left on deb — no stale duplicate source tree |
| uninstalling really restores the in-tree driver, in all three families | `scripts/pkg-proof.sh`, each family in its container, after the purge | the in-tree module is gone from the tree while ours is installed (dkms archives it to `/var/lib/dkms/<pkg>/original_module/`), and after removal both `modprobe --show-depends amdxdna` and the file are back on the `kernel/` path: `amdxdna.ko.zst` on Ubuntu and Arch, `amdxdna.ko.xz` on Fedora. Same three containers as the install rows, `PROOF-RC=0` |
| uninstalling the kernel package leaves the machine as it found it | `scripts/pkg-proof.sh`, each family in its container, after removing the kernel package | deb `dpkg -P linux-image` rc=0, rpm `rpm -e 1bit-kernel-devel 1bit-kernel` rc=0, arch `pacman -R linux-1bit-headers linux-1bit` rc=0; `/boot/vmlinuz-7.4.0-rc1-1bit` (or `/boot/vmlinuz-1bit` on Arch) is gone and the module tree under `modules/7.4.0-rc1-1bit` has **0 files** left. deb **29/29**, rpm **27/27**, arch **27/27** rows PASS, `PROOF-RC=0` |
| the memlock file is honoured, not just present | `scripts/pkg-proof.sh`, all three families, running the container with `--cap-add=sys_resource` | a probe login with the file rewritten to `4096` came back with exactly `4096` (lowering always works, so that value can only come from the file), and with the shipped file back the same login reports `ulimit -l: unlimited`. Without `CAP_SYS_RESOURCE` — which a container's default set excludes — raising the hard limit fails and the session keeps the container's `8192`, which is the container, not the package |
| the distribution-kernel path, on its own | clean `ubuntu:26.04` container, packages built with `--skip-kernel`, `scripts/pkg-proof.sh deb` | 22 rows PASS and one `INFO no kernel packages in this dir` (the kernel rows skip themselves instead of failing): install, config, memlock reaching a login session, dkms building both modules against `7.0.0-43-generic`, `amdxdna` resolving to `updates/dkms`, uninstall, and the distribution's in-tree `amdxdna` back in place. This is the path a user on their own kernel takes |
| the shipped driver answers a known answer on the NPU | this box, `npu-hold`, `1bit serve --device npu` on Qwen3-0.6B, `/v1/completions` greedy, twice | load 2.2 s, `The capital of France is` → **` Paris. The capital of Italy is Rome. The capital of Spain is Madrid.`** identically on both runs (56.9 and 50.7 tok/s), 0 `amdxdna` lines in the two windows, and the loaded module is `srcversion EE1F36A2B9BA294CE11AD80` — the build this package's source produces. No install, no `modprobe`, no udev reload |
