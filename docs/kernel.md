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
# Kernel readiness: moving the box to a new kernel

The Strix Halo box runs a kernel we build, not the distribution's: everything the engine and the NPU
need is pinned by hand, so the next kernel is a step list, not an investigation. This page is that
list, with the facts that go stale marked as facts and the commands that do not.

- **Runs today:** `7.3.0-rc4-1bit-perfopt` — Linux `v7.3-rc4` plus two backports of the IOMMU
  PerfOpt series, built with clang (`LLVM=1`), debug info off.
- **Why the backports:** with the GPU in its identity domain, PerfOpt is worth real prefill on HRX and
  costs nothing at decode. Qwen3.8-27B UD-Q4_K_XL on HRX0, same build pin, five repeats:

  | arm | pp512 | pp2048 | tg128 |
  |---|---|---|---|
  | `7.3.0-rc4-1bit` (distribution-shaped base) | 364.0 | 353.1 | 12.00 |
  | `7.3.0-rc4-1bit-perfopt`, on (`PerfOpt armed on IOMMU0`) | **446.4** | **426.1** | 12.04 |
  | `7.3.0-rc4-1bit-perfopt`, `amdgpu.iommu_perfopt=0` | 358.4 | 347.0 | 12.06 |

  So the gain is PerfOpt itself, not the compiler (the off arm matches the base), it is prefill only,
  and greedy output is byte-identical across the three arms.
- **Out-of-tree modules that must follow every kernel:** `amdxdna` (the NPU driver, [npu.md](npu.md),
  "The XDNA driver") and `ec_su_axb35` (the box's fan EC). Both live in
  `/lib/modules/<version>/updates/` and both must be rebuilt against the new kernel and signed.

## What "ready" means

1. an upstream tag to move to, and whether the PerfOpt series is **native** in it (then the
   backports are dropped, not re-applied);
2. a kernel built from our config on another machine, packaged as `.deb`;
3. kernel and modules signed with the box's own MOKs;
4. the box boots it, the two out-of-tree modules load, `mokutil` agrees;
5. the [hardware gate](hw-gate.md) passes on it;
6. the weekly package and the 1bit OS image pick it up.

## 0. Facts first

```sh
git ls-remote --tags https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git | grep 'v7\.4'
# is our series already upstream in that tag?
grep -ril perfopt drivers/iommu/amd/      # a hit = drop the backports
```

The 2026-10-23 check: **no `v7.4` tag** (`v7.3` final was tagged 2026-10-18, the 7.4 merge window was
still open, so rc1 lands about two weeks later), no `v7.3.y` stable either, and `perfopt` was **not**
in `v7.3`'s `drivers/iommu/amd/` — the series is queued for 7.4, so the backports stay until then.
Both backport commits (the `iommu/amd` one and the `drm/amdgpu` one that enables PerfOpt on iGPUs in
the identity domain) applied to `v7.3` **cleanly**, needing no rebase.

## 1. Build — never on the box

The box is for running; a build there heats the die that a model is loading on. Build on a quieter
machine with the box's config.

```sh
git clone --depth 1 --branch <tag> https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git ~/wt/k74/<tag>
cd ~/wt/k74/<tag>
cp <box>:/boot/config-$(uname -r of the box) .config      # or the config of record, see below
scripts/config --file .config --set-str LOCALVERSION "-1bit-perfopt"
# only while the tag lacks the series:
git am --3way iommu-amd-perfopt.patch
make LLVM=1 olddefconfig
export LLVM=1                    # dpkg-buildpackage re-enters make: export it, do not just pass it
nice -n 19 make LLVM=1 -j12 bindeb-pkg
```

- The **config of record** is the one the box's running kernel was built from
  (`/boot/config-$(uname -r)`): `CONFIG_AMD_IOMMU=y`, `CONFIG_DRM_AMDGPU=m`,
  `CONFIG_MODULE_SIG{,_ALL,_SHA256}=y`, `CONFIG_CC_IS_CLANG=y`, `CONFIG_DEBUG_INFO_NONE=y`. Between
  the base `7.3.0-rc4-1bit` config and `-perfopt` the only differences are the clang version and
  debug info (28 lines, nothing workload-specific).
- `bindeb-pkg` needs `debhelper`, `libelf-dev` and the LLVM tools (`llvm-21`, `lld-21`); the kernel
  is clang-built, so a gcc-only host silently builds the wrong compiler unless `LLVM=1` is exported.
- Artefacts land in the **parent** directory: `linux-image-<version>_<arch>.deb` and
  `linux-headers-<version>_<arch>.deb`.
- `make install` on the box also works (that is how the current kernel was installed before the
  debhelper path existed), but the `.deb` route is what the weekly package expects and it puts the
  modules where `depmod` and the module-signing hook see them.

## 2. Sign — two keys, on the box

The box has two machine-owned MOKs enrolled, and they are not interchangeable. Whether Secure Boot
itself is on is the owner's setting in firmware (on 2026-10-23 the box booted with it **off**, after a
firmware reset); signing does not depend on it, and skipping it means the next boot with Secure Boot
on refuses the kernel.

| key | path | signs |
|---|---|---|
| `strixhalo Secure Boot Kernel` (EKU codeSigning) | `/var/lib/shim-signed/mok/MOK-Kernel.{priv,pem,der}` | the kernel image |
| `strixhalo Secure Boot MOK` (module-signing OID) | `/var/lib/shim-signed/mok/MOK.{priv,pem,der}` | every `.ko` we build |

```sh
# kernel: automatic on install (this is what the hook does), or by hand
sudo sbsign --key /var/lib/shim-signed/mok/MOK-Kernel.priv \
            --cert /var/lib/shim-signed/mok/MOK-Kernel.pem \
            --output /boot/vmlinuz-<version> /boot/vmlinuz-<version>
# modules we built ourselves (in-tree modules carry the build-time key already)
sudo kmodsign sha512 /var/lib/shim-signed/mok/MOK.priv \
                       /var/lib/shim-signed/mok/MOK.der updates/amdxdna.ko
```

- `/etc/kernel/postinst.d/z0-mok-sign` signs any kernel image the installer did not sign, and does
  nothing when `sbverify --list` already shows a signature.
- **Trap:** a MOK whose EKU is module-signing only (Ubuntu's default style) is **rejected by shim for
  kernel images**. That is why there are two keys; signing a `vmlinuz` with `MOK.priv` produces a
  kernel that will not boot under Secure Boot.
- **Trap:** an unsigned copy left in `/boot` as `vmlinuz-*.unsigned` becomes a GRUB menu entry.
  Keep originals outside `/boot` (`/var/backups/kernels/`).
- A BIOS/firmware reset can drop the enrolled MOKs. The fix needs the owner at the console
  (MokManager), and can be primed over ssh beforehand:
  `mokutil --import /var/lib/shim-signed/mok/MOK-Kernel.der`, then the same for `MOK.der`, then the
  owner confirms both in MokManager on the next boot.

## 3. Install — on the box

```sh
sudo dpkg -i linux-image-<version>_<arch>.deb linux-headers-<version>_<arch>.deb
sudo depmod -a <version>
ls /lib/modules/<version>/updates/            # the out-of-tree modules for THIS version go here
```

Removing a kernel is deleting `/boot/*<version>*` and `/lib/modules/<version>` (the `.deb` route also
offers `dpkg -r linux-image-<version>`). Keep the Canonical fallback kernel installed: it is the
recovery path when a new kernel does not reach a console.

## 4. Boot entry — without changing the default

`update-grub` (the installer runs it) adds the new kernel under "Advanced options for Ubuntu". Leave
`GRUB_DEFAULT` pointing at the kernel that is running: the owner chooses when the box moves, and the
previous kernel stays as the fallback entry.

## 5. Reboot — the owner, with a console

The build and install steps are remote; the reboot is not. Have console access before it (MokManager
is a firmware screen), and expect one extra reboot if a MOK has to be enrolled.

## 6. Verify, in this order

```sh
uname -r                                              # the new version
sudo mokutil --sb-state                               # SecureBoot enabled/disabled (owner's setting)
sudo mokutil --list-enrolled | grep -i 1bit           # both machine MOKs still enrolled
sudo sbverify --list /boot/vmlinuz-$(uname -r)        # "signature 1" + our issuer
for f in /lib/modules/$(uname -r)/updates/*.ko; do modinfo -F signer "$f"; done
sudo dmesg | grep -iE 'amdgpu|iommu|amdxdna' | grep -iE 'fail|error|timeout|fault'   # must be empty
```

Then the two box-specific gates: the IOMMU PerfOpt line (`PerfOpt armed on IOMMU0`, or
`amdgpu.iommu_perfopt=0` when it should be off) and the NPU driver — `amdxdna` loaded at the pinned
version and the NPU visible to `xrt-smi` ([npu.md](npu.md)).

## 7. The hardware gate

A boot is not a proof. `scripts/hw-gate.sh` (and the `hw-gate` label / `hw-gate.yml`) runs the six
checks that have caught the real regressions — ops, fa-decode-q8, kat, kld, needle, placement — and
it runs them on the box's new kernel, so a kernel that changes numerics, times out on a long prompt
or moves a node off HRX fails there rather than in a customer's run. Details:
[hw-gate.md](hw-gate.md).

## 8. Packages and the 1bit OS image

- The weekly release preflight needs a **readable** kernel image: `/boot/vmlinuz-$(uname -r)` is
  root-only on Ubuntu, so pass `VMLINUZ=<copy>` ([releases.md](releases.md), "The release run").
- `os/mini/mkimage.sh` builds the unified kernel image from that `vmlinuz`, and with `MOK_KEY` +
  `MOK_CERT` it signs the UKI with the same kernel key, so a 1bit OS stick boots with Secure Boot
  left on instead of asking for it to be switched off.
- Record the kernel version in the release notes: a box that moves kernel is a release-visible
  change, and the gate result belongs next to it.

## Checklist

- [ ] tag chosen; is the PerfOpt series native (drop the backports) or not (re-apply, and note any rebase)?
- [ ] config of record copied from the box, `olddefconfig` clean, no prompts
- [ ] `bindeb-pkg` produced image + headers `.deb` with the intended `LOCALVERSION`
- [ ] `amdxdna` and `ec_su_axb35` rebuilt against the new `/lib/modules/<version>/build`
- [ ] kernel signed by `MOK-Kernel`, modules by `MOK`, both verified with `sbverify`/`modinfo -F signer`
- [ ] fallback kernel and `GRUB_DEFAULT` untouched
- [ ] owner reboots with console access; MOKs still enrolled
- [ ] hardware gate passes on the new kernel
- [ ] weekly package and OS image built from the new kernel

## Rehearsal, 2026-10-23

No `v7.4-rc` existed yet, so this was rehearsed twice: against **`v7.3`** (the newest tag) with the
backports re-applied, and against **linux-next `next-20261022`**, which is the 7.4 merge window's
content (`Makefile` says 7.3.0 + next patches) and is therefore the honest preview of `7.4-rc1`.
Both were built on a separate 16-core machine with the config of record, `LLVM=1` (clang 21.1.8),
`nice -n 19 make -j12 bindeb-pkg`.

### Facts

| question | answer |
|---|---|
| `v7.4*` tags upstream | none (`v7.3` final is the newest tag, 2026-10-18) |
| `v7.3.y` stable | none yet |
| where is 7.4? | merge window open: torvalds master head is `Merge tag 'net-next-7.4'` (2026-10-23) |
| PerfOpt in `v7.3`? | **no** — 0 hits in `drivers/iommu/amd/`; the two backports apply cleanly (343 insertions over 8 files) |
| PerfOpt in `next-20261022`? | **yes** — present in `drivers/iommu/amd/` and as `iommu_perfopt` in `amdgpu_drv.c` |

So the rule the checklist asks for resolves itself: **on the 7.4 path the backports are dropped**, on
a 7.3-based kernel they are re-applied (and needed no rebase).

### Builds

| tree | backports | `bindeb-pkg` | artefacts | kernel release |
|---|---|---|---|---|
| `v7.3` | applied (2 commits on top of the tag) | pass, ~15 min | `linux-image` 84.6 MB + `linux-headers` 10.4 MB + `linux-libc-dev` | `7.3.0-1bit-perfopt+` |
| `next-20261022` | not needed (native) | pass, ~15 min | `linux-image` 85.0 MB + `linux-headers` 10.5 MB + `linux-libc-dev` | `7.3.0-next-20261022-1bit-perfopt` |

`olddefconfig` was clean on both. Its config delta is the part to read before adopting a tag: **28
lines** on `v7.3` (clang version and debug info only), **110 lines** on `next-20261022` — new
clang-21/7.4 symbols (`CC_HAS_COUNTED_BY`, `CC_OPT_*`, `BPF_CRYPTO`, …) plus options our config had
that the newer tree no longer offers (`ANDROID_BINDER*`, `ARCH_HAS_DEBUG_WX`,
`ARCH_SUPPORTS_PER_VMA_LOCK`). None of them is needed by the engine or the NPU, but the diff is where
a silently dropped feature shows up.

### The out-of-tree module

`amdxdna` (amd/xdna-driver at the engine's submodule pin) builds against **both** trees,
`MAKE_RC=0`, producing `amdxdna.ko` — **no 7.4 API break** in the 7.4 merge-window content. That is
the single most likely break in this whole path, and it is not there yet; it is worth re-running the
moment `v7.4-rc1` is tagged, because next is a moving target.

One trap the driver's own `Makefile` sets: it exports an **empty `ARCH`** into the kernel's
`configure_kernel.sh` probe, and the kernel then stops with
`scripts/Makefile.clang: add '--target=' option`. Pass `ARCH=x86_64` on the make line:

```sh
make -f drivers/accel/amdxdna/Makefile KERNEL_SRC=<kernel tree> ARCH=x86_64 \
     BUILD_ROOT_DIR=<out> LLVM=1 all
```

### Traps that cost time, in the order they bite

| trap | symptom | fix |
|---|---|---|
| missing packaging deps | `dpkg-checkbuilddeps: unmet build dependencies: libdw-dev` | `debhelper libelf-dev libdw-dev` (`libdw-dev` is the one nobody expects) |
| `LLVM=1` only on the command line | the packaging stage rebuilds with the wrong compiler | `export LLVM=1` — `dpkg-buildpackage` re-enters make |
| empty `ARCH` (above) | `add '--target=' option to scripts/Makefile.clang` | pass `ARCH=x86_64` |
| tree past its tag | kernel release and module vermagic end in `+` (`7.3.0-1bit-perfopt+`) | expected: `scripts/setlocalversion` appends `+` for commits past the tag. The `.deb` version is clean because `KDEB_PKGVERSION` is set; for a clean `uname -r` build from a tree with no `.git` (or override `KERNELRELEASE=`, which fixes the release string but was not exercised end to end) |
| an unsigned copy in `/boot` | a `vmlinuz-*.unsigned` menu entry | keep originals in `/var/backups/kernels/` |
| module-signing-only MOK | kernel image rejected by shim | sign images with `MOK-Kernel`, modules with `MOK` |
| firmware reset | Secure Boot back on with no MOKs | re-import both MOKs and confirm in MokManager at the console |

### Not rehearsed here

Installing, signing, booting and gating on the box (that is the owner's reboot), the fan-EC module
(its source lives only on the box), and any throughput re-measurement on the new kernel. Those are
steps 2 to 8 above, and the hardware gate is what makes the result more than "it booted".
