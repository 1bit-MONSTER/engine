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
# Puppy as the base of 1bit OS v3: evaluation

The owner's direction (2026-11-01): build 1bit OS v3 on **Puppy Linux** instead of Ubuntu Server —
Puppy for both roles, the live/try stick and the installed system (frugal install on the NVMe with a
persistent save layer), headless, with **our** kernel and **our** drivers, not the distribution's.

This page is the measurement, and the recommendation. Everything below is measured on strixhalo
(2026-11-01) or in QEMU here unless it says otherwise; the numbers that are not measured yet say so.

**Verdict: adopt, conditional on four blockers — and the conditions are small, each with a measured
path.** What makes the recommendation, in one line: Puppy already has the two things our BusyBox image
cannot grow (a package manager and a persistent layer on the stick) and a save-file model that *is*
#706, at a size the initramfs cannot match; what it does not have is our kernel, and that turned out to
be a kernel **config** gap of about fifteen `=y` symbols, not a fork.

## What was measured, and how

| build | how | result |
|---|---|---|
| Puppy **noble64** (Ubuntu 24.04 base) | the woof-CE CI artifact `ubuntu-noble64` run 38837834080 (`upup64n-24.04.iso`), downloaded with `gh run download` | boots to the first-run wizard in QEMU; the numbers below are this image |
| Puppy **resolute64** (Ubuntu 26.04, `DISTRO_NAME='ResolutePup64'`) | `os/v3/puppy/build.sh` — woof-CE CI's own recipe in a throwaway container (12 GB / 4 CPUs), our kernel dropped into `kernel-kit/output` | see [The build](#the-build-and-where-it-stands); this is the only base our binaries can run on |
| our **huge kernel** | `os/v3/puppy/mk-huge-kernel.sh` from the 7.3.0-rc4-1bit-perfopt+ deb tree | `huge-7.3.0-rc4-1bit-perfopt+.tar.zst` 111 MB: vmlinuz 12 MB + zdrv 100 MB, **3819 modules**, `amdxdna.ko` and `ec_su_axb35.ko` both signed, firmware amdgpu/amdnpu/rtl_nic |
| Puppy's initrd with **our** kernel | QEMU, `-kernel vmlinuz-7.3.0-rc4-1bit-perfopt+ -initrd initrd.gz`, the ISO as CD | the kernel boots (2.1 s) and Puppy's initrd stops: `*** /puppy_upup64n_24.04.sfs not found` → [Blocker 1](#blocker-1-the-kernel-config) |

## 1. Size and memory, against our initramfs

| | our c18 initramfs | Puppy noble64 (24.04) |
|---|---|---|
| medium | `1bit-os.img.zst` **397 MB**; the single EFI file **106 MB** | ISO **1,245,708,288 B (1.25 GB)** |
| the OS itself | kernel + initramfs, everything in RAM | `puppy_upup64n_24.04.sfs` **411,115,520 B (411 MB)**, `zdrv` 114 MB, `vmlinuz` 17.7 MB, `initrd.gz` |
| unpacked root | ~106 MB (initramfs) | **1.3 GB** extracted |
| package manager | none | `dpkg` + Puppy's PETget; **no apt** in noble64 (ResolutePup64's notes say apt/Synaptic — see [Blocker 4](#blocker-4-the-2604-base-and-what-it-carries)) |
| update story | reflash the stick | new SFS + keep the save layer; apt where the recipe has it |
| **boot to the first interactive screen** (QEMU, 2 vCPU, 2 GiB, serial timestamped from QEMU start) | **24.5 s** — the "1bit OS" banner and the menu | **108.8 s** — Puppy's first-run wizard (its own stock kernel; the wizard is interactive, not a shell) |
| RAM at idle | not measured here (no `free` in the c18 gate transcript) | not measured headless: the wizard takes the console before a shell exists (see [Not measured](#not-measured)) |

Size and time are the honest trade: **+850 MB on the stick, +1.2 GB on the installed disk, and +84 s
to the first screen** (24.5 s → 108.8 s, measured on the same QEMU with the same timestamping) for a
package manager, a save layer and a real userland. The c18 number is the boot of the published image as a
USB stick (`-device usb-storage`, which is what the init requires: it refuses a non-removable disk so an
internal disk's `/data` is never taken); the Puppy number is its stock kernel, so the comparison is the
two media, not the two kernels. For a 58 GB stick and a 2 TB NVMe that is nothing; for the
"100 MB EFI file" promise of today's image it is the whole OS.

## 2. The kernel is ours, and that is where the work is

`os/v3/puppy/mk-huge-kernel.sh` builds woof-CE's kernel package from our own deb tree, so no
distribution kernel appears anywhere in the image: `vmlinuz-7.3.0-rc4-1bit-perfopt+`, and an SFS with
`/lib/modules/7.3.0-rc4-1bit-perfopt+` (3819 modules, `.ko.xz` as Ubuntu ships them, loaded by the
kernel's own xz decompressor), `/lib/firmware/{amdgpu,amdnpu,rtl_nic}` and a marker `etc/1bit/kernel`
with the image's sha256 and each out-of-tree module's signer. The script refuses a module built for
another kernel or an unsigned one (`modinfo -F vermagic` / `-F signer`), which is the os/mini lesson
applied to a second consumer.

### Blocker 1: the kernel config

Puppy's initrd mounts the medium and layers the save file **itself**, before any module can be loaded.
Our kernel is Ubuntu-shaped: almost everything is `=m`. Measured, ours against Puppy's own stock kernel
6.8.12 from kernel-kit:

| knob | ours | Puppy's stock |
|---|---|---|
| `BLK_DEV_SR` (the CD) | `=m` | `=y` |
| `BLK_DEV_SD` / `BLK_DEV_NVME` (stick, frugal disk) | `=m` | `=y` |
| `ISO9660_FS` | `=m` | `=y` |
| `SQUASHFS` (XZ and ZSTD decompressors `=y` on both) | `=m` | `=y` |
| `OVERLAY_FS` | `=m` | `=y` |
| `BLK_DEV_LOOP` (a save file is a filesystem in a file) | `=m` | `=y` |
| `EFIVAR_FS` | `=m` | `=y` |
| `FAT_FS`/`VFAT_FS` + `NLS_CODEPAGE_437`/`NLS_ISO8859_1` | `=m` | `=y` |
| `EXT4_FS`, `USB`, `USB_XHCI_HCD`, `USB_STORAGE`, `HID_GENERIC`, `USB_HID` | `=m` | `=y` |
| `DEVTMPFS_MOUNT` | not set | `=y` |
| `DRM_SIMPLEDRM` | not set | `=y` |
| `AUFS_FS` | absent | absent (Puppy uses overlay — so do we) |

`os/v3/puppy/kernel-puppy.fragment` is exactly that list, with our current value beside each line.
It is a config fragment, not a patch: our kernel keeps every perfopt and driver decision it has, and
the weekly kernel build appends the fragment before `make olddefconfig`. The cost is one kernel rebuild
and a new A-phase gate; the risk is that a `=y` symbol pulls something into the image that was not
there, which is what the boot gate measures.

Until that fragment is in a built kernel, **the Puppy base cannot boot on our kernel at all** — measured
above, at 2.1 s into the boot.

## 3. Blocker 2: the binaries need glibc 2.43, so the base must be 26.04

Measured on the c18 release tarball, with `readelf -V`:

| binary | highest versioned symbol |
|---|---|
| `1bit` (the engine) | **GLIBC_2.43** |
| `lemond` (Lemonade) | **GLIBC_2.43** |
| `llama-server` (HRX) | GLIBC_2.34 |

Puppy noble64's `libc.so.6` defines at most **GLIBC_2.39** (read from the extracted SFS). So on a
24.04-based Puppy, `llama-server` would run and `1bit`/`lemond` would not: the appliance's two entry
points are exactly the two that fail. woof-CE has a **`resolute64`** recipe (Ubuntu 26.04, glibc 2.43,
"DISTRO_NAME='ResolutePup64'", SFS compression xz, package manager apt/Synaptic per its release notes),
which is the base this evaluation builds. That recipe is not in woof-CE's CI matrix — only
`ubuntu-noble64` and `ubuntu-jammy64` are — so a resolute Puppy is *ours to build*, and that is what
`os/v3/puppy/build.sh` does.

## Blocker 4: the 26.04 base and what it carries

From the noble64 SFS (measured by extraction) and ResolutePup64's own release notes, against what
#706/#704 and the appliance want:

| need | noble64 build | resolute64 recipe (per its notes) | what v3 must add |
|---|---|---|---|
| package manager | `dpkg` + PETget, no `apt` binary | apt + Synaptic | nothing if the notes hold; else ship `apt` |
| `python3` | yes (3.12) | yes | — (turns release-check's `sealed-sig` row from SKIP into a check) |
| `efivarfs` | kernel `=y` | recipe uses the same built-in shape | our kernel: fragment above |
| `efibootmgr` | yes, `/usr/bin/efibootmgr` | yes (dpkg) | — (engine#709's tool is there) |
| `mokutil` | **no** | **no** | ship it (`mokutil` or our own enrol step) for Secure Boot |
| `sshd` | **no** (ssh client only) | **no** | ship `openssh-server`, keys carried from the stick |
| persistent layer on the stick (#706) | pupmode 5/12/13, save **file** or **folder** — the initrd's own machinery (`sbin/load_ext_file`, `validate_pupsave_upgrade`) | same | size it (`psave=`/`pupsave`) and document it as the answer to #706 |
| release identity (#704) | none | none | `/etc/1bit/release-identity`, written by our package/first boot |
| headless | first-run wizard asks keyboard/locale/X even with `pfix=nox` | same | pre-answer it (`/etc/rc.d/rc.local` hook or a seeded `PUPSTATE`) |
| user model | **root only**, no `onebit` user, no sudo model | same | decide honestly: run as root and say so, or add a user + `su`/`doas`; Puppy's tools assume root |
| X11 baggage | X.Org, GTK3, JWM, Roxfiler, Palemoon (~1 GB unpacked) | same | not started when headless (`pfix=nox`), but it is on the disk |

That table is the real cost of the Puppy plan: the appliance's identity (a user, sshd, mokutil, a
release identity) is **four packages and one seeded first boot**, and the "headless" promise is one flag
plus one pre-answered wizard — but the root-only model is a design decision, not a package.

## 5. Secure Boot: what the Puppy medium has, and what ours has to be

The noble64 ISO's `EFI/boot/` carries `bootx64.efi`, `grubx64.efi`, **`mmx64.efi`** — a shim +
grub + MokManager chain, so a Puppy medium *can* boot with Secure Boot on in principle. Our kernel is
signed by **our** MOK (`CN=strixhalo Secure Boot Kernel`), which a stock Puppy chain does not have in
its db/MOK, so the v3 medium replaces the chain with the one os/mini and os/v2 already boot: shim
(Microsoft-signed) → our MOK-signed grub → our MOK-signed kernel → our modules. The measurement that
matters and is **not** done yet: booting the swapped medium with OVMF Secure Boot on, then enrolling our
MOK once in MokManager (the same one-keyboard-step the Ubuntu plan documents). It is the same chain, the
same key and the same step as today's image, on a different initrd.

## 6. What Puppy gives that our initramfs cannot

- **A package manager and an update path on the medium itself**: a stick that can be repaired and
  extended (`apt`/PETget) rather than reflashed.
- **A persistent layer that is #706's answer**, with a decade of save-file handling (load, upgrade,
  validate) and the choice of save file in a folder or inside a file — the initrd already implements
  the hard part.
- **A real userland on the try-it medium**: python3, bash, git, editors, so the stick is a rescue system
  too, and `1bit-release-check` can run its full row set (its `sealed-sig` row needs python3, which the
  c18 image SKIPs for exactly that reason — measured in the gate log).
- **Frugal install**: the same SFS + save layer boots from the NVMe, with the OS read-only and the save
  layer writable — the "installed Puppy" mode the owner asked for, without an installer writing a
  rootfs at all.

## The build, and where it stands

`os/v3/puppy/build.sh` is woof-CE's CI recipe (`build.yml`) as one command, in a throwaway container,
with two things it does not say out loud and this evaluation found the hard way:

1. `merge2out` refuses to run when the host is not a running Puppy **unless `GITHUB_ACTIONS` is set** —
   CI sets it, so a container build must too (otherwise the build stops at "you will still need to
   download a Puppy ISO and devx sfs" and looks like a hang);
2. `3builddistro` takes the kernel from `woof-out_*/kernel-kit/output/` and **otherwise downloads a
   stock Ubuntu kernel** — our huge tarball has to be dropped there, which `build.sh` does.

Attempts on 2026-11-01, in order, each one a measured trap:

| # | stopped at | cause | fix in `build.sh` |
|---|---|---|---|
| 1 | the `vercmp` compile | the container had no `libc6-dev` | install `build-essential`/`libc6-dev` |
| 2 | `merge2out`, "you will still need a Puppy ISO and devx" | **`merge2out` refuses to run outside a running Puppy unless `GITHUB_ACTIONS` is set** — CI sets it, a container build must too | `export GITHUB_ACTIONS=true` |
| 3 | `merge2out`, "Press ENTER to continue" | `</dev/null` on a `yes "" \| ./merge2out` pipeline **discards the pipe** | drop the redirect; let `yes` feed it |
| 4 | right after `merge2out`, no error | `set -o pipefail` + `yes \| merge2out`: when `merge2out` exits, `yes` gets SIGPIPE and the pipeline's status kills the build | `set +o pipefail` around that line |
| 5 | `3builddistro` → petbuild `busybox`: `chroot: failed to run command 'bash': No such file or directory` | the petbuild chroot is an **overlay mount**, and inside Docker the upper/work dirs sit on Docker's overlay rootfs — the kernel refuses it: `mount: ... bad superblock on petbuild`, so the rootfs is empty | build in a **host bind mount** (`BUILD_ROOT`, `/work`), not in the container's rootfs |
| 6 | `3builddistro` → debootstrap of the `bdrv`: `cannot create … test-dev-null: Permission denied`, "mounted with noexec or nodev" | the petbuild overlay fix moved the build root to `/tmp`, and **`/tmp` here is tmpfs, `nodev`** — debootstrap needs a usable device node in its target | default `BUILD_ROOT` to `$HOME/wt/1bit-puppy-build` (ext4) and **refuse a nodev/noexec build root up front** |
| 7 | running | — | — |

Attempt 5 is the informative one: `0setup` ✓, `1download` ✓ (the 26.04 package set, 37k log lines),
`2createpackages` ✓, then the petbuild stage. So the recipe is proven through package creation and the
remaining dependency is a filesystem property of the build *environment*, not of Puppy or of our kernel.

What is worth saying plainly: the build system is from 2009; it is maintained (a CI matrix builds five
targets twice a month) and it is *interactive* and environment-sensitive in several places that matter,
and every one of those places cost an attempt. A product build has to run `woof-code`'s stages from a
script that answers them, on a filesystem that can host its mounts — which is what `build.sh` now is. A
finished ISO is **not** claimed here; G1 stays open until one builds and boots.

## The gates this evaluation would accept, and which of them are green

| gate | what | status |
|---|---|---|
| G1 | a resolute64 Puppy ISO builds from our recipe with **our** kernel tarball in `kernel-kit/output` | **recipe proven through `2createpackages`** (attempt 5); `3builddistro`'s petbuild stage needs the host-filesystem build root that attempt 6 uses — ISO pending |
| G2 | that ISO boots in QEMU under our kernel, headless (`pfix=nox`), to a shell | **red**: our kernel's `=m` medium/layer drivers ([Blocker 1](#blocker-1-the-kernel-config)); the fix is `kernel-puppy.fragment` + a kernel rebuild |
| G3 | `uname -r` is our kernel, and `amdgpu`, `amdxdna`, `ec_su_axb35` are present and signed (VM: modules present; board: devices) | kernel half measured (tarball 3819 modules, both OOT modules signed), boot half blocked by G2 |
| G4 | Secure Boot with our shim/MOK chain, kernel verifies, MOK enrolment is one console step | not run; the chain is today's and the ISO carries a shim/MokManager |
| G5 | `lemond` + `1bit serve` run unchanged from the c18 tarball | **blocked by glibc on 24.04** (measured 2.43 vs 2.39); needs the resolute base |
| G6 | `1bit-release-check` green as far as a VM allows | not run (needs G2+G5) |
| G7 | a save layer persists across a reboot (frugal install on a qcow2, then reboot) | not run |
| G8 | image size, boot time, idle RAM, against the c18 initramfs | size measured; boot time and RAM pending G2 |

## Recommendation

**Adopt Puppy for both roles — the live/try stick and the installed frugal system — on the resolute64
base, with our kernel and the `kernel-puppy.fragment`, and keep the Ubuntu Server autoinstall plan in
[os-v3.md](os-v3.md) as the documented fallback.**

The reasons, in the order they decide it:

1. Puppy is the only candidate that answers #706 (a persistent layer on the stick) and #704's needs
   with machinery that already exists, on a medium that can also repair itself. The Ubuntu plan
   re-installs a rootfs to change anything; Puppy layers a save file.
2. The blockers are all **small and measured**: one config fragment (fifteen `=y` symbols), one base
   version (resolute64, which woof-CE already has), four packages (apt where missing, sshd, mokutil, our
   release identity), and one pre-answered first boot. None of them is a fork of Puppy and none is a
   redesign of our kernel.
3. The cost is size (1.25 GB medium, 1.3 GB root) and a root-only user model. The size is affordable;
   the root model must be an explicit decision in v3, not a discovery.

**When the fallback wins**: if the kernel rebuild for the fragment turns up a perfopt/driver regression
that would delay the release, the Ubuntu Server autoinstall plan has no such dependency (its kernel is
the same one we ship today, and Ubuntu's userland brings sshd, apt, efivarfs, a normal user and
`unattended-upgrades` for free). The two plans share the seed's *content* (user, keys, disk policy),
the ESP chain, #709's boot entry and #710's install screen, so switching between them is a change of
medium, not of design.

## Not measured

- **idle RAM**, headless, for either: the c18 gate's transcript has no `free`, and Puppy's wizard owns
  the console before a shell exists, so both need the pre-answered first boot (or a serial hook that runs
  a `free`/`/proc/meminfo` after boot). The boot-time part of that item *is* measured now (24.5 s vs
  108.8 s, table above) with the harness that exists: a timestamping serial tee
  (`[NN.Ns] ` prefix per chunk) + `systemd-run --user --scope -p MemoryMax=6G`.
- **the persistence round trip** (G7) and **Secure Boot with the swapped chain** (G4).
- **the resolute64 ISO itself** — the build is the open item of this page.
- **project health and licensing** beyond the facts stated: woof-CE builds five targets on CI; the
  recipes are GPL-2.0-or-later with the built distro's own licences; ResolutePup64 is community-built
  and its release notes are the only English documentation of the recipe.
