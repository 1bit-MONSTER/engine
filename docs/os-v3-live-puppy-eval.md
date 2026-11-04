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

> **Update (2026-11-02): the four blockers are closed and Puppy is 1bit OS.** `kernel-puppy.fragment`
> is built into our kernel and the initrd finds and layers the SFS (G2 green); `amdxdna` and
> `ec_su_axb35` are present, signed and verify against the enrolled MOK (G3 green); the save layer
> persists across a reboot (G7 green, `PUPMODE=12` on `1BIT-DATA`); and `apt`, `sudo`, `sshd` and
> `mokutil` are in the image, which was the one open question in Blocker 4. The appliance **booted on
> the Max+ 395 with Secure Boot on** on 2026-11-02: 20 of `puppy-check`'s 25 rows PASS (the 5 that do
> not are the cycle-12 list), Lemonade answers on `:13305` with its key, chat 70.49 tok/s, TTFT 1.30 s,
> 0 `amdxdna` errors. It boots headless by default, with a second boot entry, `1bit OS (desktop)`, for
> Puppy's JWM desktop. **Nothing is published**: the next release carries it. The body of this page is
> kept as the measurement that decided it.

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

One thing the first version of the fragment got wrong, found by reading our own config rather than by a
build: a tristate cannot be `=y` while a symbol it `depends on` is `=m`, and `make olddefconfig` does not
refuse — it **silently reduces the child back to `=m`**. Ours are `=m` all the way down (`SCSI`,
`SCSI_MOD`, `SCSI_COMMON`, `CDROM`, `ATA`, `SATA_AHCI` in
`/boot/config-7.3.0-rc4-1bit-perfopt+`), so `BLK_DEV_SR=y` on its own would have built a kernel that
still could not read the ISO, with nothing in the build output saying so. The fragment now carries the
chain above the drivers (24 symbols, from 20), and the per-symbol `grep -qx "$sym=y"` check in
`os/v3/puppy/README.md` is what makes that class of mistake loud.

The same trap had a second half, and only a real build found it (2026-11-02): the *console* symbols were
under-parented. `DRM_SIMPLEDRM`, `HID_GENERIC` and `USB_HID` are children of `DRM` and `HID`, both `=m`
in our config, so with the fragment as it stood `olddefconfig` reduced exactly those three back to `=m`
— 3 of 24 missing, reported by the fragment's own verification loop and by nothing else. `DRM` and `HID`
are the only two a fragment can raise: `DRM_KMS_HELPER` and `DRM_GEM_SHMEM_HELPER` are select-only
(`tristate` with no prompt) and follow `DRM_SIMPLEDRM=y`, so writing them into the fragment would be lines
`olddefconfig` rewrites. The fragment is **26 symbols** with those two parents in it, and every one of the
26 lands `=y` on v7.3-rc4.

Until that fragment is in a built kernel, **the Puppy base cannot boot on our kernel at all** — measured
twice, and the second time on the real artifacts rather than on a borrowed initrd: the built ISO's own
`initrd.gz` (2,369,371 B, 127 cpio entries, from `initrd-progs`) on the ISO's own `vmlinuz`
(`7.3.0-rc4-1bit-perfopt+`) stops with

```
Waiting 5 seconds for slow storage devices [0/5] ... [4/5].
Finding puppy main sfs file. failed
  3: PSUBDIR= P_BP_FN= P_DEF_FN=puppy_upup64r_26.04.sfs
  HAVE_PARTS=''      TRY_PARTS=''
  6: ONE_PART= ONE_TRY_FN= PDRV=
***  /puppy_upup64r_26.04.sfs not found.
*** Error is too critical, dropping out to console...  ~ #
```

`HAVE_PARTS=''` is the part worth reading twice: the init did not fail to *mount* the medium, it found
**no partitions at all** — no `/dev/sr0`, because `sr_mod` is a module and an initrd that cannot load
modules has nothing to read the ISO with. That is precisely the first block of the fragment
(`BLK_DEV_SR`, `ISO9660_FS`, `BLK_DEV_SD`, `BLK_DEV_NVME`, `USB_STORAGE`) set to `=y`.

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
| package manager | `dpkg` + PETget, no `apt` binary | **built here: `dpkg` + PETget, no `apt` either** (the recipe's own notes claim apt/Synaptic; the built rootfs has `/usr/bin/apt` absent — measured) | **ship `apt`** if "updates from the Ubuntu feed" is the promise |
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
| 7 | `mk_iso.sh`, the last step: `mkdosfs: command not found`, then `losetup: /dev/loop16: failed to set up loop device` | the ISO wrapper builds a FAT EFI image with `mkdosfs` (the container had no `dosfstools`) and sets up a loop device (the container has no `/dev/loop-control`) | `dosfstools` in the container; `LOOP_DEVICE=1` passes `/dev/loop-control` through |
| 7 (second defect) | **nowhere — it stayed green** and produced an ISO whose `initrd.gz` was **27 bytes** ("original size modulo 2^32 0") | `initrd-progs` packs the initrd with `find . \| cpio -o -H newc > ../initrd 2>/dev/null`: the container had no **`cpio`**, so the redirect created an empty file, stderr was suppressed, `gzip` compressed nothing, and `3builddistro` carried the result into a bootable-looking ISO | `cpio` in the package list, plus a `command -v cpio` refusal before `merge2out` — this trap's whole danger is that nothing fails |
| 8 | `1download`: `DOWNLOAD FAIL: libnl-genl-3-200_3.12.0-2_amd64.deb`, all four mirrors | **not a recipe defect**: one package out of 1241, and the same URL answered `200 OK` minutes later. The real cost is that a fresh `BUILD_ROOT` re-downloads 2.6 GB and one mirror hiccup over a thousand files fails the build | reuse `local-repositories`: attempt 9 copied attempt 7's cache into the new build root and touched no mirror at all |

Attempt 7 is the informative one: it went through `0setup` ✓, `1download` ✓, `2createpackages` ✓ and
every stage of `3builddistro` up to `mk_iso.sh`, and **produced the payload** - `sandbox3/rootfs-complete`
(1.5 GB), `devx_upup64r_26.04.sfs` 168 MB, `nlsx` 115 MB, `docx` 53 MB, with **our kernel tarball taken
from `kernel-kit/output`**. It died only in the ISO wrapper, on two container facilities (`mkdosfs`,
`/dev/loop-control`), neither of which the *installed* (frugal) mode needs: a frugal install is the SFS
files on a partition, not an ISO.

### The ISO, finished on the host (attempt 7's payload, no rebuild)

The two missing facilities are on this box (`/usr/sbin/mkfs.fat` from `dosfstools`, `/dev/loop-control`,
passwordless `sudo`), so the wrapper did not need the container and did not need the 1.5-hour build
repeated. Two steps, from `sandbox3`:

```sh
cd ../initrd-progs && sudo bash ./build.sh -prebuilt -auto -arch x86_64   # 27 B -> 2,369,371 B
sudo cp initrd.gz ../sandbox3/build/initrd.gz
cd ../sandbox3 && sudo bash ../support/mk_iso.sh                          # xorriso 1.5.6, mkfs.fat 4.2
```

`upup64r-26.04.iso`, **1,294,471,168 B** (632,066 sectors), sha256
`805938104b7076b66f88046ae602eb94b16092ebc6746bc5cf213987ef9cc388`, md5
`a27ddf2ad1ad054c3284abdd79f9367b`. What is inside it, as `xorriso -find` lists it:

| file | bytes |
|---|---|
| `vmlinuz` | 11,563,520 — `7.3.0-rc4-1bit-perfopt+` (the string is in the image), from our huge tarball |
| `initrd.gz` | 2,369,371 — 127 entries, `/init` present |
| `puppy_upup64r_26.04.sfs` | 456,491,008 |
| `zdrv_upup64r_26.04.sfs` | 104,599,552 — the 3819 modules + firmware the kernel boots with |
| `fdrv_upup64r_26.04.sfs` | 698,683,392 |
| `bdrv_upup64r_26.04.sfs` | 7,708,672 |
| `/EFI/boot/{bootx64,grubx64,mmx64}.efi` | 960,080 / 3,854,336 / 850,176 — shim + grub + MokManager |
| `/boot/efi.img` | 6,188,544 — the FAT image the UEFI El Torito entry boots |
| `/isolinux.bin` | the BIOS El Torito entry ("isohybrid-suitable") |

Both boot entries are real (`xorriso -report_el_torito`: BIOS → `/isolinux.bin` LBA 3063, UEFI →
`/boot/efi.img` LBA 41). One honest gap: **`isohybrid` is not installed here**, so the image is not
isohybrid-patched — it boots from a CD or an optical-emulated stick, and a raw `dd` to a stick would get
no MBR. `syslinux-utils` is already in `build.sh`'s container list, so the container path would have it;
the host run did not.

The frugal payload (the mode the owner asked for) is those SFS files, and they are kept with the ISO in
`~/wt/puppy-out-attempt7/` — 2.4 GB together.

### Then the recipe itself, one command (attempt 9): `rc=0`, and a better ISO

That host-side finish proved the payload and the wrapper but not the *recipe*, so the fixed `build.sh`
was run end to end again — fresh build root, woof-CE passed as the 4th argument, `nice -n 19`, package
cache pre-seeded from attempt 7 (trap 9). **`build.sh resolute64 … ` exited 0 and wrote the ISO itself**:
`upup64r-26.04.iso`, **1,317,011,456 B**, sha256
`41e23602269b3bc8bf5751c646c1a3f559a846aa44751a4e584ed090286f857e`, md5
`d60cc2809fdcb5951a8bb70b90bca408`, in about 20 minutes in the 12 GB / 4 CPU container.

Diffing its file list against the host-finished image leaves exactly two differences, and the second one
makes it the better artifact:

| file | attempt 7 (host-finished) | attempt 9 (the recipe) |
|---|---|---|
| `initrd.gz` | 2,369,371 | 2,369,395 — the 24 bytes are `DISTRO_PUPPYDATE` |
| `ucode.cpio` | **0** — "Building without early microcode loading" | **21,672,960** — `amd-ucode` collected into the early-load cpio |
| everything else | identical byte counts, both El Torito entries | |

So the recipe's own output carries **early AMD microcode loading** as well, which the interrupted attempt
7 tree never got to. `build.sh` now also copies the four SFS out of `sandbox3/build/` into `<out-dir>`
next to the ISO, because those files *are* the frugal install.

G2 was re-run on this second ISO too (`os/v3/puppy/g2-probe.sh`, exit 1): same banner, same
`Finding puppy main sfs file. failed`, same `HAVE_PARTS=''`. Two independently built images behave
alike, which is what a red gate should look like when the cause is the kernel config and not the image.

The built **resolute64 (Ubuntu 26.04) rootfs** measured, which is the base the appliance would run on:

| | resolute64 (built here) | noble64 (CI artifact) |
|---|---|---|
| glibc defines up to | **GLIBC_2.43** — exactly our binaries' floor | GLIBC_2.39 (too old: `1bit` and `lemond` need 2.43) |
| python3 / dpkg / bash / curl | yes / yes / yes / yes | yes / yes / yes / yes |
| **apt** | **no** (`/usr/bin/apt` absent) | no |
| efibootmgr / mokutil / sshd | yes / **no** / **no** | yes / no / no |
| rootfs | **1.5 GB** unpacked | 1.3 GB |
| `/lib/modules` in the rootfs | empty, by design: the modules come from the zdrv at boot | same |

What is worth saying plainly: the build system is from 2009; it is maintained (a CI matrix builds five
targets twice a month) and it is *interactive* and environment-sensitive in several places that matter,
and every one of those places cost an attempt. A product build has to run `woof-code`'s stages from a
script that answers them, on a filesystem that can host its mounts — which is what `build.sh` now is.
**A finished ISO exists, and the recipe produces it**: `build.sh` run end to end exits 0 and writes
`upup64r-26.04.iso` (1,317,011,456 B, sha256 `41e23602…6f857e`) carrying our kernel, our initrd, both
boot entries and early microcode. The first ISO here was finished on the host from attempt 7's payload
(above), and the second was built by the recipe alone.

## Where the appliance stands (2026-11-02 13:4x)

This section is newer than the gate table below it and supersedes it where they differ: everything here was
measured on the artifact that ships, a **GPT disk image** (not an ISO), booted under UEFI firmware in QEMU and
driven over a serial root shell.

**The artifact** (`os/v3/puppy/mkimage.sh <puppy-out-dir> <out-dir>`): `1bit-os-puppy.img`,
**9,128,902,656 B**, sha256 `2d13c1835876eb6068…` — an ESP (shim → grub → `grub.cfg` with a serial terminal)
plus a `1BIT-DATA` ext4 partition holding the frugal install at `/1bit` and os/mini's data contract. The ISO
(`upup64r-26.04.iso`, 1,362,100,224 B, sha256 `d6319f1de2c473cc0e…`) is a side artifact for booting without a
disk. `os/mini/prepare-stick.sh` runs on it **unchanged** and passes its ten assertions (grow `1BIT-DATA`,
write `ssh/authorized_keys` at 0600 root:root, and mint `LEMONADE_API_KEY` into `lemonade/lemond.conf` at 0600
only when asked with `--mint` — since cycle 20 the appliance ships with **no key**, like stock Lemonade, and the
console's menu item 1 turns one on for a shared network).

**Rows measured on that image** (`os/v3/puppy/vm-check.sh`, QEMU + OVMF + serial shell):

| row | evidence |
|---|---|
| U1 shell, sudo | `SHELL=/bin/bash`, bash 5.3.9(1), `sudo -n id -un` → root |
| U1 apt | `apt update` + install over the LAN (`fastfetch` → `/bin/fastfetch`, `cowsay` → `/usr/games/cowsay`). **`neofetch` is not in the 26.04 archive at all** — measured on the box, both `apt-cache policy` and `apt-cache search` are empty — so U1's literal test command cannot pass on resolute for anyone; `fastfetch` is the packaged equivalent |
| U3 Lemonade | key resolves to `/data/lemonade/lemond.conf`, `http=200` on `/api/v1/models` inside the guest **and** on the forwarded port with that key; `zz-onboard-ports` opens 13305 and 22 after `rc.firewall` |
| U4 engine | `1bit 0.0.1` from `/usr/local/bin/1bit`, `/opt/1bit-package` with both launchers, `ADDLIST=ec_su_axb35` in the running image |
| data contract | `/data` mounted (7.8 G, 1.3 G used), `/etc/1bit/lemond.conf` and `/etc/ssh/ssh_host_ed25519_key` both symlinked onto the partition |
| U7 ssh | a completed login over the network with the key on the stick (`debug1: Exit status 0`) |
| modules | `/lib/modules/<kver>` visible, `enp0s3` up, `10.0.2.15/24` leased by connman, default route |

**Five defects found by booting it, none visible to a probe that only reads the image:**

| # | defect | what it looked like | fix |
|---|---|---|---|
| 1 | the zdrv put modules at `/lib/modules` while 26.04 is usrmerged and the puppy SFS carries `lib -> usr/lib` | **no module could load**: `modprobe virtio_net`/`e1000` "not found in directory", `ls /lib/modules/<kver>` missing, only `lo`, so no NIC, no `amdgpu`, no `amdxdna` | tree at `usr/lib/modules` + firmware at `usr/lib/firmware`, with the zdrv's own `lib -> usr/lib` (`mk-huge-kernel.sh`) |
| 2 | `3builddistro` starts with `rm -rf sandbox3/rootfs-complete` and rebuilds it from `rootfs-skeleton` | the ADDLIST edit and the embedded engine were **silently discarded**; the image booted and simply had neither | write into `rootfs-skeleton`; verify the packed SFS and `unsquashfs -cat MODULESCONFIG` (`build.sh`) |
| 3 | Puppy's firewall is `-P INPUT DROP` with `tcp_inbound` RETURNing to that policy | LAN clients could not reach lemond while loopback answered 200; inserting one ACCEPT turned the outside request into a 401 | `zz-onboard-ports`, named to sort **after** `rc.firewall` |
| 4 | `/etc/hosts.deny` = `ALL: ALL` with `/etc/hosts.allow` = `ALL: LOCAL`, and `LOCAL` means this machine | ssh closed at `kex_exchange_identification` while sshd listened on 0.0.0.0:22; lemond was unaffected because it is not libwrap-linked | the RFC1918 ranges appended to `hosts.allow` (the LAN, not the world) |
| 5 | installed files carried the builder's uid (`cp -a` from a host bind mount) | `/root/.ssh/authorized_keys` was `ftp ftp`, so sshd's StrictModes accepted the connection and refused the key | `chown 0:0` on the installed trees, and again in `20onboard-ssh` before sshd starts |

The general lesson, and the reason `vm-check.sh` exists: **an artifact probe cannot see any of these.** G3 as it
was written ("the zdrv carries MOK-signed `amdxdna` with the right vermagic") was green on an image where no
module could load at all. Reachability, ownership and module resolution are properties of the *running*
system.

**The board's finding, 2026-11-02 14:5x — the sixth defect, and the first one only hardware could show:** with
the stick in the Max+ 395 the chain `shim -> Canonical GRUB` loaded and then GRUB refused the kernel:
`error: bad shim lock signature / you need to load the kernel first`. The Puppy `vmlinuz` came straight out of
`make bindeb-pkg` and carried **no signature at all** (`sbverify --list` → "No signature table present"),
while every out-of-tree module was properly MOK-signed — which is what G3 checked, so G3 was green. Secure
Boot is ON on the board and **every VM row in this document ran with it off**; a `-kernel` QEMU boot verifies
nothing, so no row in this lane could have seen it. Three fixes, none of which lowers a gate:

| # | fix | where |
|---|---|---|
| a | the kernel image is signed with the machine's enrolled MOK before packaging; the key is never printed or copied off the box, and re-signing is a no-op | `sign-vmlinuz.sh`, and `mk-huge-kernel.sh`/`mkimage.sh` refuse an unsigned image |
| b | G3 gains a **kernel-signature row**: unsigned `/vmlinuz` is a FAIL, and `--cert <mok.pem>` requires it to verify against the enrolled certificate. Measured on the pre-fix image: exactly one FAIL, the kernel; everything else in G3 stayed green | `g3-probe.sh` |
| c | the VM boots the disk image with **Secure Boot on**: OVMF's secboot firmware with this machine's certificate enrolled in `db`. An unsigned kernel now stops at GRUB in the VM with the board's own message (`FAIL sb chain … you need to load the kernel first`), and the guest confirms the other way with `[0.000000] Secure boot enabled` + `integrity: Loaded X.509 cert '1bit-MONSTER: strixhalo Secure Boot Kernel'` | `sb-vars.sh`, `vm-check.sh` |

**Still not run, on purpose:** G4 on the board (the VM half is now green — see above), G5/G6 on hardware
(HRX chat on the 8060S, the NPU through Lemonade, `1bit-release-check`), U2's desktop on the console/JetKVM,
U5's device rows, U6 (installer to a second disk), U8 (boot time, idle RAM, recording), and the save layer
across a real reboot (G7) — that one is testable in a VM next and is the last software row left.

**The board's finding, 2026-11-02 16:3x — the seventh defect, and the one that took the machine down:** with the
cycle-10 stick in the Max+ 395 the chain `shim -> Canonical GRUB -> our signed kernel` all worked and then, at
**26.7 s**, the board said

```
Kernel panic - not syncing: Attempted to kill init! exitcode=0x00000000
Comm: sh   PID 1
```

A **clean** exit code on PID 1 is the whole diagnosis: Puppy's init script EXITED. Its only failure path is

```
exec /bin/sh >/dev/console 2>&1          init:164, init:174, init:1519
```

and a shell whose console reads fail (or return EOF) exits 0 — so a boot that could not settle on its medium
took the machine down, with no log and no console. The JetKVM screen at ~3 s showed three candidates for that
medium in the same box: the JetKVM's own USB mass storage (3-2:1.6), the Gigastone stick (4-1), and the internal
NVMe carrying the Ubuntu ESP. Cycle 10's grub entries said `pmedia=usbhd psubdir=1bit` and nothing else, i.e.
init was left to scan them all.

Two fixes, and neither is a gate lowered:

| # | fix | where |
|---|---|---|
| a | **the boot media is pinned** on the kernel command line: `pupsfs=1BIT-DATA` sets Puppy's `PDRV`, and with `PDRV` set the init does not scan any drive at all (`LOOK_PUP` stays empty). `pupsfs`/`pdrv`/`pdev1` are init's own parameters, and the pinned form also gets a 15 s `blkid` retry, which covers USB enumeration. All three grub entries carry it | `mkimage.sh` |
| b | **init can no longer exit**: `one_rescue` is installed with `trap 'one_rescue $?' EXIT`, and init's three `exec /bin/sh` failure paths are replaced by it — exec runs no EXIT trap, so the trap alone would not have covered the paths that actually fire. On failure it writes `/init-fail.log` (`/proc/cmdline`, `/tmp/bootinit.log`, `dmesg`) onto the `1BIT-DATA` partition and keeps a shell on the console for ever, re-spawning it if it exits | `patch-init.sh`, run by `build.sh` before `3builddistro` and verified again on the packed `initrd.gz`; `mkimage.sh` refuses an image without it |

**Measured in QEMU, on the board's device set** (`os/v3/puppy/panic-repro.sh`: the artifact as a USB stick, a
second disk with an ESP **and** an ext4 root like the board's NVMe, an extra USB mass-storage device like the
JetKVM, Secure Boot ON from this machine's enrolled store — and `--no-media`, which pins a label that does not
exist so the failure happens on purpose):

| image | configuration | transcript |
|---|---|---|
| cycle 10 (no pin, no guard) | pin broken | `Finding puppy main sfs file. failed` → `Partition NO-SUCH-LABEL not found.` → `Error is too critical, dropping out to console...` → a shell at `~ #`. That is the board's 26.7 s moment: on the board that shell's reads returned and it exited 0, so PID 1 died. In QEMU the same shell blocks on read, which is why this half alone does not panic here |
| **cycle 11** (pin + guard) | pin broken | **no panic**: `*** 1bit: init stopped (status 1) - logging, and this console stays`, `*** 1bit: log written to 1BIT-DATA:/init-fail.log`, console alive — and the file is really on the partition: **41,409 B** carrying the command line, `/tmp/bootinit.log` and `dmesg`, read back off the image afterwards |
| **cycle 11** | normal, board's device set | boots to the appliance: all four SFS layered, `Loading user-selected modules... ec_su_axb35`, `Setting up services` — with `pupsfs=1BIT-DATA` on the command line, i.e. no drive scan at all |

So the two halves of the board's failure are now covered from both sides: the pin removes the *need* to scan
(which is what made the medium ambiguous), and the guard makes any remaining init failure a logged, live
console instead of a machine-wide panic. `vm-check.sh` gained two rows that read the **running** boot rather
than the tree — `boot pin` (`pupsfs=1BIT-DATA` on `/proc/cmdline`) and `init guard` (`/initrd/init`, the file
that actually ran, carries `one_rescue`, the trap and all three guarded paths) — because an artifact probe
cannot see either: G3 was green on both the unsigned kernel and this.

## The gates this evaluation would accept, and which of them are green (as first written)

| gate | what | status |
|---|---|---|
| G1 | a resolute64 Puppy image builds from our recipe with **our** kernel tarball in `kernel-kit/output` | **green**: `build.sh` alone exits 0 and writes `upup64r-26.04.iso` **1,317,011,456 B**, sha256 `41e23602…6f857e` — our `vmlinuz` (`7.3.0-rc4-1bit-perfopt+`), a 127-entry initrd, four SFS, early AMD microcode, `isolinux` + an EFI `efi.img` with shim/grub/MokManager |
| G2 | that ISO boots in QEMU under our kernel, headless (`pfix=nox`), to a shell | **green now** — the fragment kernel layers the medium and reaches Puppy's userspace (see the section above); this row records what it was before the fragment |: our initrd on our kernel stops at `Finding puppy main sfs file. failed` with `HAVE_PARTS=''` in `/tmp/bootinit.log` — no partitions seen, because the medium/layer drivers are `=m` ([Blocker 1](#blocker-1-the-kernel-config)). The fix is `kernel-puppy.fragment` + a kernel rebuild |
| G3 | `uname -r` is our kernel and **that kernel image is signed**, and `amdgpu`, `amdxdna`, `ec_su_axb35` are present and signed (VM: modules present; board: devices) | **green in the artifact** since the signing fix: `/vmlinuz` verifies against the enrolled MOK, zdrv carries `usr/lib/modules/<kver>` with 3780 modules, both OOT modules MOK-signed with the right vermagic, `amdgpu`/`amdnpu` firmware present. The board half (devices binding) is still U5's |
| G4 | Secure Boot with our shim/MOK chain, kernel verifies, MOK enrolment is one console step | **green in the VM**: the disk image boots with OVMF secboot + this machine's certificate in `db`, GRUB accepts the signed kernel, and the guest reports `Secure boot enabled` with our certificate loaded into the integrity keyring. The board's version of this row is what failed on 2 Nov and is now unblocked, not yet re-run |
| G5 | `lemond` + `1bit serve` run unchanged from the c18 tarball | **blocked by glibc on 24.04** (measured 2.43 vs 2.39); needs the resolute base |
| G6 | `1bit-release-check` green as far as a VM allows | **green in the VM (cycle 12)**: the tool now ships in the appliance (`/usr/local/bin/1bit-release-check`, `/bin/1bit-release-check`) with `tools/seal_manifest.py`, the release **public** seal key and `etc/1bit/onebit-models.txt`, so `sealed`/`sealed-sig` run instead of skipping: `RESULT: PASS (19 checks, no failures)` with only the GPU/NPU and `sha256sums` rows SKIPping. The board's version of this row is the one that must be 0 FAIL / 0 SKIP (U4) |
| G7 | a save layer persists across a reboot (frugal install on a qcow2, then reboot) | not run |
| G8 | image size, boot time, idle RAM, against the c18 initramfs | size measured (ISO 1.32 GB, rootfs 1.5 GB, frugal payload 1.27 GB of SFS); boot time and RAM pending G2 |

## Cycle 12 (2026-11-02 18–19h): the board's five FAILs, closed and measured

The cycle-10 stick booted on the Max+ 395 and `puppy-check.sh` came back 20 PASS / 5 FAIL. Cycle 12 is
cycle 11 plus one change per FAIL, and one row in the evaluation is now a measurement rather than a plan:

* **U1** — `os/v3/puppy/shell-login.sh` makes root, `spot` and every uid 1000–65533 log in to `/bin/bash`
  (service accounts keep `nologin`/`false`; the first version gave `daemon` and `nobody` a login shell, which
  is not what the row means). `build.sh` asserts `usr/bin/bash` in the packed image.
* **U3** — `/usr/local/bin/lemonade` → `/opt/1bit-package/bin/lemonade`; `u3-service` was a false FAIL in
  the board's copy of the check (it looked for the `lemond-onebit` launcher, not `lemond`).
* **U4** — `1bit-release-check` ships, with the layout it needs: it *finds* the Lemonade package
  (`/opt/lemonade` on os/mini, `/opt/1bit-package` here), prefers a **downloaded** model for the chat rows,
  never lets an NPU entry answer `chat-hrx`, and looks for the licence files where this package keeps them.
  The sealed NPU sets are copied onto `/data/npu/kernels` at boot and their manifests verify against the
  release public key, so `sealed-sig` runs.
* **U6** — `1bit-install` ships (and `sfdisk`, which the built rootfs did not have: the 26.04 `fdisk`
  package carries it and the file needed a spec row of its own).
* **Item 6** — the shipped catalogue carries **23 HRX + 4 NPU** onebit entries; `/v1/models` lists
  **downloaded** models only (`show_all=false`), which is why the board saw one. The image now ships an HRX
  GGUF as well as the NPU set, and a `.q4nx` directory is registered under the catalogue's own
  `…-NPU-1bit` id instead of the HRX id it used to wear.
* **Item 7** — headless is the default grub entry (`pfix=copy,nox`): no X, and tty1 is a login prompt from
  Puppy's own `plogin` → `getty -n -l /bin/autologin`. The desktop is entry 1.

On the artifact (`~/wt/puppy-out-c12/1bit-os-puppy.img`, sha256 `5d30b75f…`, 9,128,902,656 B): G2 green,
G3 green with `--cert --require-mok`, G6 green in the VM, and `vm-check.sh` **23 PASS / 0 FAIL** under
Secure Boot — including the two rows this cycle added to the appliance itself (`U1 login shell`,
`U2 headless`), the CLI, the release check and the installer.

### Cycle 13: the headless default reaches tty1 unattended

Cycle 12's default entry was headless, but the console never got to the prompt: `rc.country` asks its three
questions there on every `pfix=nox` boot (`init` creates `/tmp/bootcnt.txt`, which sets `ASKKMAP` and
`ASKLANG`; a missing `/etc/keymap` asks for the layout; a missing `en_US.utf8` locale sets `ASKCLOCK=yes`,
and `timezone-set cli` then asks for the timezone on `/dev/console`). Cycle 13 turns the two flags off in
`rc.country` (the build refuses a packed image that still asks) and ships `/etc/keymap` = `us` and
`/etc/localtime` → UTC; the desktop entry keeps `quicksetup` on its first X start. The row that proves it
reads tty1's own VGA text buffer through the image's new `/usr/local/bin/1bit-tty1`, not ttyS0:

```
PASS U2 tty1      ResolutePup64 | Linux 7.3.0-rc4-1bit-perfopt+ [x86_64 arch] | root@puppypc23253:~$ | tty1-no-wizard
PASS U2 headless  pfix=copy,nox | no-X-server
```

### Cycle 14: the board boot that reached the engine, and the six reds it left

Cycle 13 passed 25 of 29 acceptance rows on the Max+ 395 (NPU chat through Lemonade at 93.2/95.3 tok/s,
`lemonade bench` decode 88.5 tok/s, `ec_su_axb35` loaded, Secure Boot on, ssh with the owner's key) and left
six. All six are closed in the image, and each one was a measurement rather than a guess:

| row | the cause, measured | the fix |
|---|---|---|
| `hrx-device` (no `HRX0` at all), `chat-hrx` | **nothing opened `libhsa-runtime64.so.1`.** `ONEBIT_HRX_LIBHSA` is a CMake *compile definition* (`CMakeLists.txt:279`), not a variable, so the line in `/etc/1bit/libhsa.env` was inert; the engine resolves `--hrx-libhsa` → the path baked at build time → a recursive scan of `/opt/rocm-therock` (`app/serve.cpp:201`), and the baked path is the release builder's TheRock. Second defect on the same path: the pinned libhsa is `RUNPATH=$ORIGIN` with four `librocm_sysdeps_*` + `librocprofiler-register` NEEDED entries, and the image carried the one file (`ldd`: 4 `not found`) | `LIBHSA_DIR` installs the whole set at `/opt/rocm-therock/lib`, `/opt/1bit/libhsa` symlinks to it, `libhsa.env` exports the trio and `LD_LIBRARY_PATH`, and the build refuses an incomplete or unloadable set |
| boot cmdline | the appliance booted `amdgpu cwsr_enable=1` with no TTM parameters, while the install the image was validated against boots `amdgpu.cwsr_enable=0 ttm.pages_limit=31457280 ttm.page_pool_size=31457280` — and an APU serves models out of GTT, whose ceiling is that limit | every grub entry carries all three; row `boot gtt` reads them off the running boot |
| `u2-console` | inittab's tty1 entry is `plogin`, whose tail is `exec getty -n -l /bin/autologin … tty1`; on that boot nothing ended up on tty1, while the same image gave a prompt in QEMU | inittab's tty1 entry is the appliance's own `1bit-console`: that last step on a headless boot, `plogin` on every other |
| `gguf` | not a registration bug: `05onboard-data` registers a `.gguf` under the catalogue's lower-cased id and `release-check.sh` compared the file's own spelling, case-sensitively | the row compares case-insensitively |
| `fans` SKIP | `os/mini/board.sh` was never installed in the appliance | `/usr/sbin/1bit-board` + `26onboard-board` |
| boot time on the console | `pfix=copy` copies every SFS into RAM at the medium's speed before userspace exists, silently | two console lines from the init patch with the guest's own uptime: what it waits for, and how long it took |

Plus the owner's `1bit OS - install to a disk (1bit.installer=1)` entry: the same system, a guarded console
installer (disks with model and size, the boot disk refused by device *and* by a `1BIT-DATA` partition on
it, the whole device name typed twice), and `1bit-install` doing the writing.

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
- **boot past `Finding puppy main sfs file`** (G2): the ISO exists and its init runs, but the kernel has
  no built-in medium driver, so the init sees no partitions. This is the one item a kernel rebuild in
  `docs/kernel.md` §1 closes — and that build is deliberately not done on this box.
- **project health and licensing** beyond the facts stated: woof-CE builds five targets on CI; the
  recipes are GPL-2.0-or-later with the built distro's own licences; ResolutePup64 is community-built
  and its release notes are the only English documentation of the recipe.
