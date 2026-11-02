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

## The gates this evaluation would accept, and which of them are green

| gate | what | status |
|---|---|---|
| G1 | a resolute64 Puppy image builds from our recipe with **our** kernel tarball in `kernel-kit/output` | **green**: `build.sh` alone exits 0 and writes `upup64r-26.04.iso` **1,317,011,456 B**, sha256 `41e23602…6f857e` — our `vmlinuz` (`7.3.0-rc4-1bit-perfopt+`), a 127-entry initrd, four SFS, early AMD microcode, `isolinux` + an EFI `efi.img` with shim/grub/MokManager |
| G2 | that ISO boots in QEMU under our kernel, headless (`pfix=nox`), to a shell | **red, measured on both built ISOs**: our initrd on our kernel stops at `Finding puppy main sfs file. failed` with `HAVE_PARTS=''` in `/tmp/bootinit.log` — no partitions seen, because the medium/layer drivers are `=m` ([Blocker 1](#blocker-1-the-kernel-config)). The fix is `kernel-puppy.fragment` + a kernel rebuild |
| G3 | `uname -r` is our kernel, and `amdgpu`, `amdxdna`, `ec_su_axb35` are present and signed (VM: modules present; board: devices) | kernel half measured (tarball 3819 modules, both OOT modules signed), boot half blocked by G2 |
| G4 | Secure Boot with our shim/MOK chain, kernel verifies, MOK enrolment is one console step | not run; the chain is today's and the ISO carries a shim/MokManager |
| G5 | `lemond` + `1bit serve` run unchanged from the c18 tarball | **blocked by glibc on 24.04** (measured 2.43 vs 2.39); needs the resolute base |
| G6 | `1bit-release-check` green as far as a VM allows | not run (needs G2+G5) |
| G7 | a save layer persists across a reboot (frugal install on a qcow2, then reboot) | not run |
| G8 | image size, boot time, idle RAM, against the c18 initramfs | size measured (ISO 1.32 GB, rootfs 1.5 GB, frugal payload 1.27 GB of SFS); boot time and RAM pending G2 |

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
