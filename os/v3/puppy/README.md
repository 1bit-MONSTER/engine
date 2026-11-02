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
# `os/v3/puppy` — the Puppy recipe for 1bit OS v3

The owner's direction (2026-11-01) is a Puppy-based v3: the same image is the live/try stick and the
installed system (a frugal install on the NVMe with a persistent save layer), headless, with **our**
kernel, **our** drivers and **no distribution kernel anywhere in it**. The evaluation, the numbers and
the gates are in [docs/os-v3-live-puppy-eval.md](../../../docs/os-v3-live-puppy-eval.md); this page is
how to build it. The Ubuntu Server `autoinstall` design stays the documented fallback.

| file | what it is |
|---|---|
| `kernel-puppy.fragment` | the kernel config delta: the medium/layer drivers Puppy's initrd needs **built in**. A fragment the kernel build appends, not a patch |
| `sign-vmlinuz.sh` | signs the kernel image with this machine's enrolled MOK, so Secure Boot's GRUB will load it. The key never leaves the box: a copy goes into a root-owned 0700 dir, is converted (it is an OpenSSH key) and is proved against the certificate before anything is signed. Idempotent |
| `sb-vars.sh` | OVMF's variable store with our certificate enrolled in `db`, so a QEMU boot verifies the appliance the way the board does |
| `mk-huge-kernel.sh` | our kernel as woof-CE's kernel package (`huge-<kver>.tar.*`: `vmlinuz-<kver>` + a zdrv SFS with the module tree, the firmware and a marker). **Refuses an unsigned kernel image** |
| `build.sh` | woof-CE's CI recipe in a throwaway container, with our kernel and the build root on a host filesystem |
| `g2-probe.sh` | gate G2: boot the built ISO headless in QEMU and keep the init's transcript, so "it boots on our kernel" is a measurement |
| `g3-probe.sh` | gate G3, artifact half: the ISO carries our kernel, **that kernel image is signed**, its zdrv carries the module tree, and `amdxdna`/`ec_su_axb35` are built for that kernel and signed. `--cert <mok.pem>` requires the image's signature to be the enrolled MOK's; `--require-mok` for the release gate |
| `mkimage.sh` | the shipped artifact: a GPT disk image (ESP with shim → grub, `1BIT-DATA` ext4 with the frugal install at `/1bit` and os/mini's data directories), so `os/mini/prepare-stick.sh` and the USB write work unchanged. The ISO stays a side artifact |
| `vm-check.sh` | the TPUPPY-RELEASE rows that need no board, in QEMU over a serial root shell, **with Secure Boot on for the disk image** (`sb-vars.sh`'s store, `OVMF_CODE_4M.secboot.fd`, `smm=on`): bash + `sudo` + `apt install` (U1), lemond up with its key and `/api/v1/models` from inside the guest and over the forwarded port (U3), `1bit` on PATH and the EC driver in `ADDLIST` (U4). Hardware rows print as SKIP with the reason |
| `packages-extra` | package rows the acceptance needs and woof-CE's set does not have (`sudo`, `apt`, `mokutil`, `openssh-server`), with the measurement that justifies each |
| `spec-insert-rows.sh` | puts those rows inside `PKGS_SPECS_TABLE`, before its closing quote; idempotent |
| `embed/` | the files PAYLOAD=... drops into the rootfs (into `rootfs-skeleton`, which is what `3builddistro` copies): `05onboard-data` (mounts `1BIT-DATA` at `/data` and wires the contract), `10lemonade` (the server), `20onboard-ssh` + `root/.ssh/authorized_keys` (U7: host keys, StrictModes' two modes, sshd), `21serial-console` (a root shell on ttyS0, only on `1bit.serial=1`), `etc/1bit/libhsa.env` (the no-SVM trio, engine#715), `etc/profile.d/90onboard-limits` (memlock) |

## A kernel bump, end to end

The kernel is ours, so a kernel move is this list. Nothing in Puppy is patched for it.

```sh
# 0. what the fragment must say now: each line carries our value today, so a symbol that becomes =y
#    upstream is visible in the diff, and the comments name the failure it prevents
less os/v3/puppy/kernel-puppy.fragment

# 1. build the kernel with the fragment appended (the weekly kernel build already takes a fragment):
#    the .config gets the fragment, then `make olddefconfig`, then the usual clang/LLVM=1 build, then the
#    release kernel packages (linux-image-<kver>.deb + its headers), as docs/kernel.md describes.
#    Verify the deltas actually landed before packaging. A fragment symbol that this version's Kconfig
#    does not know is silently dropped by olddefconfig, so check every line of the fragment itself
#    (26 symbols today) rather than a hand-written list that drifts:
missing=0
for sym in $(grep -oE '^CONFIG_[A-Z0-9_]+=y' os/v3/puppy/kernel-puppy.fragment | cut -d= -f1); do
    grep -qx "$sym=y" /boot/config-<kver> || { echo "MISSING $sym"; missing=$((missing+1)); }
done
echo "$missing missing (0 is what a gate wants)"
#    One of them, CONFIG_DRM_SIMPLEDRM, is a *driver* choice: it is what gives tty1 a console on a box
#    whose GPU driver is not up yet, and in a VM. If it is ever dropped, the install screen has no screen.

# 2. build the out-of-tree modules against that kernel and sign them with the MOK (docs/kernel.md):
#    amdxdna (0.18.0, third_party/xdna-driver's pin) and ec_su_axb35. They land in
#    /lib/modules/<kver>/updates/ and mk-huge-kernel.sh refuses them otherwise.

# 3. stage the kernel tree the way os/v2/mkdebs.sh takes it:
#      <stage>/boot/{vmlinuz,config}-<kver>   and   <stage>/lib/modules/<kver>/
#    (image deb unpacked + the whole /lib/modules tree, updates/ included)

# 3b. SIGN the kernel image, before it is packaged. `make bindeb-pkg` output is unsigned, and with Secure
#    Boot on the chain stops at grub with `error: bad shim lock signature / you need to load the kernel
#    first` - which is exactly what the 2 Nov board boot did, on a stick whose modules were all signed and
#    whose every VM row passed, because a `-kernel` QEMU boot verifies nothing. Signing uses the same MOK
#    the box's own Ubuntu kernel carries, on the box the key lives on, and the key is never printed or
#    copied off it (a 0700 temp copy is converted, proved against the certificate, and wiped):
os/v3/puppy/sign-vmlinuz.sh <stage>/boot/vmlinuz-<kver>
#    prints the release string, the signer subject and the sha256 of the signed image; re-running on an
#    already-signed image is a no-op (sbsign would otherwise stack a second signature).

# 4. our kernel as woof-CE's kernel package:
os/v3/puppy/mk-huge-kernel.sh <stage> <out-dir>
#    writes <out-dir>/huge-<kver>.tar.zst  (vmlinuz + kernel-modules-zdrv-<kver>.sfs)
#    and prints the module count, the sizes and each out-of-tree module's signer. It refuses:
#      - a vmlinuz whose embedded release string is not <kver> (file(1) reads the setup header),
#      - a module built for another kernel (modinfo -F vermagic),
#      - an unsigned module (modinfo -F signer) - Secure Boot refuses those at load,
#      - an unsigned KERNEL IMAGE (sbverify) - step 3b, and grub refuses it under Secure Boot.

# 5. rebuild the Puppy with it (about 15-25 min in the container, mostly downloads the first time).
#    The build root defaults to $HOME/wt/1bit-puppy-build because it has to be on a filesystem that
#    allows device nodes: 3builddistro mounts an overlay for every petbuild chroot, and debootstrap needs
#    a usable device node in its target - /tmp on this box is tmpfs,nodev and fails both (measured, traps
#    6 and 7 in the evaluation). BUILD_ROOT=<dir> overrides it, and a nodev/noexec root is refused
#    before the downloads start.
os/v3/puppy/build.sh resolute64 <kernel-out-dir> <puppy-out-dir> [woof-CE-dir]
#    (woof-CE defaults to $HOME/wt/woof-CE - not /tmp, which a reboot empties; the 4th argument is the
#    clone, and the build copies it into the build root itself)
#    A new BUILD_ROOT downloads 2.6 GB again, and one mirror hiccup over a thousand files fails the
#    build (measured: one package, all four mirrors, and the URL was fine minutes later). If the box has
#    built before, seed the cache instead: sudo cp -a <old-build-root>/local-repositories <new-build-root>/
#    <puppy-out-dir> gets the ISO and the four SFS - the SFS files are the frugal install.
#    PAYLOAD=<unpacked lemonade-onebit-<tag>> embeds the release package (U3/U4): copied to
#    /opt/1bit-package with /etc/init.d/10lemonade starting it at boot, and the build refuses an image
#    whose SFS set lacks the engine, the launcher, /usr/local/bin/1bit, the service, the owner's key or
#    the ADDLIST line in the packed /etc/rc.d/MODULESCONFIG. It writes into rootfs-skeleton, because
#    3builddistro deletes and rebuilds sandbox3/rootfs-complete from it (measured: writing into the
#    sandbox copy is silently discarded, image still green).
#    build.sh copies <out-dir>/huge-*.tar.* into woof-out_*/kernel-kit/output/ before 3builddistro,
#    which is what makes the image carry OUR kernel. Empty that directory and woof-CE silently
#    downloads a stock Ubuntu kernel instead - the trap the evaluation paid for.

# 6. re-gate: G2 (boots on our kernel), G3 (the kernel image is signed, amdgpu/amdxdna/ec_su_axb35
#    present and signed; devices on the board) and G4 (Secure Boot with the shim/MOK chain), then the
#    release check inside the VM.
#    g2-probe.sh <iso> <work> and g3-probe.sh <iso> <work> [--kver <kver>] [--cert <mok.pem>]
#    [--require-mok] read the image; vm-check.sh <iso|img> <work> boots it and answers the U1/U3/U4 rows
#    over a serial shell. For the .img it also builds sb-vars.sh's variable store and boots with Secure
#    Boot ON (OVMF_CODE_4M.secboot.fd, smm=on), so an unsigned kernel is caught here and not on the board;
#    the `sb state` row is the guest confirming it. A failed Secure Boot boot prints `FAIL sb chain` with
#    grub's own message. The .iso cannot carry that row: a -kernel boot verifies nothing.
os/v3/puppy/sb-vars.sh <work>/vars.fd    # what vm-check.sh runs for itself
```

## What runs in the image at boot

PID 1 is not systemd: `/etc/rc.d/rc.services` runs every **executable** script in `/etc/init.d` with
`start` and every script there again with `stop` at shutdown (a built resolute64 rootfs ships 295 files
under `/lib/systemd/system` and nothing reads them). So a service is executable-or-invisible, and the three
this recipe ships are:

| service | what it does |
|---|---|
| `05onboard-data` | finds `1BIT-DATA` (`1bit.data=` on the boot line, else the label), mounts it at `/data`, creates the contract dirs, points `/etc/1bit/lemond.conf` at `/data/lemonade/lemond.conf`, copies `/data/ssh/authorized_keys` and links the host key from `/data/ssh`, and registers `/data/models` in Lemonade's `user_models.json` |
| `10lemonade` | starts `/opt/1bit-package/lemond-onebit`, mints `LEMONADE_API_KEY` into `/etc/1bit/lemond.conf` (mode 600) on first boot, binds `0.0.0.0:13305`, logs to `/var/log/lemond-onebit.log` |
| `20onboard-ssh` | `ssh-keygen -A`, `/root/.ssh` 700 + `authorized_keys` 600, then sshd. An image with no key provisions no listener |
| `21serial-console` | a root shell on `/dev/ttyS0`, only when the command line carries `1bit.serial=1` (that shell has no password, so it is opt-in) |

Two facts about the console that cost VM runs: the boot transcript is written to ttyS0 but the console
Puppy leaves you with is **tty1**, so serial input goes nowhere without `21serial-console`; and `pfix=nox`
is what makes `rc.country` ask for a keyboard layout and timezone (`init` creates `/tmp/bootcnt.txt`), so
the appliance's normal boot asks nothing and those dialogs are a debug-mode artefact.

Two naming rules make this work, and both are silent when broken:

- **the kver comes from the tarball name.** woof-CE parses it out of `huge-<kver>.tar.*`, and `3builddistro`
  then renames `vmlinuz-<kver>` to `vmlinuz` and `kernel-modules*<kver>*` to the zdrv. `mk-huge-kernel.sh`
  writes exactly that name; renaming the file breaks the pairing with no error.
- **the zdrv is where the modules live at boot**, mounted and unioned over `/` by Puppy's initrd, which
  is why the firmware sits inside it at `/lib/firmware/...` and why `modules.dep` is generated at build
  time (`depmod -b`) instead of on the machine.

## Why the fragment exists

Measured, 2026-11-01: on our kernel as it is built today, Puppy's initrd stops 2.1 s into the boot with
`*** /puppy_upup64n_24.04.sfs not found`, because the initrd mounts the medium and layers the save file
**before** any module can be loaded — and our kernel has `BLK_DEV_SR`, `ISO9660_FS`, `SQUASHFS`,
`OVERLAY_FS`, `BLK_DEV_LOOP`, `EFIVAR_FS`, `FAT_FS`/`VFAT_FS`, `EXT4_FS`, `USB_STORAGE`, `HID_GENERIC`
and friends as `=m`, the Ubuntu shape. Puppy's own stock kernel from kernel-kit has every one of them
`=y`. The fragment is that list with our current value beside each line.

## What the medium is, in one paragraph

woof-CE produces an ISO of `puppy_<prefix>_<version>.sfs` (the OS, read-only), `zdrv` (our kernel's
modules), `fdrv` (firmware) and a bootloader. Written to a stick it is the live/try medium; copied to a
partition it is the frugal install; either way a *save layer* (a file or a folder, Puppy's pupmode
5/12/13) is what persists, which is #706's answer. The ESP chain is ours — shim → our MOK-signed grub →
our MOK-signed kernel → our modules — so Secure Boot and the firmware boot entry (#709) are the same
work as in the Ubuntu plan, not new work.

## Licence

These scripts are Apache-2.0, like the rest of this repository. What they *package* is not: our kernel
and modules are signed with our MOK, the firmware belongs to its vendors under the licences
`os/LICENSES.md` records, and Puppy's own tree is GPL-2.0-or-later with each recipe's licences carried
into the built image.
