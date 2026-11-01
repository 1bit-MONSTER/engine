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
| `mk-huge-kernel.sh` | our kernel as woof-CE's kernel package (`huge-<kver>.tar.*`: `vmlinuz-<kver>` + a zdrv SFS with the module tree, the firmware and a marker) |
| `build.sh` | woof-CE's CI recipe in a throwaway container, with our kernel and the build root on a host filesystem |

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
#    (24 symbols today) rather than a hand-written list that drifts:
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

# 4. our kernel as woof-CE's kernel package:
os/v3/puppy/mk-huge-kernel.sh <stage> <out-dir>
#    writes <out-dir>/huge-<kver>.tar.zst  (vmlinuz + kernel-modules-zdrv-<kver>.sfs)
#    and prints the module count, the sizes and each out-of-tree module's signer. It refuses:
#      - a vmlinuz whose embedded release string is not <kver> (file(1) reads the setup header),
#      - a module built for another kernel (modinfo -F vermagic),
#      - an unsigned module (modinfo -F signer) - Secure Boot refuses those at load.

# 5. rebuild the Puppy with it (about 15-25 min in the container, mostly downloads the first time).
#    The build root defaults to $HOME/wt/1bit-puppy-build because it has to be on a filesystem that
#    allows device nodes: 3builddistro mounts an overlay for every petbuild chroot, and debootstrap needs
#    a usable device node in its target - /tmp on this box is tmpfs,nodev and fails both (measured, traps
#    6 and 7 in the evaluation). BUILD_ROOT=<dir> overrides it, and a nodev/noexec root is refused
#    before the downloads start.
os/v3/puppy/build.sh resolute64 <kernel-out-dir> <puppy-out-dir> [woof-CE-dir]
#    (woof-CE defaults to $HOME/wt/woof-CE - not /tmp, which a reboot empties; the 4th argument is the
#    clone, and the build copies it into the build root itself)
#    build.sh copies <out-dir>/huge-*.tar.* into woof-out_*/kernel-kit/output/ before 3builddistro,
#    which is what makes the image carry OUR kernel. Empty that directory and woof-CE silently
#    downloads a stock Ubuntu kernel instead - the trap the evaluation paid for.

# 6. re-gate: G2 (boots on our kernel), G3 (amdgpu/amdxdna/ec_su_axb35 present and signed; devices on
#    the board) and G4 (Secure Boot with the shim/MOK chain), then the release check inside the VM.
```

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
