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
# 1bit OS v3 — the installed machine is an Ubuntu 26.04 Server

This is the design for engine#712. Today one image serves both roles: the live stick and the disk it
installs to are the same BusyBox initramfs, so the installed machine has no package manager, no users,
no updates and an empty efivarfs (engine#712, engine#709). v3 keeps that image as the *live* medium and
makes the **installed** machine a real OS: Ubuntu 26.04 LTS Server, installed unattended by `autoinstall`
from the same stick, carrying our kernel, our drivers, our runtime, the engine, Lemonade and the
appliance units as packages that `apt` updates.

Nothing here is built yet: `os/v3` does not exist in the tree, `os/mini` keeps shipping in v2026.44, and
engine#712 says v3 is the next release, not the Sunday one. What *is* measured is in
[What is measured now](#what-is-measured-now) — most of it on the Strix Halo box this design was written
on, which is itself the reference install: it runs Ubuntu 26.04.1 with our kernel, our out-of-tree
modules, TheRock's ROCm, the distro's XRT, the engine and Lemonade, and it serves HRX and the NPU.

## What changes, and what stays

| | today (`os/mini`) | v2 (`os/v2`, in tree, not shipped) | v3 |
|---|---|---|---|
| live medium | BusyBox initramfs, everything in RAM | Ubuntu rootfs squashfs + our debs | **unchanged**: the `os/mini` image is the try-it medium |
| installed system | the same initramfs on the ESP + a `/data` partition | squashfs unpacked to ext4, our debs preinstalled | **Ubuntu 26.04 Server, installed by subiquity `autoinstall`** |
| updates | reflash the stick | `apt` from `1bit.gg/apt` | `apt` from `1bit.gg/apt`, unattended security updates on |
| users | root only, keys from `/data/ssh` | root only, keys from `/data` | **`1bit` with sudo**, keys carried from the stick, root locked |
| firmware entry | none (engine#709) | `efibootmgr -c` when it exists | **created, ordered, stale entries removed** (engine#709) |
| efivarfs | absent (module never staged, engine#709) | present (`CONFIG_EFIVAR_FS=m`, systemd loads it) | present, and a `release-check` row for it |
| console | BusyBox shell + `1bit-menu` | `1bit-menu` on tty1, autologin | `1bit-menu` on tty1 as a unit, ssh and ttyS0 stay plain |
| headless | yes, but no server software | yes | **yes, and it is a server**: sshd, apt, users, `python3`, journald, `unattended-upgrades` |

The live side deliberately does not move. `os/mini` has passed the QEMU A/R/I/B gate and the owner's
board test for v2026.44; rewriting the try-it medium in the same release that rewrites the installed
machine would put both at risk and prove less. The stick gains one partition and one menu entry.

## The stick

Three partitions, the same shape as v2's medium (`os/v2/mklive.sh`) so `1bit-install`'s refusals, the
`prepare-stick.sh` checks and the gates keep working:

| # | size | fs | label | what |
|---|---|---|---|---|
| 1 | 512 MiB | FAT32 | `1BIT-OS` | shim, our MOK-signed grub, `1bit-os.efi` (the live appliance, `os/mini` unchanged), `grub.cfg` |
| 2 | `ISO size + 1 GiB` | ext4 | the server ISO's own volume label | the Ubuntu 26.04 Server ISO unpacked, plus `/1bit/` — the seed, our signed repo, the MOK, the install log |
| 3 | the rest | ext4 | `1BIT-DATA` | models, Lemonade's cache and config, `hrx-jit`, NPU sets, `ssh/` — copied to the target's `/data` |

Partition 2 keeps the ISO's own volume label on purpose: that label is how casper finds its squashfs and
mounts it as `/cdrom`, which is where our seed and our repository have to be visible. Renaming it (to
`1BIT-INSTALL`, say) would mean a `findiso=`/`iso-scan` kernel option and a casper that ignores the medium
— more moving parts for a cosmetic name. `os/v3/pin.txt` records the ISO's URL, size, sha256 and volume
label next to the Ubuntu snapshot timestamp that `mkrootfs.sh` already pins, so the medium is
reproducible from that file alone.

`/1bit/` on partition 2:

```
/1bit/autoinstall.yaml      the seed (generated at stick-write time, below)
/1bit/repo/                 our signed apt repository: Release, Release.gpg, InRelease, *.deb
/1bit/keys/1bit-archive-keyring.gpg
/1bit/mok/MOK-Kernel.der    the key the kernel and the out-of-tree modules are signed with
/1bit/identity              username, password hash, hostname, target disk serial, ssh keys (root 0600)
/1bit/install.log           written by the install, on both media (#710)
```

### The grub menu

Two entries; the default is the live appliance, so a stick left in a machine still comes up as today.

```
menuentry "1bit OS (live)" {
    linuxefi /1bit-os.efi
}
menuentry "Install 1bit OS onto this machine" {
    search --no-floppy --label "<ISO label>" --set=root
    linux  /casper/vmlinuz autoinstall "ds=nocloud;s=/cdrom/1bit/" \
           --- console=tty1 console=ttyS0,115200
    initrd /casper/initrd
}
```

`autoinstall` plus a NoCloud seed in the medium's own root (`s=/cdrom/1bit/`) is subiquity's documented
unattended path; the `console=` pair is what puts the same transcript on the screen (#710) and on the
serial line the QEMU gate reads. `1bit-os.efi` stays a single-file EFI stub, so the live entry is the
byte-identical payload `os/mini/mkimage.sh` writes today.

### `prepare-stick.sh` grows an identity

The seed is per-stick, not per-release: it names the machine's user, password hash, hostname and target
disk. `prepare-stick.sh` (already the tool that writes `authorized_keys` and mints Lemonade's API key)
gains:

```sh
prepare-stick.sh /dev/sdX --ssh-pubkey ~/.ssh/id_ed25519.pub \
    --user 1bit --hostname 1bit-strixhalo --disk-serial YMA62T0KA2417407NP \
    --model ~/models/Qwen3-0.6B-Q4_K_M.gguf
```

It writes `/1bit/identity` (`0600`, root), hashes the password with `openssl passwd -6`, and renders
`/1bit/autoinstall.yaml` from the template below, substituting the identity and the disk serial.
`--disk-serial` is required for a destructive install: the storage config matches that serial, so a stick
written for one machine cannot wipe another, and a re-run finds the same disk. `--any-disk` (largest disk)
exists for the QEMU gate and prints a warning that the stick is not machine-bound.

## The installed layout

| # | size | fs | label | contents | on v2 → v3 |
|---|---|---|---|---|---|
| 1 | 1 GiB | FAT32 | `1BIT-OS` | Ubuntu's shim + grub (Microsoft- and Canonical-signed), `grub.cfg`, `/boot/efi` | created |
| 2 | `min(128 GiB, max(32 GiB, 25%))` | ext4 | `1BIT-ROOT` | the system, `/boot` (kernels and initramfs), `/var`, `/home`, an 8 GiB `/swap.img` | created |
| 3 | the rest | ext4 | `1BIT-DATA` | models, Lemonade's cache and config, `hrx-jit`, NPU sets, keys — `/data` | **adopted, never formatted** |

Numbers: 1 GiB is the Ubuntu default and the measured size of this box's ESP (1.1 GiB, 6.4 MB used, so
two kernels and their initramfs fit with room); 32 GiB is the floor because a base Ubuntu 26.04 install
plus the runtime subset measured below is about 5 GiB and `/var` grows; 128 GiB is the cap so a 2 TB disk
gives the models the disk rather than the OS. No swap *partition*: Ubuntu's own `/swap.img` on the root
filesystem keeps the partition table to what the appliance needs, and it is the image `unattended-upgrades`
and hibernate-free servers already run.

`/boot` is on the root filesystem, not its own partition: `update-initramfs` and `update-grub` then work
on any kernel package without a size limit, and the ESP only holds the boot chain (v2's `install.sh`
already does this, and its grub entry searches `1BIT-ROOT` for `/boot`).

`/data` as partition 3 is the one layout decision carried over from v2 unchanged, for the reason v2
documents: models are hundreds of GiB and must outlive the OS. Here it also carries the machine's
identity across a v2 → v3 reinstall (below).

**fstab by UUID, not by label.** v2 mounts `LABEL=1BIT-*` because its image is generic. That bit the
installer: with the stick in the machine there are two `1BIT-DATA` partitions and `findfs` may answer
with the wrong one (engine#539, measured). v3 generates the target's `/etc/fstab` at install time from the
UUIDs it just created, and uses the labels only for the installer's own discovery.

## The autoinstall

```yaml
#cloud-config
autoinstall:
  version: 1
  # Nothing is asked. The two questions that must not be silent - which disk, which user - are answered
  # by /1bit/identity at stick-write time and shown on the install screen before anything is written.
  interactive-sections: []
  refresh-installer: {update: false}          # the medium is pinned; no network fetch of subiquity
  locale: en_US.UTF-8
  keyboard: {layout: us}
  timezone: America/Halifax
  identity:
    hostname: "@HOSTNAME@"
    username: "@USER@"
    password: "@PASSWORD_HASH@"               # openssl passwd -6, generated on the build machine
  ssh:
    install-server: true
    allow-pw: false                           # keys only; the password is for sudo and the console
    authorized-keys: ["@SSH_KEY@"]            # from /data/ssh/authorized_keys on the stick
  storage:
    swap: {size: 0}
    config:
      - {type: disk, id: disk0, ptable: gpt, grub_device: true, preserve: false,
         match: {serial: "@DISK_SERIAL@"}}    # a stick cannot wipe a machine it was not written for
      - {type: partition, id: part-esp, device: disk0, number: 1, size: 1G, flag: boot, wipe: superblock}
      - {type: format, id: fmt-esp, volume: part-esp, fstype: fat32, label: 1BIT-OS}
      - {type: mount, id: mnt-esp, device: fmt-esp, path: /boot/efi}
      - {type: partition, id: part-root, device: disk0, number: 2, size: 64G, wipe: superblock}
      - {type: format, id: fmt-root, volume: part-root, fstype: ext4, label: 1BIT-ROOT}
      - {type: mount, id: mnt-root, device: fmt-root, path: /}
      # preserve: true is the v2 -> v3 upgrade path in one key: an existing 1BIT-DATA is adopted with its
      # contents, and on a fresh disk curtin creates it. No separate "upgrade" installer exists.
      - {type: partition, id: part-data, device: disk0, number: 3, size: -1, preserve: true}
      - {type: format, id: fmt-data, volume: part-data, fstype: ext4, label: 1BIT-DATA, preserve: true}
      - {type: mount, id: mnt-data, device: fmt-data, path: /data}
  apt:
    geoip: false
    preserve_sources_list: false
    primary:
      - {uri: "file:///cdrom/1bit/repo", arches: [amd64]}   # ours first: the install needs no network
    fallback: offline-install
  packages: [openssh-server, efibootmgr, mokutil, unattended-upgrades, ca-certificates, curl,
             python3-minimal, rsync, zstd, pciutils, lm-sensors, ethtool]
  early-commands:
    - /cdrom/1bit/steps plan          # the whole step list, before subiquity draws (#710)
  late-commands:
    - curtin in-target -- /cdrom/1bit/postinstall   # the steps in the next section
  error-commands:
    - /cdrom/1bit/steps failed        # last step turns red, its output and the log path stay on screen
  reporting:
    builtin: {type: print}
```

What each part is for, and what it is not:

- **`packages:` is only the archive side.** `openssh-server` and `efibootmgr`/`mokutil` are the three
  tools engine#709 found missing; `python3-minimal` turns `release-check`'s `sealed-sig` row from SKIP
  into a real check (the c18 image SKIPs it for exactly that reason, measured in the gate log);
  `unattended-upgrades` is the update path.
- **Our packages come from `/1bit/repo`, not from `packages:`**, in `postinstall`, because a deb install
  must run *after* the base is unpacked and before the first boot, and because it must be visible in the
  install screen's step list.
- **`fallback: offline-install`** is the difference between a stick that installs on a machine with no
  network and a subiquity that aborts at `apt update`. The archive is only needed for the stock
  packages above; our repo is on the medium.
- **`preserve: true` on partition 3** is the v2 → v3 path (see [Upgrade](#the-upgrade-path)). It is also
  what makes the installer idempotent: a second run over a v3 machine keeps `/data`.
- **Not used, deliberately**: `source:` (we install the stock server variant), `drivers:`/`codecs:`
  (they pull third-party repositories and prompts), `kernel-crash-dumps:`, `snaps:`, `network:`
  (netplan's DHCP default is right, and a fixed address belongs in the console menu, not the seed).

### `postinstall` — our steps

Runs in the target chroot (`curtin in-target`), so every path is the target's except `/cdrom`:

| step | command | what it writes |
|---|---|---|
| 1 | `apt-get install` our repo's debs (`1bit-kernel-<kver>`, `1bit-kernel-modules-<kver>`, `1bit-rocm`, `1bit-xrt`, `1bit-engine`, `1bit-lemonade`, `1bit-appliance`, `1bit-archive-keyring`) | `/opt/1bit`, `/opt/lemonade`, `/boot/vmlinuz-<kver>`, `/lib/modules/<kver>/updates/{amdxdna,ec_su_axb35}.ko` |
| 2 | `mokutil --test-key /cdrom/1bit/mok/MOK-Kernel.der` else `mokutil --import` (password on stdin, twice) | the MokNew efivars the next boot's MokManager reads |
| 3 | `update-initramfs -u -k <kver>`, `update-grub` | an initramfs with `amdgpu`, `amdxdna`, `efivarfs`, and one grub entry per kernel in `/boot` |
| 4 | `efibootmgr -c -d <disk> -p 1 -L "1bit OS" -l '\EFI\ubuntu\shimx64.efi'`, `-o` first, `-n` BootNext, delete entries whose loader is not on this ESP | the firmware entry engine#709 asks for |
| 5 | adopt `/data`: `chown 1bit`, `install -d models lemonade/cache lemonade/config cache/hrx-jit npu ssh`, copy `/data/ssh/host_ed25519` to `/etc/ssh/ssh_host_ed25519_key` | a machine whose SSH identity survives a reinstall |
| 6 | verify: `mount -o ro` both partitions, `sha256sum /boot/vmlinuz-<kver>`, `dpkg-query -W '1bit-*'`, `systemctl is-enabled` the unit set, `1bit-release-check` | `/data/install.log` on both media, the screen, ttyS0 |

The `1bit-appliance` postinst enables the unit set exactly as v2 does, including the measured
`ssh.socket` rule (enabling `ssh.service` as well makes a connection hang at banner exchange) and
`systemctl --root`-safe links for the chroot install.

## The install screen (engine#710)

One renderer, the same one v2's `os/mini/tui.sh` work already implements: a full-screen step list on
tty1 with a fixed log tail, one line per step that flips pending → running → done with its elapsed time,
a determinate bar with MB/s and ETA for the steps that move bytes, and a red step that stops the screen
with the failing command and its output. `[ -t 1 ]` false (the serial line, the QEMU harness) prints the
same lines as plain text, byte-for-byte, which is what the gates assert.

The screen covers the *whole* install, which for v3 means one list that spans both processes:

```
 1  target disk and identity          /dev/nvme0n1  YMTC PC41Q-2TB  1.9 TB   user 1bit  host 1bit-strixhalo
 2  base system (Ubuntu 26.04 Server)   curtin: partitions, unpack, packages, bootloader
 3  our repository from the stick      49 packages, 612 MB
 4  kernel and modules                 vmlinuz-7.3.0-rc4-1bit-perfopt+, amdxdna, ec_su_axb35, initramfs
 5  Secure Boot key                    already enrolled  |  queued for MokManager at the next boot
 6  firmware boot entry                1bit OS -> \EFI\ubuntu\shimx64.efi   BootOrder 0000,...
 7  data partition                     adopted 1BIT-DATA (1.1 GB, 4 models)  |  created empty
 8  verification                       ESP listed, kernel sha256, 8 packages, units enabled
 9  done                               remove the stick; rebooting into the installed system in 10 s
```

Steps 1, 3–9 are ours (`early-commands`, `postinstall`, `error-commands`), step 2 is subiquity's and is
shown by subiquity. The handoff is explicit and is the one thing in this design to prove in the gate:
subiquity draws tty1 from the moment it starts, so our step lines are written to tty1 **and** ttyS0
**and** `curtin`'s log (whose tail subiquity displays); if the tail does not show them, the renderer takes
tty1 after curtin's postinstall stage instead, and the QEMU gate decides between the two by grepping the
serial transcript — either way the transcript carries every step, which is what the gate asserts.

Engine#710's other requirements map as: every step also to `/data/install.log` on the stick (`/cdrom/...`
is read-only) and on the disk, plus ttyS0; errors never a bare exit; the final line and its countdown are
ours, printed before subiquity's own "Installation complete" reboot — and because step 4 sets `BootNext`
to the disk's entry, that reboot lands on the installed system **with the stick still in**, which is the
part the owner actually needs after engine#709.

## The firmware boot entry (engine#709)

The failure that motivated it, measured on this box: `efibootmgr -v` shows ten entries in `BootOrder`,
`Boot0000` "Ubuntu" → `\EFI\ubuntu\shimx64.efi` on the current ESP, and a stale `Boot0002 "1bit OS USB"`
that points at a `VenHw` device path with no file. After the first v2 install the firmware tried the wiped
Ubuntu entry, failed, and fell through to the stick.

Three fixes, all in the installed system rather than in a special-case installer:

1. **efivarfs exists.** Our kernel is built with `CONFIG_EFIVAR_FS=m` (measured in
   `/boot/config-7.3.0-rc4-1bit-perfopt+`) and an Ubuntu base loads it, so `/sys/firmware/efi/efivars` is
   populated (97 entries on this box). The v2 image never staged the module and BusyBox never loaded it,
   which is why the live installer could not fix anything from *inside* the installed system either
   (engine#709). v3 installs from a full Ubuntu session, where `efibootmgr` works, and asks for `=y`
   in the kernel config anyway so the next BusyBox image cannot regress it.
2. **`efibootmgr` is in the image and in the target** (`packages:` above), and the entry points at the
   shim Ubuntu's bootloader step installed, not at a path we invented.
3. **The entry is created, ordered and cleaned**, and the resulting `BootOrder` is printed on the install
   screen (step 6) and written to the log. Deleting entries whose loader does not exist on the target
   disk is the part that stops a batch of dead entries from reordering future boots.

The QEMU gate asserts the entry because OVMF has working efivars: after the install, with the same NVRAM
copy, `efibootmgr -v` in the installed system must show `1bit OS` first in `BootOrder`.

## The package set

`1bit.gg/apt`, same names and layout as v2, with the pieces Ubuntu 26.04 or the release already provide
not re-built:

| package | what | size (measured where noted) |
|---|---|---|
| `1bit-kernel-image-<kver>` | `/boot/vmlinuz`, config, `System.map`; postinst runs `update-initramfs`; versioned name = the rollback | 86 MB `.deb` (this box) |
| `1bit-kernel-headers-<kver>` | installed only when the out-of-tree modules must be rebuilt on a kernel bump; not in the seed's package list, so a fresh install has the eight below | 10 MB `.deb` (this box) |
| `1bit-kernel-modules-<kver>` | `amdxdna.ko` (0.18.0 pin) and `ec_su_axb35.ko` in `/lib/modules/<kver>/updates/`, signed with our MOK (`strixhalo Secure Boot Kernel`, measured) | 34 KB + 450 KB loaded |
| `1bit-rocm` | the pinned TheRock user space the release was gated with, at its measured paths: `libhsa-runtime64.so.1`, `libamd_comgr.so.3`, `libamdhip64.so`, `librocblas`, `libhipblaslt`, `librocsolver`, `librocsparse`, `llvm/amdgcn/bitcode`; `/etc/1bit/libhsa.env` with the no-SVM trio | 173 MB + 3.5 MB bitcode |
| `1bit-xrt` | `Depends:` the archive's `libxrt-npu2`, `libxrt-utils-npu`, `libxrt2` (measured 1:2.21.75+dfsg-4) — the device nodes and tools, not rebuilt | from the archive |
| `1bit-engine` | `/opt/1bit` as v2: `1bit`, the HRX `llama-server` and its libraries, `1bit-selftest`, `1bit-release-check`, `1bit-board`, sealed NPU sets, `/etc/1bit/engine.conf` | release tarball |
| `1bit-lemonade` | the release's own `lemond` + resources (the `onebit` recipe, one hop to `1bit serve`), `/opt/lemonade`, its unit | 39 MB `.tar.zst` (c18) |
| `1bit-appliance` | units (`1bit-console` on tty1, `data.mount`, `1bit-board`, `1bit-boost`, `1bit-first-boot`), banner, ssh socket, `unattended-upgrades` policy, `/etc/1bit/power`, `/data` tmpfiles | — |
| `1bit-archive-keyring` | the public half of the key that signs `1bit.gg/apt` | — |
| `1bit-openwebui` | optional, as v2: published, installed by nobody, nothing depends on it | — |

Two things v3 takes from the archive instead of building, because engine#712 is about the OS, not the
supply chain: **`linux-firmware-amd-graphics` and `linux-firmware-amd-misc`** (26.04's split packages;
`-amd-misc` is "Firmware for AMD NPU accelerators", measured 20260319.git217ca6e4-0ubuntu1.1, and it is
what puts `amdnpu/17f0_11/npu.sbin.zst` under `/lib/firmware`), and **XRT**, as above. Both carry their
own licences and land in `/usr/share/doc/<package>/copyright`, which is what `os/LICENSES.md` already
records for the current image.

The runtime is pinned by sha256, and the three runtimes that exist on a v3 box are the three engine#715
measured:

| runtime | sha256 (measured) | v3's use |
|---|---|---|
| release-gated `libhsa` (c18) | `f6ef689d…` | what `1bit-rocm` installs |
| TheRock nightly at `/opt/rocm-therock` | `21a66824…` | the same file on a box built from the SDK |
| distro ROCm 7.1 `libhsa-runtime64.so.1` | `5b113248…` | present because XRT pulls the distro stack — **not** what HRX loads |

`/etc/1bit/libhsa.env` (the `EnvironmentFile=-` hook v2's lemonade unit already has) carries the measured
fault-free arm by default:

```
HSA_USE_SVM=0
HSA_USERPTR_FOR_PAGED_MEM=0
HRX_ALLOW_NO_SVM=1
ONEBIT_HRX_LIBHSA=/opt/1bit/libhsa/libhsa-runtime64.so.1
```

This box reports `XNACK enabled: NO` and `Coherent Host Access: FALSE`; without the trio HRX page-faults
on its first graph replay (#715, 58 fault lines, HTTP 500 in <10 ms) and the release-tarball runtime
refuses the device outright. `1bit serve --hrx-cwsr-placement` is the alternative knob; the doc sets the
trio because it is the arm that was measured fault-free at 260.1 tok/s.

**Kernel and modules move together.** `1bit-kernel-image-<kver>` and `1bit-kernel-modules-<kver>` are
the same upstream version and both versioned; the image `Depends:` the modules of its own `<kver>`, so no
`apt` resolution can install one without the other, and `unattended-upgrades` upgrading the image pulls
the modules in the same run. That is the v2 lesson (a module built for another kernel is a machine whose
fans and NPU are gone) applied to the package graph instead of to `mkdebs.sh`'s checks alone.

**`unattended-upgrades` policy**: security updates automatic; `1bit.gg/apt` automatic for everything
except `1bit-kernel-*` and `1bit-rocm` (a kernel needs a reboot, a modules pair, and the gate; a ROCm
move is a measured decision), which are `Package-Blacklist`ed and reported by `release-check`'s `updates`
row; `Automatic-Reboot "false"`, and the console menu shows "reboot required" instead. That is the
appliance's contract: it cannot silently become a machine nobody measured.

**User and ssh**: `1bit` in `sudo`, root locked, `ssh.socket` enabled with keys carried from the stick,
`PasswordAuthentication no`, `PermitRootLogin no`. Headless means the console is the fallback, not the
other way round: `1bit-menu` on tty1 can drop to a shell as `1bit` (sudo), and ttyS0 gets a console unit
so a serial cable works when ssh does not.

**`1bit-menu` on tty1** is a unit (`1bit-console.service`, `Conflicts=getty@tty1.service`), not an
autologin shell: it draws the banner (model, IP, API key hint, GPU/NPU state, pending updates), drives
`1bit-release-check`, and its whole job after that is to be the screen the owner looks at. ssh and ttyS0
stay plain, which is what keeps the gates unchanged.

## The upgrade path

**v2 → v3 is a reinstall, not an upgrade, and the installer says so.** v2's installed machine is a
BusyBox initramfs on the ESP and one ext4 `/data`; there is no dpkg database, no user, no package for
`apt` to upgrade. What carries is exactly what is on `/data`, because `preserve: true` keeps partition 3:

| carried | how |
|---|---|
| models (`/data/models`) | left in place; `/data` mounted by UUID in the new fstab |
| Lemonade's config, cache and backends | left in place; the unit's `HOME=/data/lemonade` is unchanged |
| `hrx-jit` kernel cache, sealed NPU sets (`/data/npu`) | left in place; `GGML_HRX_JIT_CACHE_DIR` unchanged |
| SSH host key and `authorized_keys` | `/data/ssh/host_ed25519` → `/etc/ssh/ssh_host_ed25519_key`; `authorized_keys` → `/home/1bit/.ssh/` |
| Lemonade's API key | `/data/lemonade/lemond.conf` read as before |

| not carried, by design | why |
|---|---|
| v2's kernel and initramfs on the ESP | v3 keeps its kernels in `/boot` and its boot chain in the ESP; the old files are removed with the partition table |
| `/data/.installed` | v3 has no live/installed marker; `release-check`'s `mode` row reads the mount, not the file |
| root-only ssh | replaced by the `1bit` user; the same keys work |

The stick for this is the v3 stick with the machine named: `prepare-stick.sh --disk-serial …`, then the
install. The doc's rule is that a v2 machine can also be **returned** to v2 by writing the v2 image,
because `/data` was never touched.

**v3 → v3 is `apt`**: `apt update && apt upgrade` from `1bit.gg/apt`, unattended for security and for our
non-kernel packages, with the previous kernel kept installed (versioned package names) so the grub menu
still has the entry to go back to. After any upgrade the machine's own `1bit-release-check` runs and its
`updates`/`reboot-required` rows say what is pending. No `do-release-upgrade` is offered from v3: the next
OS change is the next v3 installer, because the base snapshot moves with the release and the kernel is
ours anyway.

## The gates

Same four QEMU phases the release already runs (`gate-c18.sh`), retargeted at the v3 stick, plus one
phase that is new because the install is new:

| phase | what runs | what must hold |
|---|---|---|
| A (stick) | boot the stick's live entry | unchanged from c18: banner, `1bit-selftest`, `llama-server` on HRX0 |
| R (release check) | `1bit-release-check` on the stick | unchanged, plus `sealed-sig` is still SKIP (the live image has no python3) |
| I (install) | boot the install entry, unattended | the serial transcript contains every step line 1–9, the `1bit OS` boot entry and the printed `BootOrder`, `postinstall` exit 0, the install log written to both media |
| B (installed) | boot the installed disk alone, same NVRAM | `efivarfs` mounted; `dpkg-query -W '1bit-*'` = 8 packages; `systemctl is-enabled` the unit set; `ssh.socket` listening; `getent passwd 1bit` and `sudo -l`; `/data` mounted from partition 3 by UUID; `apt-get update` and `apt-cache policy` from `1bit.gg/apt`; `unattended-upgrades --dry-run` clean; `1bit-release-check` PASS with `sealed-sig` now checked (`python3` present); `1bit-console` active on tty1 |
| **V** (v2 → v3) | install a v2 image, then the v3 stick over it | `/data` contents identical before and after (manifest sha256), the ssh host key unchanged, the v2 ESP gone, the machine boots v3 |

QEMU specifics, all measured as available on this box: `qemu-system-x86_64` with
`-drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd` and a writable copy of
`OVMF_VARS_4M.fd` (the efivar assertions need a NVRAM that survives a reboot), `-nographic` with
`console=ttyS0,115200`, one NVMe disk with a fixed serial (`-device nvme,serial=1BITTEST…`) so the seed's
`match: {serial:}` resolves, 8 GiB RAM, and a second NVMe/`-drive` for the v2 image in phase V. Secure
Boot stays off in the VM (`OVMF_CODE_4M.fd`, not `…secboot.fd`), exactly as v2's `vmtest` documents: MOK
enrolment is the machine's step, not a VM's.

**Board gate** (the owner, on the Strix Halo box, stick in hand): boot the stick live, `1bit-release-check`
(every row PASS or SKIP with a reason — the VM run's 18-row transcript is the reference for what each row
asserts), then the install entry, watch the screen, then — **with the stick still in** — confirm the reboot lands on the installed machine; then
`efibootmgr -v` (`1bit OS` first), `systemctl status` the unit set, `1bit-menu` on the monitor, ssh from
the build machine with the carried key, HRX chat and NPU chat from `1bit-menu`, fans/power units active,
and `apt upgrade` on the installed box. Secure Boot is on: the install queues our MOK, so the very first
boot after install stops in MokManager and needs one keyboard round trip; on a box that already has the
key enrolled (`mokutil --test-key` true) there is none. That step is why the doc says the *install* needs
a console; everything after it is headless.

## What is measured now

Reference box (strixhalo, 2026-11-01, the machine this design was written on):

| fact | measured |
|---|---|
| base | Ubuntu 26.04.1 LTS, resolute |
| our kernel | `7.3.0-rc4-1bit-perfopt+`, `linux-image-7.3.0-rc4-1bit-perfopt+` 7.3.0~rc4-4, `CONFIG_EFIVAR_FS=m` |
| out-of-tree modules | `amdxdna.ko`, `ec_su_axb35.ko` in `/lib/modules/<kver>/updates/`, both signed `strixhalo Secure Boot Kernel`, vermagic `<kver> SMP preempt mod_unload modversions` |
| MOK | `CN=strixhalo Secure Boot Kernel, O=1bit-MONSTER`, self-signed, EKU Code Signing; Secure Boot enabled |
| firmware | `efibootmgr` 10 entries; `Boot0000 Ubuntu` → `\EFI\ubuntu\shimx64.efi`; stale `Boot0002 1bit OS USB`; 97 efivars |
| firmware packages | `linux-firmware-amd-graphics`, `linux-firmware-amd-misc` 20260319.git217ca6e4-0ubuntu1.1 (`amdnpu/17f0_11/npu.sbin.zst`) |
| XRT | `libxrt-npu2`, `libxrt-utils-npu`, `libxrt2` 1:2.21.75+dfsg-4 |
| ROCm | `/opt/rocm-therock` 7.2 GB; the runtime subset 173 MB + 3.5 MB bitcode; `libhsa` `21a66824…` |
| the three libhsa | `f6ef689d…` (release), `21a66824…` (nightly), `5b113248…` (distro 7.1) |
| packages | kernel image 86 MB `.deb`, headers 10 MB; c18 `lemonade-onebit` 39 MB; live `.img.zst` 397 MB; live `.efi` 106 MB |
| v2 in the field | ESP 512 MiB + `1BIT-DATA`; `1bit-install` writes `/data/.installed`; no efivarfs, no `efibootmgr` (engine#709) |
| c18 gate | A/R/I/B green in QEMU, install writes ESP + `/data` (1.1 GB copied), the installed boot passes `1bit-release-check` 18 rows with `sealed-sig` SKIP for lack of python3 |

Not measured, and therefore the risk list for the build:

- **the server ISO's own size, sha256 and volume label** — pinned in `os/v3/pin.txt` at the first build,
  not quoted here; the QEMU phase I run is what makes the number real;
- **subiquity's screen handoff** — whether our step lines appear in the tail it displays, or whether the
  renderer must take tty1 after curtin (the gate decides; both keep the serial transcript complete);
- **`mokutil --import` reading its password from stdin** in a curtin chroot — if it refuses, the fallback
  is a first-boot oneshot that imports it with the console, and the doc's "no MOK" note for boxes whose
  key is already enrolled stays valid;
- **`preserve: true` over a v2 partition table** — curtin's behaviour on a disk whose existing layout does
  not match the config is the phase-V gate's job, and if it wipes, the fallback is `1bit-install
  --adopt-data` grown from the live entry's v2 shell, which has `blkid` and `sfdisk` and can write the
  seed's storage section with explicit ids after inspecting the disk);
- **the ROCm subset's completeness** — 173 MB is what `ldd` on the release's `llama-server`, `ds4` and
  `libggml-hrx.so` needs *here*; a box with a different GPU path may want more, so `mkdebs` derives the
  list from those binaries rather than from a hand-written list;
- **the ISO repack** (`os/v3/mkstick.sh`) — extracting a pinned ISO to a partition keeps casper working on
  the media it was built for; the gate's phase I proves it, and if extraction is not enough the fallback
  is `loopback` booting the ISO file, which needs `iso-scan/filename=` and a second labelled partition.

## The `os/v3` tree

| file | what |
|---|---|
| `pin.txt` | snapshot, server ISO (url, size, sha256, volume label), package pins |
| `mkstick.sh` | `mkrootfs`'s replacement: extract the pinned ISO to partition 2, add `/1bit/`, write the ESP from `os/mini`'s payload, `prepare-stick`-style identity |
| `mkdebs.sh` | v2's `mkdebs.sh` plus `1bit-rocm` (subset derived from `ldd`), `1bit-xrt` (Depends), the kernel/modules pair, and the archive-keyring |
| `seed.sh` | render `autoinstall.yaml` + `identity` from the stick's inputs; `--check` refuses a seed with no disk serial, no password hash, no key |
| `postinstall` | the nine steps, `set -e` with the failure trap that keeps the screen |
| `steps` | the renderer entry points `plan`, `step`, `failed` (v2's `os/mini/tui.sh`, Apache-noticed copy) |
| `vmtest-v3.sh`, `vmtest-install-v3.sh` | phases A/B and I/V, transcripts in `/data` as v2's do — the host reads the guest's transcript, so no port forwarding decides a verdict |
| `diag-v3.sh` | boot the installed disk and ask the guest what failed; judges nothing |

## Licence

The scripts and this page are Apache-2.0, like the rest of the repository. What the packages carry is in
[os/LICENSES.md](../os/LICENSES.md); the Ubuntu base adds the archive's own licences, and the appliance
package ships the aggregate under `/usr/share/licenses/1bit-os/` as v2's does.
