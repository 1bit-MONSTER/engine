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
# Install 1bit OS

1bit OS is the Puppy Linux appliance in [os/README.md](../os/README.md): Lemonade Server with this engine
inside, our MOK-signed kernel, headless by default. It runs from the stick it was written to, and it can
install itself onto a disk: `1bit-install` copies the system you booted, so an install downloads nothing
and takes about a minute.

| what you want | how | what it erases |
|---|---|---|
| try it, keep nothing | boot the stick — [run from the stick](#1-run-from-the-stick-no-install) | nothing on the machine |
| run it from the internal disk | `sudo 1bit-install <disk> --yes` on the stick | **that disk, all of it** |
| carry it on another USB drive | the same command onto that drive | **that drive, all of it** |
| keep the OS already on the machine | not in this release — [see 4](#4-keep-the-system-already-on-the-machine) | — |
| add the engine to an existing Linux | the engine package — [see 5](#5-the-engine-on-an-existing-linux-advanced) | nothing |

## What you need

| | |
|---|---|
| stick | **16 GB or more**: the image is 9,128,902,656 bytes (8.5 GiB) and the data partition takes the rest of the stick |
| image | the 1bit OS image from the [releases page](https://github.com/1bit-MONSTER/engine/releases) |
| machine | an AMD Ryzen AI MAX+ 395 (Strix Halo) box, the machine this release is built and tested on |
| firmware | UEFI, Secure Boot **on** (see [6](#6-secure-boot-and-our-mok)) |

Write it and prepare it (the second line is what puts the API key, your SSH key and any models on the
stick — see [1](#1-run-from-the-stick-no-install)):

```sh
sudo dd if=1bit-os-puppy.img of=/dev/sdX bs=4M conv=fsync status=progress
sudo os/mini/prepare-stick.sh /dev/sdX --mint --ssh-pubkey ~/.ssh/id_ed25519.pub
```

`prepare-stick` writes `/ssh/authorized_keys` and a freshly minted Lemonade API key into
`/lemonade/lemond.conf` on the stick's `1BIT-DATA` partition, and copies `--model` files into
`/models`. A stick written with `dd` alone has **no API key**: the appliance boots, but Lemonade says so
in its log and serves every endpoint unauthenticated
(`Serving on non-loopback host '0.0.0.0' without an API key ... all endpoints ... are reachable from
other machines unauthenticated`), which is why `--mint` is part of the protocol. That the appliance does
not mint one itself when the file is empty is a defect of this image
([#867](https://github.com/1bit-MONSTER/engine/issues/867)); until it is fixed, treat `--mint` as
mandatory, and check a stick you did not prepare with
`sudo sed -n 's/^LEMONADE_API_KEY=//p' /data/lemonade/lemond.conf` — an empty answer means open.

## 1. Run from the stick (no install)

1. Put the stick in the machine. On the Bosgame BeyondMax use the **front USB-A** port
   ([7](#7-the-bosgame-beyondmax-has-one-usable-usb-port)).
2. Boot it. The default entry, `1bit OS (headless console, 7.3.0-rc4-1bit-perfopt+)`, comes up headless:
   a console with no X. The second entry, `1bit OS (desktop, JWM)`, starts Puppy's desktop on the same
   system; the third, `1bit OS - serial root shell`, gives a root shell on the serial port for a rack
   machine with no monitor.
3. Nothing on the machine is written. The system runs in RAM from the stick's squashfs
   (`pfix=copy`); the models, Lemonade's config and cache, your SSH keys and the API key live on the
   stick's `1BIT-DATA` partition.

Check that it is up:

```sh
curl -s -H "Authorization: Bearer $(sudo sed -n 's/^LEMONADE_API_KEY=//p' /data/lemonade/lemond.conf)" \
     http://127.0.0.1:13305/api/v1/models
```

That command is one of the ones measured below; a client on another machine uses the box's address
instead of `127.0.0.1` ([clients.md](clients.md)).

**What persists on the stick**, all of it on `1BIT-DATA` (Puppy's save layer, `PUPMODE=12`):
`models/`, `lemonade/config/`, `lemonade/cache/`, `lemonade/lemond.conf` (the API key), `ssh/` (host key
and `authorized_keys`), `npu/kernels/`, `cache/hrx-jit/`. A file written elsewhere (`/etc`, `/root`, `/tmp`) lives in RAM: measured, a marker
written in `/etc` and then `poweroff -f` was gone on the next boot, while the same boot's marker on
`/data` was still there.

## 2. Install to an internal disk

From the running stick:

```sh
lsblk -o NAME,SIZE,MODEL,MOUNTPOINT      # find the disk: model and size, never the stick
sudo 1bit-install /dev/nvme0n1 --yes
```

`/dev/nvme0n1` is an example: **pass the disk you mean to erase**. The installer refuses to continue
unless the argument is a whole disk, it is not the disk this system booted from, and none of its
partitions is mounted — and it prints the disk's model, its size and its current partition table, then
waits 10 seconds before writing:

```
1bit OS install
  target : /dev/nvme0n1  (Samsung SSD 990 PRO 2TB, 2000.4 GB)
  booted from: /dev/sda (/dev/sda2) - refused as a target
  source : ESP /dev/sda1, frugal install + data /data
  anything on /dev/nvme0n1 is about to be destroyed:
    ...
  writing in 10 s (Ctrl-C now and nothing has been touched)
```

**It erases the target disk.** What it writes, in its own seven steps:

| step | what |
|---|---|
| 1 | a **GPT** with a 512 MiB EFI system partition (vfat, label `1BIT-OS`) and a `1BIT-DATA` ext4 partition filling the rest |
| 2 | the formats, `mkfs.vfat` and `mkfs.ext4` with a fresh UUID for the data partition |
| 3 | the boot chain copied from the stick's own ESP: `BOOTX64.EFI` (shim), `grubx64.efi`, `mmx64.efi` (MokManager) and `grub.cfg` |
| 4 | the whole data partition — the frugal install `/1bit` (`vmlinuz`, `initrd.gz`, the four squashfs files) **and** `/data` (models, Lemonade's config and cache, the API key, SSH keys) — plus `.installed` and `install.log` |
| 5 | `grub.cfg` re-pointed at **this disk**: `search --fs-uuid <uuid>` and every entry pinned with `pupsfs=UUID=<uuid> 1bit.data=UUID=<uuid>`, because two `1BIT-DATA` labels exist once a machine is installed |
| 6 | a read-back: the kernel's sha256 on the new disk must equal the source's, `initrd.gz` and the SFS files must be there |
| 7 | a firmware boot entry via `efibootmgr`, label `1bit OS`, pointing at the new ESP, put first in `BootOrder` |

Then power off, **remove the stick**, and boot. The firmware's boot menu lists `1bit OS`; if the machine
comes up in its own menu instead of booting, pick it there once.

> **On the cycle-13 image the installed disk does not boot yet.** The install itself is complete and
> correct, but two defects stop the first boot: installer step 5 writes `pupsfs=UUID=<uuid>`, which
> Puppy's init does not resolve ([#865](https://github.com/1bit-MONSTER/engine/issues/865)), and step 7's
> `BootNext` lands on the firmware's own boot-manager entry instead of the new one
> ([#866](https://github.com/1bit-MONSTER/engine/issues/866)). Both are in the appliance lane for the
> release; until they land, boot the installed disk by editing its `grub.cfg` entry to
> `pupsfs=1BIT-DATA 1bit.data=UUID=<uuid>` — measured to boot, with the models and the API key in place.

Check the result from the installed system:

```sh
lsblk -f                                  # ESP 1BIT-OS + 1BIT-DATA on the target
sudo blkid /dev/nvme0n1*                  # the ESP is vfat, the data partition the UUID grub pinned
efibootmgr -v | grep "1bit OS"            # the firmware entry, pointing at this disk's ESP
```

The installer ends by telling you to run `1bit-release-check`, which says `INSTALLED` and runs the same
rows as the stick. The install log is on the installed `1BIT-DATA` partition as `/install.log`.

## 3. Install to another USB drive

The same command, pointed at the drive instead of an internal disk:

```sh
lsblk -o NAME,SIZE,MODEL,TRAN              # TRAN usb: which drive is which
sudo 1bit-install /dev/sdb --yes           # the stick is /dev/sda, so this is the other one
```

It is the same installer with the same refusals and the same seven steps; the target may be USB, SATA or
NVMe, it only has to be a whole disk that is not the one you booted. Afterwards **boot the machine from
that drive** (the firmware entry it wrote points at it): it is the same system, portable, with the models
and the API key carried across.

The stick you installed *from* is untouched, so you can keep it as the installer/recovery stick.

## 4. Keep the system already on the machine

**There is no dual-boot wizard in this release, and there is no "install alongside" mode.** The installer
takes one whole disk and erases it. What that means per case:

| the machine today | what to do |
|---|---|
| one disk you want to keep | install to a **second** disk (option 2), or to a USB drive (option 3) and boot that |
| a disk with nothing you need | install over it (option 2) |
| a Windows machine you want to keep intact | install to another disk or a USB drive; the appliance does not touch the Windows disk, and the firmware's own boot menu is how you go back |

The installer does not shrink, move or resize any partition, and it writes nothing to a disk you did not
name. It does add its own firmware entry, `1bit OS`, which the firmware's boot menu then offers next to
the other OS.

## 5. The engine on an existing Linux (advanced)

The supported install for this release is the appliance: 1bit OS carries the kernel, the NPU driver, the
signed modules and the HRX runtime that the engine needs, and a plain distribution does not. What exists
for an existing Linux:

| | |
|---|---|
| Lemonade with the engine as its backend | the `lemonade-onebit` package from the releases page, for a machine already running Lemonade ([docs/lemonade.md](lemonade.md)) |
| the engine alone, packaged for a distribution | `scripts/mk-plugin-package.sh` builds `1bit-engine_<version>_amd64.deb` and a tarball and wires a stock Lemonade to it (PR [#860](https://github.com/1bit-MONSTER/engine/pull/860)) |

Both are **advanced, and not the supported path for this release**: they put the engine on a system whose
kernel, `amdxdna` driver and HSA runtime we did not choose, so the device rows of
[1bit-release-check](../os/README.md) are yours to run. There is **no PPA route in this release** — it is
parked.

## 6. Secure Boot and our MOK

The stick and the install boot with **Secure Boot on**: shim → our MOK-signed grub → our MOK-signed
kernel (`7.3.0-rc4-1bit-perfopt+`) → our signed modules (`amdgpu`, `amdxdna`, `ec_su_axb35`). Verified on
the shipped image: `sbverify --cert MOK-Kernel.pem /1bit/vmlinuz` → `Signature verification OK`.

| your machine | what to do |
|---|---|
| our key is already enrolled (the box this release is tested on) | nothing — boot it |
| our key is not enrolled | the firmware will refuse the kernel. **Turn Secure Boot off** in the firmware setup for now: the system runs, everything works, and Secure Boot is a signature check, not a feature of the appliance |
| you want Secure Boot on anyway | enroll the public key with `mokutil --import MOK-Kernel.der` from an OS already installed on that machine, reboot, and confirm in MokManager (`Enroll MOK` → `Continue` → `Yes` → password) |

The release does **not** put the certificate on the stick — measured: the ESP carries `BOOTX64.EFI`,
`grubx64.efi`, `mmx64.efi` and `grub.cfg`, and the image carries no `.der` file — so a machine without
our key cannot enroll it from the stick alone. The signing key and what is signed with it are in
[kernel.md](kernel.md).

Check the state on a booted system:

```sh
mokutil --sb-state          # SecureBoot enabled
mokutil --list-enrolled | grep -i "1bit\|strixhalo"
```

## 7. The Bosgame BeyondMax has one usable USB port

On the Bosgame BeyondMax (`AXB35-02`, BIOS 1.09) **only the front USB-A port works under Linux**: a drive
plugged into the front or rear Type-C, or a rear USB-A port, produces no kernel event at all — the xHCI
port registers stay in RxDetect, and three of the four controllers have taken no interrupt. The same
ports work in Windows on the same box, so it is a firmware/EC gap, not a driver in this image
([#823](https://github.com/1bit-MONSTER/engine/issues/823)). Boot from the front USB-A port.

## What was verified for this page

On the cycle-13 image (`sha256 dcc619ff74882f484e4c7dfb373980f294ce2f39b02291fb816ac7c62776beb7`,
9,128,902,656 B) in QEMU with Secure Boot on and our MOK in `db`, kernel `7.3.0-rc4-1bit-perfopt+`:

| claim | measured |
|---|---|
| the image boots to Lemonade with Secure Boot on | boot, `lemond` on `0.0.0.0:13305`, web app on `/`, `ssh` on `22`, kernel signature OK |
| `1bit-install` refuses the disk it booted from | `1bit-install: REFUSED: /dev/sda is the disk this system booted from (/dev/sda2 is on it)`, exit 2, the stick's UUIDs unchanged afterwards |
| an install to a bare disk completes | all seven steps, exit 0: GPT `1BIT-OS` 512 MiB + `1BIT-DATA` 7.5 GiB, `grub.cfg` pinned to the new UUID, kernel sha256 read back equal, `.installed` and `install.log` written |
| the firmware entry is written | `efibootmgr` lists `1bit OS` → `HD(1,GPT,<new ESP>)/\EFI\BOOT\BOOTX64.EFI` (id `0004`, not the `0000` the step's own log line printed — [#866](https://github.com/1bit-MONSTER/engine/issues/866)) |
| what the install copies | the installed data partition holds `/1bit` (`vmlinuz`, `initrd.gz`, the four SFS), `/data` with the models, `lemonade/` and `ssh/`, and `.installed` |
| boot of the installed disk as written | **fails** on this image: init says `Partition UUID=<uuid> not found` and drops to the rescue console ([#865](https://github.com/1bit-MONSTER/engine/issues/865)) |
| boot of the same disk with `pupsfs=1BIT-DATA` (label) instead of `pupsfs=UUID=…` | **boots**: `puppypc20143`, `/data/.installed` present, `lemond` up, `GET /api/v1/models` 401 without the key and 200 with it |
| the boot device is rejected, not guessed | the refusal above is by device: `/proc/mounts` names `/dev/sda2` as `/data`, and the installer resolves the disk behind it |
| `pmedia=usbhd` is not the cause | the failed boot was from a USB device, where `pmedia=usbhd` is the correct media class; the label/UUID A/B above isolates the cause to the UUID form |

Not verified here: an install to a real internal NVMe on the hardware (the install gate in
[docs/releases.md](releases.md) covers it), the firmware's own boot menu on a non-QEMU machine, and
Secure Boot enrolment on a machine without our key (the table above is the documented route, not a
measurement of it). The image used was the cycle-13 artifact, not the next release's.
