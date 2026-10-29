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
# 1bit OS v2 — the appliance on a minimal Ubuntu base, updated with `apt`

`os/mini` builds the BusyBox image that ships today: everything in RAM, a reflash to update. v2 keeps the
same appliance — Lemonade Server by AMD, running on the 1bit engine, our kernel, Secure Boot through the
enrolled MOK — on a **minimal Ubuntu 26.04 root filesystem with systemd**, where everything we ship is a
`.deb` from our own signed repository and `apt update && apt upgrade` is the update path.

The design and the reasons are in `~/briefs/TOSAPT-DESIGN.md`; this file is what is in the tree and how to
reproduce it. `os/mini` keeps shipping until v2 passes the same checks on hardware.

## Build it

```sh
# 1. the packages. A release build produces the two inputs: an engine stage (scripts/package-linux.sh's
#    tarball, or a build tree with ENGINE_BUILD=<dir>) and a Lemonade prefix (bin/lemond, bin/resources).
#    KERNEL_STAGE is the release kernel as it sits on the build machine:
#      <dir>/boot/{vmlinuz,System.map,config}-<kver> and <dir>/lib/modules/<kver>/
os/v2/mkdebs.sh <engine-stage> <lemonade-prefix> v2026.43 <out-debs> [archive-keyring.gpg]
               # KERNEL_STAGE=<dir> ENGINE_BUILD=<build dir>

# 2. the apt repository those debs are published from (signed: Release, Release.gpg, InRelease)
os/v2/mkrepo.sh <out-debs> <repo-dir> [gnupg-home] [key-id]

# 3. a root filesystem from one Ubuntu snapshot (pin.txt) that apt-installs 1bit-appliance from it
os/v2/mkrootfs.sh <repo-dir> <rootfs-dir> [ssh-key.pub]

# 4. the live medium: one image to dd onto a stick
os/v2/mklive.sh <rootfs-dir> <data-seed-dir> <out.img> [mok.priv mok.pem]

# 5. the two tests (no hardware, no GPU: QEMU)
os/v2/vmtest-v2.sh <rootfs-dir> <data-seed-dir> [ssh-key]        # the appliance, from a rootfs
os/v2/vmtest-install-v2.sh <rootfs-dir> <data-seed-dir> <work-dir> [mok.priv mok.pem]   # live + install
```

`os/v2/proto.sh` runs 1..5 in one command against a throwaway signing key, which is what a lane without
the operator's key uses. `os/v2/diag-v2.sh` boots the rootfs and asks the guest what failed — it is a
debugging tool, it judges nothing.

`MIRROR=<url>` overrides the pinned snapshot in `mkrootfs.sh` for one run. That exists for a
`snapshot.ubuntu.com` outage (it 503'd for an hour on 25 Oct); a release build is the pinned snapshot.

## The packages

| Package | What is in it |
|---|---|
| `1bit-engine` | `/opt/1bit`: the engine, the HRX `llama-server` and its libraries, the ds4 programs, the sealed NPU kernel sets, `1bit-selftest`, `1bit-release-check`, `/etc/1bit/engine.conf` |
| `1bit-lemonade` | `/opt/lemonade`: our build of Lemonade Server (`lemond`, the CLI, the web app) and its unit |
| `1bit-kernel-image-<kver>` | `/boot/{vmlinuz,System.map,config}-<kver>`; its postinst runs `update-initramfs`. The **kver is in the name**, so an upgrade keeps the previous kernel installed — that is the rollback |
| `1bit-kernel-modules-<kver>` | every module for `<kver>`, including the out-of-tree EC board driver (MOK-signed) and its source under `/usr/src/1bit-ec-<ver>` |
| `1bit-ryzenadj` | `/usr/bin/ryzenadj`, built from the pinned v0.19.0 source, which ships beside it (LGPL-3.0) |
| `1bit-appliance` | the meta/config package: units, banner, autologin, network, `/data`, first boot, apt source, `/usr/share/licenses/1bit-os`. `Depends:` the six above |
| `1bit-openwebui` | **optional — published, installed by nobody**: the unit that runs the upstream OpenWebUI container, its console line and its preset (engine#594). Nothing depends on it |
| `1bit-archive-keyring` | the public half of the key that signs `1bit.gg/apt` |

Two things the packages must get right, both learned the hard way and both checked by `mkdebs.sh`:

- **The engine's own libraries are found beside it.** `scripts/pack-elf.sh`'s `pack_localize` runs over the
  package (`ENGINE_BUILD`), so `libllama-server-impl.so`, `libhrx`, `libloomc`, `libomp` are in the package
  and `RUNPATH` is `$ORIGIN`. A build-machine path in a package is a package that does not load (#549).
- **Signed, or Secure Boot refuses it.** `mkdebs.sh` fails if the EC module is unsigned or built for another
  kernel, and if the kernel image carries no signature (`sbverify --list` → `signature 1`).

## The optional chat front end (OpenWebUI)

Lemonade serves its own web app at `http://<box>:13305/` — chat works with nothing else installed. The
`1bit-openwebui` package adds the other front end (users, accounts, chat history, documents) as the
upstream OpenWebUI container pointed at the Lemonade API. It is **optional on purpose**: it is in the
repository, no package `Depends:`/`Recommends:` it and no target wants its unit, so an absent package, an
absent runtime or a failed image pull cannot hold a boot. Nothing of it is in the RAM live image
(`os/mini`): it is several hundred MB and wants a disk.

Install it on a box that is already running (`sudo` or root):

```
sudo apt install podman          # or docker.io; podman is preferred (no daemon to be up first)
sudo apt install 1bit-openwebui  # enables the unit and the console line, starts nothing
sudo systemctl start 1bit-openwebui
```

The first start pulls `ghcr.io/open-webui/open-webui:main` (a multi-GB download, so it is the operator's
step and not dpkg's; a box with no network yet simply retries — `Restart=always`, `RestartSec=15`). Then
`http://<box>:8080/` is the chat UI, the console banner gains a `Chat UI` line beside the Lemonade one, and
the admin account is created on the first page load. Without the package, or before it answers, the line
reads `(starting)`; without the package at all the line is not printed and nothing is probed.

```
systemctl status 1bit-openwebui           # active (running) once the container is up
journalctl -u 1bit-openwebui -n 50        # the pull, and the runtime error if there is one
journalctl -u 1bit-openwebui-console      # the line writer (timer: 20 s after boot, then every 60 s)
```

What it reads and writes: the API key from `/etc/1bit/lemonade.env` (`LEMONADE_API_KEY`, or from
`/data/lemonade/lemond.conf` on a box prepared from a stick), a generated environment at
`/etc/1bit/openwebui.env` (`0600`, rewritten at every start with the persistent `WEBUI_SECRET_KEY` kept),
and the container's `/app/backend/data` — accounts and chats — at `/data/openwebui`, so an image update,
a `systemctl restart` or a package upgrade keeps them. To pin the image instead of following `:main`,
write the operator's override file the unit reads (root, `0644`):

```
# /etc/1bit/openwebui.conf
ONEBIT_OPENWEBUI_IMAGE=ghcr.io/open-webui/open-webui:v0.6.5
```

then `sudo systemctl restart 1bit-openwebui`. `1bit-openwebui.env` is generated at every start (do not
edit it: it is rewritten); `openwebui.conf` is never written by the package.

Remove it and keep the chats: `sudo apt remove 1bit-openwebui` stops the console line with the package and
leaves `/data/openwebui` alone; `apt purge` also drops `/etc/1bit/openwebui.env` and `openwebui.conf`. `rm -rf /data/openwebui`
is the only thing that deletes accounts or chats, and nothing does it for you.

## Trust and the boot chain

```
firmware (db) -> shim (Microsoft-signed) -> grubx64.efi (our MOK) -> vmlinuz (our MOK) -> initramfs-tools
                                                                 \-> ec_su_axb35.ko (our MOK, module-signing OID)
```

`apt` trusts `1bit.gg/apt` through `signed-by=/usr/share/keyrings/1bit-archive-keyring.gpg` and nothing
else: no `apt-key`, no `[trusted=yes]`. An install copies the medium's own signed chain, so the machine's
MOK enrolment is never touched by installing.

## The live medium, and what an install writes

`mklive.sh` writes one disk image, in the same three partitions an installed box has:

| | live medium | after `1bit-install /dev/nvme0n1 --yes` |
|---|---|---|
| 1 | 512 MB ESP `1BIT-OS`: shim, our grub, kernel + initramfs, `grub.cfg` | the same, with a `grub.cfg` for the disk |
| 2 | squashfs, `PARTLABEL=1BIT-LIVE` — the whole root filesystem, read-only | ext4 `1BIT-ROOT` = `min(64 GB, max(24 GB, 25 %))`, the squashfs unpacked, `/boot`, `fstab` by label |
| 3 | ext4 `1BIT-DATA` — models, backends, config, keys | the same content, copied from the medium |

The live system is the squashfs with a tmpfs overlay: an initramfs-tools script
(`scripts/local-bottom/1bit-live`) mounts it, so userspace has nothing custom at boot and a live boot
cannot write the medium. `1bit-install` refuses before writing anything — no `--yes`, a partition instead
of a disk, a removable device, the disk the live boot runs from, a disk with something mounted, a missing
GPT tool — prints the disk and waits 10 s, exactly as the `os/mini` installer does (`#539`).

## What is proven, and where

`vmtest-v2.sh` runs the checks **inside** the guest (a test-only oneshot unit writes its transcript to
`/data` and powers the machine off); the host reads that transcript, so no port forwarding decides a
verdict. `vmtest-install-v2.sh` does the same for the live medium and then boots the installed disk alone.

| What | How |
|---|---|
| all seven installed packages are the release's (**eight** are built: `1bit-openwebui` is the optional one) | `dpkg-query -W '1bit-*'` in the guest |
| the banner, `/data` on its own disk, `1bit-selftest` | guest transcript |
| the engine runs; `llama-server`'s own libraries resolve | `1bit --help`, `ldd` on the packaged `llama-server` |
| our kernel package produced `vmlinuz` **and** `initrd.img` | `/boot` in the guest |
| Lemonade: key generated, 401 without it, the GGUF listed, one chat answered | `curl` against `127.0.0.1:13305` |
| apt answers from the repository, and the shipped source is `1bit.gg/apt` with `signed-by` | `apt-get update`, `apt-cache policy`, the `.list` |
| `1bit-release-check`: every row passes except the sealed-NPU one (a dev stage carries no sealed sets) | its own output |
| the live medium boots the squashfs with the overlay; all six refusals; the install writes three partitions with the kernel in `/boot`, `fstab` by label and the live marker gone | `vmtest-install-v2.sh` pass 1 |
| the installed disk boots alone: INSTALLED banner, first-boot marker and own SSH keys, `/data` on its disk, Lemonade 401 + GGUF + chat, `1bit-selftest` reports INSTALLED | `vmtest-install-v2.sh` pass 2 |

The VM boots **OVMF** (EFI, Secure Boot off) rather than a legacy BIOS, so `/sys/firmware/efi` exists and
the `secureboot` row answers; enrolling the MOK is the machine's step, not a VM's.

## Known gaps

- **The ROCm userspace is not packaged.** The engine's HIP path (`llama-server`, `ds4`) links
  `libamdhip64`, rocBLAS, hipBLASLt, the HSA runtime and `libamd_comgr` from the SDK the build machine had.
  Ubuntu 26.04 has packages for them (`libamdhip64-7`, `libhipblaslt1`, `librocblas*`), but gfx1201
  support against the distro's ROCm is a hardware question, so the engine's `Depends` do not name them yet
  and a VM (no GPU) cannot settle it. The engine's *own* libraries are packaged and resolve — that is what
  `vmtest-v2` asserts.
- **A development stage has no sealed NPU sets**, so `1bit-release-check`'s `sealed` row fails by
  construction in these runs (the VM has no NPU either). A release stage always carries them.
- **The optional `1bit-openwebui` package is not exercised by the vmtest runs.** Installing it needs a
  container runtime from the Ubuntu archive and its first start pulls a multi-GB image, so the VM would
  spend the run on a download and prove nothing about 1bit OS. What is checked instead is `mkdebs.sh`
  building the `.deb` and the unit/console scripts being syntax-clean; a box check of the whole thing is
  in `vmtest-install-v2.sh`'s style but is not run here — see engine#594.
- **The v2 rootfs in these runs came from the archive mirror**, not the pinned snapshot, because
  `snapshot.ubuntu.com` was 503. `pin.txt` is unchanged and is what a release build uses.

## Licence

The scripts here are Apache-2.0, like the rest of this repository. What the packages carry is in
`os/LICENSES.md`; each package ships its own licences under `/usr/share/doc/<package>/` and the aggregate
under `/usr/share/licenses/1bit-os/`.
