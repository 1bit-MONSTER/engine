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
# 1bit OS v3 — the Puppy Linux appliance

## Which plan is v3

**Puppy is v3, and Puppy is 1bit OS.** It is `ResolutePup64`, built with woof-CE on the Ubuntu 26.04
"resolute" base, with our kernel (MOK-signed; `7.4.0-rc1-1bit` since cycle 21, `7.3.0-rc4-1bit-perfopt`
before it), our drivers, the engine and Lemonade inside it. The same image is the live/try stick and the
frugal install, `/data` is Puppy's save
layer on `1BIT-DATA`, the userland is `bash`, `sudo`, `apt` and `sshd`, and it boots **headless by
default** - a console login, no X - with a second boot entry, `1bit OS (desktop)`, for Puppy's desktop.
Secure Boot stays on: shim → our MOK-signed grub → our MOK-signed kernel → our signed modules.

The evaluation - sizes, the glibc ceiling, the kernel config gap, the package inventory, the build recipe
and the gates table - is [os-v3-live-puppy-eval.md](os-v3-live-puppy-eval.md); the build recipe is
[`os/v3/puppy/`](../os/v3/puppy/README.md); the page for what the appliance is, is
[os/README.md](../os/README.md).

Where it stands, measured 2026-11-01 to 2026-11-04: every blocker of the evaluation is closed or
answered - `kernel-puppy.fragment` is built into our kernel (G2 green, the initrd finds and layers the
SFS), `amdxdna` and `ec_su_axb35` are present, signed and verify against the enrolled MOK (G3 green), the
save layer persists across a reboot (G7 green, `PUPMODE=12` on `1BIT-DATA`), and `sudo`, `apt`, `sshd`
and `mokutil` are in the image - and the appliance **boots on the Max+ 395 with Secure Boot on**. Cycle 19
(2026-11-03) came back **33 `puppy-check` rows, 0 failed, 1 skipped** (the skip is `u2-desktop`, correct on
a headless entry); cycle 20 (2026-11-04) booted and served three chats on the NPU at 88.06 / 80.91 /
87.43 tok/s (TTFT 0.37 / 0.26 / 0.18 s), and its two red rows were traced to a defect in the acceptance
script rather than the image (engine#945, fixed in #951). Cycle 23 (2026-11-04) came back
**41 rows, 1 FAIL and 2 SKIP** with the image's own release check at **PASS (22 checks, 0 failures)** -
the single red is a bug in the check's own `u7-mdns` row, not in the image - serving the NPU at 93.65 / 94.03
tok/s and HRX at 305.09 tok/s, with 0 TDRs up a 400/600/1000-token ladder. Later board rows, c27's and c28's, are in
[releases.md](releases.md). Since cycle 20 the appliance
ships with **no API key** - `10lemonade` leaves `LEMONADE_API_KEY` unset and binds `0.0.0.0:13305` anyway,
and console item 1 turns a key on (`os/v3/puppy/embed/etc/init.d/10lemonade:43-62`). **Nothing is
published**: the next release carries it.

**There is no Ubuntu `autoinstall` any more, and no second plan to fall back to** (owner, 2026-11-06: *"puppy and
we will offer install packages for distros"*). Puppy is the one target, and `1bit-install` puts this same appliance
on the disk as a persistent OS — [The installed machine](#the-installed-machine-is-a-persistent-os-712). Other
distributions are reached with our **install packages**, not with this installer
([install-any-distro.md](install-any-distro.md)). Everything from
[The Ubuntu plan](#the-ubuntu-plan-history-not-a-target) to the end of the page is kept as the design history it
is: the package set, the gates and the `os/v3` tree rows there are still the reference for what a full 1bit OS
carries, but its layout, its `autoinstall` and its upgrade path describe a medium that is not built.

## The installed machine is a persistent OS (#712)

The installer's contract is still one line — `1bit-install /dev/sdX --yes` — and the shape it writes is still the
stick's: a 512 MB `1BIT-OS` ESP carrying the stick's own shim → grub chain, and one `1BIT-DATA` ext4 taking the rest
of the disk, holding the frugal install (`/1bit`) and the data contract (`/data`) on the same filesystem. What the
install adds, and what makes the installed machine an OS instead of the live image, is **two things, both written at
install time rather than left to a first boot**:

| | live stick | installed disk |
|---|---|---|
| the writable root | RAM (`PUPMODE=5`) — the session is thrown away on power-off, which is what a *try it* medium is for | a **save layer** `/1bit/<DISTRO_FILE_PREFIX>save` on `1BIT-DATA` (`PUPMODE=12`) — every write lands on the disk as it is made |
| the SFS | `pfix=copy`: copied into RAM, so a slow stick is read once | `pfix=nocopy`: mounted from the disk, ~1.4 GB of RAM not spent on a copy of the OS |

The save layer is a save **folder**, which is Puppy's own write-through root: the init finds it by name
(`find_save_file` looks for `<psubdir>/<DISTRO_FILE_PREFIX>save*`) and `setup_psave` makes `<folder>/upper` the
overlay's upperdir. Its name therefore comes from the image's own `/etc/DISTRO_SPECS` and not from this page, and
`1bit-install` **refuses to start** if it cannot read that — a guessed name installs cleanly and then boots with no
persistence at all. Puppy's own creator does the same two things (`shutdownconfig`'s `save_directory`: `mkdir` plus a
`.DirIcon`), and a layer with no `/initrd/DISTRO_SPECS` in it is a normal first boot: `rc.update`'s
`rc_update_force_pm5` pass rebuilds `/etc` from the SFS. A `SAVESPEC` copied from an installed source is removed,
because it names *that* machine's partition.

Both facts are asserted on the **installed disk's own boots**, not on the installer's log line (`install-probe.sh`):
`PUPMODE=12`, the root's upperdir under `/1bit/<prefix>save/upper`, no squashfs mount coming from `/mnt/tmpfs`, and
a write to `/root` that is still there after the guest reboots itself.

### The installed machine's account

An installed machine gets a normal account (`07onboard-user`), created on its first installed boot and kept after:
`onebit` with a home and bash, **the machine's own keys** (`/data/ssh/authorized_keys`, `0600`, re-applied at every
installed boot exactly as `20onboard-ssh` does for root), a **locked password** — a key is the only way in, there is no
console login for it and no `su` from another account — and **one** sudo rule,

```
onebit ALL=(ALL) NOPASSWD: ALL
```

appended to Puppy's `/etc/sudoers` once, backed up first and proved by running `sudo -n true` as that account, then
rolled back if it does not work (a broken sudoers costs root its own sudo). `NOPASSWD` widens nothing here: a key
authorised for this account is a key root already accepts. Puppy's own rules grant `%users` sudo only with the **root**
password (`Defaults:users runaspw`), which is no use to an account that has none — hence the explicit rule. The **live
stick stays root-only**: a machine that boots from RAM and forgets should not mint an account either.

### Updating an installed machine

An installed machine carries **two slots** on its data partition and **one** save layer:

| | |
|---|---|
| `/1bit` | `psubdir=1bit` — the appliance the machine was installed with |
| `/1bit-alt` | `psubdir=1bit-alt` — created by the first update; both slots' entries carry the same UUID-pinned boot line, so nothing else about the disk changes |
| the save layer | stays where the machine was installed (`/1bit/<prefix>save`) and the other slot reaches it through a **relative `upup64rsave` symlink** — Puppy's init finds a save folder under its own `psubdir`, follows the link and writes through it (`PUPMODE=12`), so both sets share one machine's state instead of forking it |

`1bit-update <release-dir | image.img>` writes the next appliance into the slot that is **not** running. It checks the
release first (it must carry the running set's own file names, and a `SHA256SUMS` is verified for those files when the
release has one), copies, then compares every byte, links the save layer, adds that slot's GRUB entries built from the
running entry's own boot line, and flips `set default` to the new headless entry — re-reading the config the way GRUB
counts entries before it reports success. The running set is never touched, so the previous appliance stays bootable:
that is the rollback. `1bit-update --rollback` flips `set default` back; `1bit-update --status` says which slot is
running, what each holds, and which entry GRUB boots.

**Updates repeat.** The layer the machine uses is the one the **running** slot holds or links to, so after the first
update (running `/1bit-alt`, whose link points at `/1bit`) the second one targets `/1bit` — the slot that *holds* the
layer. That directory is this machine's own state: it is kept as it is and not re-linked. Only a real layer that is
**not** the running one (another machine's, in the target slot) is refused, because two layers cannot be merged by
an update. "The running one" is decided by **file** (`-ef`, device and inode), which says "the directory the running
slot reaches" as one test instead of two — a clarification rather than a hole being closed: the slot name had to match as
well, and a foreign layer is by definition in another slot, so the name form refused it too (pi-os3 raised the case, the
#1385 review measured that the older form already refused it; the test arm is the regression guard either way). The
decision is a set of functions in `1bit-update` that `tests/one_bit_update_slots_test.sh` runs over fake data partitions
on every CI run (updates 1, 2 and 3; a foreign layer; a same-named foreign layer; a stale link), without root or a VM.

An update also refuses an unhashed release unless `--no-verify` says otherwise — this path replaces the appliance the
machine boots, so a `SHA256SUMS` is the default requirement — and the **image form verifies against the image's own**
`/1bit/SHA256SUMS`, which `mkimage.sh` writes from the bytes it just copied (before that, every weekly image would have
needed `--no-verify`). The ESP's free space is checked **before** the copy, next to the data partition's own check,
because the ESP is the one place a half-write is not recoverable without a rescue stick.

Which is not the same as package updates: `apt` (and the appliance's own payload) update **inside** the running set and
land in the save layer. An OS change — kernel, initrd, SFS — is this, and it is the only thing that writes a slot.

**Measured** (`os/v3/puppy/update-gate.sh`, three boots of one installed disk, the "next release" being that disk's own
SFS with one added file): the update reports `UPDATED`, the machine comes back on `psubdir=1bit-alt` with `PUPMODE=12`,
the new set's marker file present and the marker written before the update still there; `--rollback` brings it back to
`psubdir=1bit` with the previous appliance intact and the state unchanged.

### The server-shaped rest, present or named

An installed 1bit OS is a headless server, so the things a server is expected to have are either there and measured, or
named here as what they are:

| | what the machine has | measured |
|---|---|---|
| firewall default | Puppy's `/etc/init.d/rc.firewall` sets `-P INPUT/-P FORWARD DROP` on the v4 **and** the v6 stack, and `zz-onboard-ports` — last in name order, which is the mechanism: `rc.services` runs `/etc/init.d/*` in order and `rc.firewall` sorts under "r" — inserts the two inbound rules the appliance serves, tcp/**22** (ssh, the owner's key) and tcp/**13305** (Lemonade), plus mDNS, each with `-C` so a re-run is a no-op. **When** it is in place is part of the property: `rc.sysinit` starts `rc.services` in the background, so for the first seconds of a boot the policy is still ACCEPT and no port rule exists yet — the serial shell appears *before* the last script has run, because that shell is itself one of them | `install-probe.sh` on the installed boot, **after waiting** (bounded, and the wait is printed): `INPUT DROP` in both stacks, both ACCEPT rules, loopback and established/related exempted, and a port nothing serves (9999) in no rule. Measured on an installed disk 2026-11-07: in place **10 s** after the shell (and the row that did not wait reported "no firewall" — a measurement artifact, recorded so nobody trusts the earlier claim) |
| time | `connman` owns the network and syncs from `FallbackTimeservers = pool.ntp.org` (`/etc/connman/main.conf`); `ntpd`, `ntpdate` and `adjtimex` are in the image for a machine that wants a daemon | the installed boot's own connman config |
| the machine's name | `/etc/hostname` is `1bit-os`, and the console's network item sets it — on an installed machine that write lands in the save layer, so it **persists** (on a stick it is RAM) | the installed boot's `hostname` |
| logging | `06onboard-logs` keeps the boot's logs on `/data/logs` — the partition, not the filesystem that the next appliance replaces | `/data/logs` non-empty on the installed boot |
| packages and services | `apt` works and its packages land in the save layer; the appliance's own services are `/etc/init.d` scripts, which is what `rc.services` runs (there is no systemd here, and nothing pretends there is) | — |

**Not present, and named:** a **persisted static address**. The console's "set a static address" is *session only* on
purpose — a wrong one on a headless box is a walk to the machine — and the persistent form is not a one-line change: on
this base connman owns the interface, so it has to be a connman service config, with the console (or the install
screen, #710) as its recovery path. That is the remaining piece of engine#712's third item, and it is a product decision
rather than a row to add to a gate.

## The Ubuntu plan (history, not a target)

*Kept for the measurements and the reasoning, not as a plan: the owner removed this target on 2026-11-06 (see the
top of the page). Nothing below is built by any cycle.*

This is the design for engine#712. Today one image serves both roles: the live stick and the disk it
installs to are the same BusyBox initramfs, so the installed machine has no package manager, no users,
no updates and an empty efivarfs (engine#712, engine#709). v3 keeps that image as the *live* medium and
makes the **installed** machine a real OS: Ubuntu 26.04 LTS Server, installed unattended by `autoinstall`
from the same stick, carrying our kernel, our drivers, our runtime, the engine, Lemonade and the
appliance units as packages that `apt` updates.

The first slice is `os/v3/pin.txt` (the pinned server ISO), `os/v3/seed.sh` and its `seed-selftest.sh`
(**PR #732**, 32 checks, no device; not in this PR either, so this page stands alone at any merge
order). Everything after the seed — `mkstick.sh`, `mkdebs.sh`,
`postinstall`, `steps`, the vmtest scripts — is design. `os/mini` keeps shipping in v2026.44, and
engine#712 says v3 is the next release, not the Sunday one. One difference worth naming, because both
media are 1bit OS: **the Puppy appliance (`os/v3/puppy`) ships with no Lemonade API key at all** (cycle 20,
stock-Lemonade behaviour; console menu item 1 turns one on), while the `os/mini` image still **mints** one
into `/data/lemonade/lemond.conf` on first boot (`os/mini/init.sh`) because it was built when every stick was
meant to be keyed. Nothing in this page depends on either; `prepare-stick.sh --mint` is how a *shared*
network gets a key on both. What *is* measured is in
[What is measured now](#what-is-measured-now) — most of it on the Strix Halo box this design was written
on, which is itself the reference install: it runs Ubuntu 26.04.1 with our kernel, our out-of-tree
modules, TheRock's ROCm, the distro's XRT, the engine and Lemonade, and it serves HRX and the NPU.

## What changes, and what stays

| | today (`os/mini`) | v2 (`os/v2`, in tree, not shipped) | v3 (Puppy, what ships) |
|---|---|---|---|
| live medium | BusyBox initramfs, everything in RAM | Ubuntu rootfs squashfs + our debs | **the Puppy SFS on the stick, with the save layer on `1BIT-DATA`** |
| installed system | the same initramfs on the ESP + a `/data` partition | squashfs unpacked to ext4, our debs preinstalled | **a frugal install: the SFS on a partition, the save layer on `1BIT-DATA`** |
| updates | reflash the stick | `apt` from `1bit.gg/apt` | **`apt` and PETget are in the image; the OS itself is replaced by a new SFS, the save layer stays** |
| users | root only, keys from `/data/ssh` | root only, keys from `/data` | **`root` with `sudo`, keys carried from the stick, `sshd` on** |
| firmware entry | none (engine#709) | `efibootmgr -c` when it exists | **written during the install with `efibootmgr`** (engine#709) |
| efivarfs | absent (module never staged, engine#709) | present (`CONFIG_EFIVAR_FS=m`, systemd loads it) | **built in, mounted at boot** |
| console | BusyBox shell + `1bit-menu` | `1bit-menu` on tty1, autologin | **headless: a console login on tty1, no X; the `desktop` boot entry starts Puppy's JWM desktop** |

The `os/mini` row is the BusyBox image, and it stays what it is: the CI/VM target of the QEMU gates,
not 1bit OS. The rest of this page is the Ubuntu fallback's detail.

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
disk. `prepare-stick.sh` (already the tool that writes `authorized_keys`, and the one that can hand a stick
a Lemonade API key with `--mint`) gains:

```sh
prepare-stick.sh /dev/sdX --ssh-pubkey ~/.ssh/id_ed25519.pub \
    --user onebit --hostname 1bit-strixhalo --disk-serial YMA62T0KA2417407NP \
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
| 2 | 64 GiB | ext4 | `1BIT-ROOT` | the system, `/boot` (kernels and initramfs), `/var`, `/home`, an 8 GiB `/swap.img` | created |
| 3 | the rest | ext4 | `1BIT-DATA` | models, Lemonade's cache and config, `hrx-jit`, NPU sets, keys — `/data` | **adopted, never formatted** |

Numbers: 1 GiB is the Ubuntu default and the measured size of this box's ESP (1.1 GiB, 6.4 MB used, so
two kernels and their initramfs fit with room); 64 GiB for the root because a base Ubuntu 26.04 install
plus the 180 MB runtime below, its journal and `/var` fit in a tenth of that, and because the rest of the
disk belongs to the models — a fixed number, not a percentage, so the same seed gives the same table on
any disk (`size: -1` on partition 3 takes the remainder). No swap *partition*: Ubuntu's own `/swap.img` on the root
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
    # The archive, for the stock packages only. Ours come from the medium in postinstall and are *not* a
    # persistent source: a file:///cdrom URI left in the target's sources.list is a dead source the moment
    # the stick is out, so the installed machine's one source is 1bit.gg/apt (written by 1bit-appliance, as
    # v2 does), and postinstall installs the medium's repo through a temporary source file it removes.
    primary:
      - {uri: "http://archive.ubuntu.com/ubuntu", arches: [amd64]}
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
- **`fallback: offline-install`** is what lets a machine with no network install at all: curtin then takes
the stock packages from the ISO's own pool instead of aborting at `apt update`. Our own packages never
depend on the network, because they are on the medium.
- **`preserve: true` on partition 3** is the v2 → v3 path (see [Upgrade](#the-upgrade-path)). It is also
  what makes the installer idempotent: a second run over a v3 machine keeps `/data`.
- **Not used, deliberately**: `source:` (we install the stock server variant), `drivers:`/`codecs:`
  (they pull third-party repositories and prompts), `kernel-crash-dumps:`, `snaps:`, `network:`
  (netplan's DHCP default is right, and a fixed address belongs in the console menu, not the seed).

### `postinstall` — our steps

Runs in the target chroot (`curtin in-target`), so every path is the target's except `/cdrom`. Its steps
are the screen's steps 3–8, in this order (1 and 2 are the seed's identity line and curtin's own base
install):

| screen step | command | what it writes |
|---|---|---|
| 3 | `apt-get install` our repo's debs (`1bit-kernel-<kver>`, `1bit-kernel-modules-<kver>`, `1bit-rocm`, `1bit-xrt`, `1bit-engine`, `1bit-lemonade`, `1bit-appliance`, `1bit-archive-keyring`) | `/opt/1bit`, `/opt/lemonade`, `/boot/vmlinuz-<kver>`, `/lib/modules/<kver>/updates/{amdxdna,ec_su_axb35}.ko` |
| 4 | `update-initramfs -u -k <kver>`, `update-grub` | an initramfs with `amdgpu`, `amdxdna`, `efivarfs`, and one grub entry per kernel in `/boot` |
| 5 | `mokutil --test-key /cdrom/1bit/mok/MOK-Kernel.der` else `mokutil --import` (password on stdin, twice) | the MokNew efivars the next boot's MokManager reads |
| 6 | `efibootmgr -c -d <disk> -p 1 -L "1bit OS" -l '\EFI\ubuntu\shimx64.efi'`, `-o` first, `-n` BootNext, delete entries whose loader is not on this ESP | the firmware entry engine#709 asks for |
| 7 | adopt `/data`: `chown onebit`, `install -d models lemonade/cache lemonade/config cache/hrx-jit npu ssh`, copy `/data/ssh/host_ed25519` to `/etc/ssh/ssh_host_ed25519_key` | a machine whose SSH identity survives a reinstall |
| 8 | verify: `mount -o ro` both partitions, `sha256sum /boot/vmlinuz-<kver>`, `dpkg-query -W '1bit-*'`, `systemctl is-enabled` the unit set, `1bit-release-check` | `/data/install.log` on both media, the screen, ttyS0 |

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
 1  target disk and identity          /dev/nvme0n1  YMTC PC41Q-2TB  1.9 TB   user onebit  host 1bit-strixhalo
 2  base system (Ubuntu 26.04 Server)   curtin: partitions, unpack, packages, bootloader
 3  our repository from the stick      8 debs: kernel, modules, rocm, xrt, engine, lemonade, appliance, keyring
 4  kernel and modules                 vmlinuz-7.4.0-rc1-1bit, amdxdna, ec_su_axb35, initramfs
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
   `/boot/config-7.4.0-rc1-1bit`) and an Ubuntu base loads it, so `/sys/firmware/efi/efivars` is
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
supply chain: **`linux-firmware-amd-graphics` and `linux-firmware-amd-misc`** (26.04's split packages,
on two revisions - measured here, `-amd-misc` `20260319.git217ca6e4-0ubuntu1.1` and `-amd-graphics`
`20260319.git217ca6e4-0ubuntu3.1`; `-amd-misc` is "Firmware for AMD NPU accelerators", and it is
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

**User and ssh**: `onebit` in `sudo`, root locked, `ssh.socket` enabled with keys carried from the stick,
`PasswordAuthentication no`, `PermitRootLogin no`. The account is `onebit` and not `1bit` because a Linux
user name has to start with a letter — `adduser`'s default `NAME_REGEX` is `^[a-zA-Z][a-zA-Z0-9_-]*$`
(measured in `/etc/adduser.conf` here) and the installer applies the same rule, so `1bit` is refused
before the first byte is written; `seed.sh` refuses it too and its selftest pins that. Headless means the
console is the fallback, not the other way round: `1bit-menu` on tty1 can drop to a shell as `onebit`
(sudo), and ttyS0 gets a console unit so a serial cable works when ssh does not.

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
| SSH host key and `authorized_keys` | `/data/ssh/host_ed25519` → `/etc/ssh/ssh_host_ed25519_key`; `authorized_keys` → `/home/onebit/.ssh/` |
| Lemonade's API key | `/data/lemonade/lemond.conf` read as before — and **blank is the shipped default** (cycle 20): no key line means stock-Lemonade behaviour, and console menu item 1 turns one on |

| not carried, by design | why |
|---|---|
| v2's kernel and initramfs on the ESP | v3 keeps its kernels in `/boot` and its boot chain in the ESP; the old files are removed with the partition table |
| `/data/.installed` | v3 has no live/installed marker; `release-check`'s `mode` row reads the mount, not the file |
| root-only ssh | replaced by the `onebit` user; the same keys work |

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
| B (installed) | boot the installed disk alone, same NVRAM | `efivarfs` mounted; `dpkg-query -W '1bit-*'` = 8 packages; `systemctl is-enabled` the unit set; `ssh.socket` listening; `getent passwd onebit` and `sudo -l`; `/data` mounted from partition 3 by UUID; `apt-get update` and `apt-cache policy` from `1bit.gg/apt`; `unattended-upgrades --dry-run` clean; `1bit-release-check` PASS with `sealed-sig` now checked (`python3` present); `1bit-console` active on tty1 |
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

Reference box (strixhalo, 2026-11-01, the machine this design was written on). These are the readings
the design was written against and stay as the record; the box and the stick have run
`7.4.0-rc1-1bit` since 2026-11-04 (plain `v7.4-rc1`, `CONFIG_EFIVAR_FS=m` there too).

| fact | measured |
|---|---|
| base | Ubuntu 26.04.1 LTS, resolute |
| our kernel | `7.3.0-rc4-1bit-perfopt+`, `linux-image-7.3.0-rc4-1bit-perfopt+` 7.3.0~rc4-4, `CONFIG_EFIVAR_FS=m` |
| out-of-tree modules | `amdxdna.ko`, `ec_su_axb35.ko` in `/lib/modules/<kver>/updates/`, both signed `strixhalo Secure Boot Kernel`, vermagic `<kver> SMP preempt mod_unload modversions` |
| MOK | `CN=strixhalo Secure Boot Kernel, O=1bit-MONSTER`, self-signed, EKU Code Signing; Secure Boot enabled |
| firmware | `efibootmgr` 10 entries; `Boot0000 Ubuntu` → `\EFI\ubuntu\shimx64.efi`; stale `Boot0002 1bit OS USB`; 97 efivars |
| firmware packages | `linux-firmware-amd-misc` 20260319.git217ca6e4-**0ubuntu1.1** and `linux-firmware-amd-graphics` 20260319.git217ca6e4-**0ubuntu3.1** (two revisions, measured; the first is what carries `amdnpu/17f0_11/npu.sbin.zst`) |
| XRT | `libxrt-npu2`, `libxrt-utils-npu`, `libxrt2` 1:2.21.75+dfsg-4 |
| ROCm | `/opt/rocm-therock` 7.2 GB; the runtime subset 173 MB + 3.5 MB bitcode; `libhsa` `21a66824…` |
| the three libhsa | `f6ef689d…` (release), `21a66824…` (nightly), `5b113248…` (distro 7.1) |
| packages | kernel image 86 MB `.deb`, headers 10 MB; c18 `lemonade-onebit` 39 MB; live `.img.zst` 397 MB; live `.efi` 106 MB |
| v2 in the field | ESP 512 MiB + `1BIT-DATA`; `1bit-install` writes `/data/.installed`; no efivarfs, no `efibootmgr` (engine#709) |
| c18 gate | A/R/I/B green in QEMU, install writes ESP + `/data` (1.1 GB copied), the installed boot passes `1bit-release-check` 18 rows with `sealed-sig` SKIP for lack of python3 |
| the install medium | `ubuntu-26.04.1-live-server-amd64.iso`, 2,927,861,760 bytes, sha256 `cc8a95cd…`, last modified 2026-08-26, volume label `Ubuntu-Server 26.04.1 LTS amd64` — the label read from the ISO's own primary volume descriptor, which is what casper matches; all four in `os/v3/pin.txt` |

Not measured, and therefore the risk list for the build:

- **the medium itself has never been booted** — the pin is verified (size, sha256, label) but the QEMU
  phase I run is what proves casper finds `/cdrom` and subiquity takes the seed;
- **subiquity's screen handoff** — whether our step lines appear in the tail it displays, or whether the
  renderer must take tty1 after curtin (the gate decides; both keep the serial transcript complete);
- **`mokutil --import` reading its password from stdin** in a curtin chroot — if it refuses, the fallback
  is a first-boot oneshot that imports it with the console, and the doc's "no MOK" note for boxes whose
  key is already enrolled stays valid;
- **`preserve: true` over a v2 partition table** — curtin's behaviour on a disk whose existing layout does
  not match the config is the phase-V gate's job, and if it wipes, the fallback is `1bit-install
  --adopt-data` grown from the live entry's v2 shell, which has `blkid` and `sfdisk` and can write the
  seed's storage section with explicit ids after inspecting the disk);
- **the ROCm subset's completeness** — 173 MB is what `ldd` on the release's `llama-server` and
  `libggml-hrx.so` needs *here*, measured while the release still carried DwarfStar's `ds4` (removed 2026-11-02, out of
  the shipped build since v2026.44, [#757](https://github.com/1bit-MONSTER/engine/pull/757)), so today's
  set is no larger; a box with a different GPU path may want more, so `mkdebs` derives the
  list from those binaries rather than from a hand-written list;
- **the ISO repack** (`os/v3/mkstick.sh`) — extracting a pinned ISO to a partition keeps casper working on
  the media it was built for; the gate's phase I proves it, and if extraction is not enough the fallback
  is `loopback` booting the ISO file, which needs `iso-scan/filename=` and a second labelled partition.

## The `os/v3` tree

Everything in this table is what this page proposes; none of it lands in this PR. The two rows marked
**#732** are that PR's files.

| file | what |
|---|---|
| `pin.txt` | the server ISO exactly (url, size, sha256, volume label, date) and the series; the engine/kernel/ROCm pins stay the release's own |
| `mkstick.sh` | `mkrootfs`'s replacement: extract the pinned ISO to partition 2, add `/1bit/`, write the ESP from `os/mini`'s payload, `prepare-stick`-style identity |
| `mkdebs.sh` | v2's `mkdebs.sh` plus `1bit-rocm` (subset derived from `ldd`), `1bit-xrt` (Depends), the kernel/modules pair, and the archive-keyring |
| `seed.sh` | **#732**: renders `autoinstall.yaml` + `identity` from the stick's inputs, `check` validates a rendered seed (fields, mode, placeholders, and the YAML's structure with PyYAML), `selftest` proves both directions |
| `seed-selftest.sh` | **#732**: the `selftest` target — good-seed acceptances, 15 refusals with their reasons, and seven structure regressions (no `preserve: true`, a `file://` apt source, an unsubstituted placeholder, a root size the identity does not have, a `/data` that is not `1BIT-DATA`, an md5 hash, an ESP without `grub_device`); 32 checks, no device |
| `postinstall` | the nine steps, `set -e` with the failure trap that keeps the screen |
| `steps` | the renderer entry points `plan`, `step`, `failed` (v2's `os/mini/tui.sh`, Apache-noticed copy) |
| `vmtest-v3.sh`, `vmtest-install-v3.sh` | phases A/B and I/V, transcripts in `/data` as v2's do — the host reads the guest's transcript, so no port forwarding decides a verdict |
| `diag-v3.sh` | boot the installed disk and ask the guest what failed; judges nothing |

## Licence

The scripts and this page are Apache-2.0, like the rest of the repository. What the packages carry is in
[os/LICENSES.md](../os/LICENSES.md); the Ubuntu base adds the archive's own licences, and the appliance
package ships the aggregate under `/usr/share/licenses/1bit-os/` as v2's does.
