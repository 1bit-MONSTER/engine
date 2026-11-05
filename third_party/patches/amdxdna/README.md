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

# The amdxdna patches this fleet's builds run (not upstream)

`third_party/xdna-driver` pins upstream `amd/xdna-driver` at **`526c6081974e`** ("accel/amdxdna: honour a user-requested
partition start column"). That pin is **pre-fix** for three defects that take the NPU down on this hardware, and none of
the fixes is upstream, so **a module built from the pin alone has none of them**:

| # | patch | fixes | where it is measured |
|---|---|---|---|
| 0001 | `accel/amdxdna: fix the bo_invalid_list use-after-free at close` (4 files, +48/−4) | a client that closes while a BO sits on `client->bo_invalid_list` frees the client and leaves the BO linked into it → `kernel BUG at lib/list_debug.c:32` in `amdxdna_hmm_invalidate` | the 2026-11-04 01:48 panic on strixhalo; shipped since ~03:0x; **#971** carries the image-side half of shipping it |
| 0002 | `accel/amdxdna: recover the array after a TDR` (5 files, +91/−19) | a TDR restart gated on a condition a fully-wedged context never satisfies, so `aie2_hwctx_restart()` fails `-22` and every later request hangs until a reboot | provocation run 2026-11-04 10:48–10:50 on the loaded build: the recovery branch executes, the NPU is power-cycled, the array answers again **with no reboot**; **#1005** |
| 0003 | `accel/amdxdna: two TDR-recovery outcomes are not failures` (3 files, +60/−4: `aie2_tdr.c`, `aie2_pci.c/h`) | 0002's recovery latches `hw_dead` — *"the NPU array is unusable until the driver is reloaded"*, every later open refused `-ENODEV` — for two cases that are not failures: **(a)** a power-cycle inside the 60 s rate-limit window answers `-EBUSY`, and with `tdr_timeout_ms=15000` the watchdog ticks four times inside that window, so **any two TDRs within 60 s killed the array** (the check is now `aie2_hw_reset_throttled()`, tested *before* any context is stopped: throttled = retry next tick); **(b)** a context whose client is already closing has no host buffer address, its restart is refused `-ENODEV`, and that one context condemned the array although the power-cycle had succeeded — it is now skipped (its jobs are already unpaused) | strixhalo 2026-11-04: **21:50:23** one closing context, **22:28** two TDRs inside 60 s, **22:40** two closing contexts — three `hw_dead` latches in 50 min, each costing every lane the NPU until a resume/reload (evidence `~/amd-oss/hrx-study/evidence/fabric/npu-longctx/incident-20261104-215023/`). Owner word 2026-11-04 ~23:5x: *"go A"* — built/signed/installed by the kernel-74 lane from the same patch (`~/wt/xdna-tdr-recover/patches/0008-…`) |

> **What actually runs (2026-11-05 00:10 AST):** the fleet's installed series is the kernel lane's numbering —
> **`0008`** (owner word "go A", installed, srcversion `6483D31718A008B3602114E`), **`0009`** (my throttle-ordering fix:
> test the rate-limit window *before* stopping any context, because 0008 still restarted contexts against an array that was
> never power-cycled and `rarg->failed` could latch `hw_dead` through that) and optionally **`0010`** (root-only debugfs
> `tdr_test` hook that runs one recovery pass on demand). Files: `~/wt/xdna-tdr-recover/patches/`. This branch's `0003` is
> the same defects in an earlier shape and is **not** byte-identical to what is installed; fingerprints for the installed
> series with the pin's `MODULE_VER_STR` are `BBE1E5F0EE7A5FB235CA811` (0008+0009) and `646FFC305CAFF8F9BD736FF`
> (0008+0009+0010).


They are a **series**: 0002 applies after 0001 and 0003 after 0002, all against `526c6081`. Applying 0001+0002 to a clean
checkout of the pin reproduces, byte for byte, the sources the module loaded on the box before 21:52 was built from
(checked by `cmp` over every `drivers/accel/amdxdna/*.c` and `*.h`), and rebuilding that tree reproduces that module's
fingerprint — see *Expected fingerprints* below. 0003 is the patch the owner approved as **"go A"** on 2026-11-04 ~23:5x; it builds clean against this kernel
(`make -C drivers/accel/amdxdna modules ARCH=x86_64` → rc 0, no warnings) and the kernel-74 lane installs and proves it
from the same file, under the series name `0008`.

## Building, signing, installing, loading

`KERNEL_VER` is the kernel the module is for; a new kernel means a new build, and a module built for a different kernel
will not load (vermagic). Build through the driver's own `modules` target, not `make -C <kdir> M=…` directly: it
generates `config_kernel.h` by feature-testing the target kernel, and that step needs `ARCH`.

```sh
git clone https://github.com/amd/xdna-driver && cd xdna-driver
git checkout 526c6081974e                                  # the engine's pin
git apply <engine>/third_party/patches/amdxdna/0001-*.patch # order matters
git apply <engine>/third_party/patches/amdxdna/0002-*.patch
git apply <engine>/third_party/patches/amdxdna/0003-*.patch # the hw_dead escalation fixes (owner word "go A")

make -C drivers/accel/amdxdna modules ARCH=x86_64 KERNEL_VER=$(uname -r) \
     MODULE_VER_STR='2.21.0_20261104,526c6081974ea4f0f7dd22329186c5d6adb8fad5'
```

`MODULE_VER_STR` is optional (the driver defaults it to `<version>_<today>,<HEAD>`), but it is what makes the built
module's fingerprint comparable with the table below, because `srcversion` hashes the built module's contents and the
version string is part of them — a rebuild of *identical sources* with a different string came out at
`1184D9F60CB1099D47420E9` when measured on 2026-11-04, so a srcversion mismatch on its own does not mean the patches
are missing. Check the source with `cmp`/`git diff`, and the fingerprint with the same string.


Under Secure Boot the module must be signed before it can load (`MOK-Kernel` here; the MOK must already be enrolled):

```sh
cp drivers/accel/amdxdna/amdxdna.ko /tmp/amdxdna.signed.ko
sudo /lib/modules/$KERNEL_VER/build/scripts/sign-file sha256 \
     /var/lib/shim-signed/mok/MOK-Kernel.priv /var/lib/shim-signed/mok/MOK-Kernel.pem /tmp/amdxdna.signed.ko
modinfo -F signer /tmp/amdxdna.signed.ko                    # must name the enrolled key

sudo cp -n /lib/modules/$KERNEL_VER/updates/amdxdna.ko /lib/modules/$KERNEL_VER/updates/amdxdna.ko.bak-$(date +%Y%m%d)
sudo modprobe -r amdxdna                                   # requires the NPU to be free: take an npu-hold ticket first
sudo install -m 644 /tmp/amdxdna.signed.ko /lib/modules/$KERNEL_VER/updates/amdxdna.ko
sudo depmod -a $KERNEL_VER && sudo modprobe amdxdna
```

**Regenerate the initramfs, or the next reboot undoes the install.** On 2026-11-04 this box installed 0002 at 10:23 and
then booted the *pre-patch* module for three reboots, because `/boot/initrd.img-$KERNEL_VER` still carried the old
`updates/amdxdna.ko` and the boot load takes the initramfs copy over the file in `/lib/modules/`. `modinfo -n`, `lsmod`
and `modinfo -F srcversion` all reported the file and none of them could see it.

```sh
sudo update-initramfs -u -k $KERNEL_VER
```

## Expected fingerprints

`srcversion` is what to assert, because it is a property of the built module rather than of the file sitting next to it.
Both values below were produced on strixhalo against the `7.4.0-rc1-1bit` headers, with
`MODULE_VER_STR='2.21.0_20261104,526c6081974ea4f0f7dd22329186c5d6adb8fad5'`:

| build | `srcversion` |
|---|---|
| pin + 0001 | `EBC78CDA6FD7F3F6E824064` |
| pin + 0001 + 0002 (what the box and the c22/c23 images run) | `56AB94758C18FEEC838D18C` |
| pin + 0001 + 0002 + **0003** (built 2026-11-04 22:47, the `go A` patch) | `5E616DCAA2C1AD65076E3CB` |

Reproducing the second one is a measured result, not an inference: on 2026-11-04 the pin + the two patch files + the
version string above built to `56AB94758C18FEEC838D18C`, which is exactly `cat /sys/module/amdxdna/srcversion` on the
box that had been running that build since 10:23.

**Check the loaded module, never the file:**

```sh
cat /sys/module/amdxdna/srcversion                 # what the kernel is running
modinfo -F srcversion /lib/modules/$KERNEL_VER/updates/amdxdna.ko   # what is on disk — can differ, see above
```

A build that comes out at the pin's own srcversion, or at a value that is not in the table, means either the patches
were not applied or the version string differed — check both. An image whose zdrv carries a different `amdxdna.ko`
than the module tested on the board means the board tested something nobody measured.

## When the pin moves

`.github/workflows/bump-xdna.yml` moves the submodule by hand, and a move invalidates this series: re-apply the patches,
rebuild, re-run the 0002 provocation (a TDR must power-cycle the NPU and the array must answer afterwards), and update
the table above before the new pin is merged. `git apply --3way` is the sane way to see whether upstream moved under
a hunk.

The kernel's own in-tree `amdxdna` (kernel 7.4-rc1 and later) is a **rewrite**, not this driver: nothing here applies to
it, and it carries neither fix.
