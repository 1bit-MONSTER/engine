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
partition start column"). That pin is **pre-fix** for two defects that both take the NPU down on this hardware, and
neither fix is upstream, so **a module built from the pin alone has neither**:

0003 is a different kind of row: a **candidate**, built but not yet proven on hardware (see its own row). It touches only
`aie2_solver.c`, `aie2_solver.h` and `aie2_ctx.c`, so it applies on top of any of the TDR series below and does not depend on which one is loaded.

| # | patch | fixes | where it is measured |
|---|---|---|---|
| 0001 | `accel/amdxdna: fix the bo_invalid_list use-after-free at close` (4 files, +48/−4) | a client that closes while a BO sits on `client->bo_invalid_list` frees the client and leaves the BO linked into it → `kernel BUG at lib/list_debug.c:32` in `amdxdna_hmm_invalidate` | the 2026-11-04 01:48 panic on strixhalo; shipped since ~03:0x; **#971** carries the image-side half of shipping it |
| 0002 | `accel/amdxdna: recover the array after a TDR` (5 files, +91/−19) | a TDR restart gated on a condition a fully-wedged context never satisfies, so `aie2_hwctx_restart()` fails `-22` and every later request hangs until a reboot | provocation run 2026-11-04 10:48–10:50 on the loaded build: the recovery branch executes, the NPU is power-cycled, the array answers again **with no reboot**; **#1005** |
| 0003 | `accel/amdxdna: never share an array partition between two clients` (3 files, +23/−2) — **candidate, not installed anywhere in this shape yet** | the solver marks every partition `exclusive = false` ("always set exclusive to false for now"), so a second *client* with no free column range is handed the *same* columns as a live one (`nshared++`) and its configuration takes the array out from under the first: every client then wedges (`614`, `DPU PC 0xffffffff`). The patch records the owning client in `alloc_requests.owner` and only shares a partition between contexts of the same client — the per-op route that holds one context per design keeps working, a second process is refused | measured 2026-11-05 on the c24 payload under `npu-hold`: two clients that had each dispatched once shared `Partition Index 0, Columns [0..7]` and the next dispatch wedged 3/3 sequential and 3/3 concurrent, while one context was clean in every row (1 000–3 408 positions) and a second serve that never dispatched was clean too. A **per-context** version of this patch was measured live and **refused the second context of one client's multi-context route (ZR1 per-op forward, 3/3)** — that is why the unit is the client, not the context. The per-client shape's own proof (second client refused while the first finishes; two contexts of one client still load) is owed — it needs this build installed |

They are a **series**: 0002 applies after 0001, both against `526c6081`. Applying them to a clean checkout of the pin
reproduces, byte for byte, the sources the module the c22/c23 images run was built from (checked by `cmp` over
every `drivers/accel/amdxdna/*.c` and `*.h`), and rebuilding that tree reproduces the loaded module's fingerprint — see
*Expected fingerprints* below.

## Building, signing, installing, loading

`KERNEL_VER` is the kernel the module is for; a new kernel means a new build, and a module built for a different kernel
will not load (vermagic). Build through the driver's own `modules` target, not `make -C <kdir> M=…` directly: it
generates `config_kernel.h` by feature-testing the target kernel, and that step needs `ARCH`.

```sh
git clone https://github.com/amd/xdna-driver && cd xdna-driver
git checkout 526c6081974e                                  # the engine's pin
git apply <engine>/third_party/patches/amdxdna/0001-*.patch # order matters
git apply <engine>/third_party/patches/amdxdna/0002-*.patch
git apply <engine>/third_party/patches/amdxdna/0003-*.patch # optional, a candidate

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
All values below were produced on strixhalo against the `7.4.0-rc1-1bit` headers, with
`MODULE_VER_STR='2.21.0_20261104,526c6081974ea4f0f7dd22329186c5d6adb8fad5'`:

| build | `srcversion` |
|---|---|
| pin + 0001 | `EBC78CDA6FD7F3F6E824064` |
| pin + 0001 + 0002 (what the box and the c22/c23 images run) | `56AB94758C18FEEC838D18C` |
| pin + 0001 + 0002 + 0003 | `FA680E5F8334570CA318449` |

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

## What this directory does not carry (yet)

The TDR series that the box and the c25 candidate actually run is **newer than 0002**: `0008` (two recovery outcomes are
not failures: a throttled power-cycle and a closing context must not latch `hw_dead`) and `0009` (check the throttle
before touching any context) live in `~/briefs/TKERNEL-74-STATE.md`'s patch list and are built/installed by the kernel
lane, not from this directory. A module built from `pin + 0001 + 0002 + 0003` therefore reproduces a coherent stack but
**not** the module `/sys/module/amdxdna/srcversion` reports on strixhalo (`646FFC305CAFF8F9BD736FF` on 2026-11-05, which
is 0007 + 0008 + 0009 + the 0010 debugfs test hook). Carrying those two files here is the obvious next step and is not
done, because the one honest way to do it — `cmp` the rebuilt sources against the tree the running module came from —
needs that tree, which is the kernel lane's.
