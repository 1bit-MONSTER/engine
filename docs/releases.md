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
# Releases

The first release, **v2026.44**, is a **draft**: it is not published, and nothing is tagged — see
[v2026.44](#v202644-1-nov-2026--the-first-release-a-draft) for why, and for the candidate that replaces the one it was
built from.

1bit engine is pinned upstream: every backend it builds is a submodule at a fixed commit
(`third_party/`), and the `bump-*` workflows open a PR whenever an upstream moves. The engine
runs inside Lemonade, which releases on Fridays. The release run (`scripts/weekly.sh`) rebuilds
everything from scratch at the current pins, so each release carries that week's upstreams.

Releases are tagged `v<ISO year>.<ISO week>` (`v2026.40`), like Lemonade's `2026.39.1`; a second
release in the same week gets `.1`.

## v2026.44 (1 Nov 2026) — the first release, a draft

**Not published.** Candidate **c19** (`rc-2026-11-01a`, engine `bcb28cc8f573`) passed every software gate below and
was written to a 64 GB stick — and that is where it stopped: it was **never booted on hardware and never installed on
hardware**, so the owner pulled v2026.44 back to a draft. Nothing here is a hardware result for c19.

c19 is replaced by **candidate 20** (`rc-2026-11-01b`): built from `main` by `scripts/weekly.sh` once the issue board is
clear, and gated with the whole checklist — the same software gates as below **plus the two that decide a release**: a
live boot from the USB stick with Secure Boot on, and a full install to an internal disk, each with
`1bit-release-check` **18 checks / 0 failures**. c20 is what becomes v2026.44 when the owner publishes.

c19's raw image (what the `.img.zst` decompresses to): **2,395,996,160 B**, sha256
`3c3c08c28072c534071bf87de141cf85f1f9e2b3d306c236153dcf80b3a4b588` — the value a stick write and its readback are
verified against.

**HIP kernels only:** this image has no DwarfStar backend — `--device ds4` was removed (owner 2026-11-01, out of
the tree 2026-11-02), because that route is antirez's CUDA kernels through a HIP shim plus hipBLAS, not our kernels.
DeepSeek V4.x, GLM 5.x and Qwen3.8-Flash-Next run on **HRX** here, and the catalogue carries no `ds4` recipe.

| component | pin |
|---|---|
| engine | `bcb28cc8f57387865ecdbe744a7e3cdcf65eb7ad` |
| HRX (llama.cpp fork, ggml-hrx) | `4a58da9b087d` |
| hrx-system | `82804197bcbe` |
| HRX toolchain (ROCm/TheRock SDK) | `10.3.0a20261031`, `gfx1151` (`config/hrx-toolchain.config`), HSA runtime `21a6682452e159cf…` |
| Lemonade Server | `2026.46.0~1.ef0f3b49` (our fork, the `onebit` recipe) |
| kernel in the image | `7.3.0-rc4-1bit-perfopt` (IOMMU PerfOpt backports, MOK-signed) |
| NPU kernel sets | 3 sealed sets, 30 kernels, every manifest signed |

The pinned toolchain's HSA runtime, so a bump is a diff here and a wrong runtime is visible
without a GPU (`scripts/fetch-rocm-sdk.sh --check` prints the tree; the engine's first HRX backend
logs this same file and digest — `docs/hrx.md`, "The HSA runtime"):

| what | value |
|---|---|
| `libhsa-runtime64.so.1` in `rocm-sdk-10.3.0a20261031-gfx1151` (the pin a dev tree uses) | 5,141,585 B, sha256 `21a6682452e159cfcdc6bdc84676f6b7f9c1236f3a2957a97a2e7170889cdfa3`, ROCr build id `1.22.0-local-build-430d3d8f`, clang 24 |
| the runtime the image ships, `/opt/rocm-therock/lib/libhsa-runtime64.so.1` (`/opt/1bit/libhsa` → that dir) | 4,957,049 B, sha256 `f6ef689d163b42ded68cb99ffa4f9103251f05ba91fc3ca70f4a1d1909d9c34a`, ROCr build id `1.21.0-local-build-d134b782`, clang 23 — a different compilation, so this is the digest a release check's `identity` row prints |
| measured capabilities, Strix Halo `gfx1151` (rocminfo from the pinned tree, 2026-11-03) | `XNACK enabled: NO`, `Coherent Host Access: FALSE`, `Memory Properties: APU`, Runtime Version 1.22, System Timestamp Freq. 1000 MHz |

Match each digest to its own row: a dev build logs the **pin** (`21a66824…`), while `1bit-release-check`'s
`identity` line reports the **image copy** (`f6ef689d…`, unchanged on the c17, c19 and c20 board boots). The
image's ROCr 1.21.0 exports every one of the 121 HSA entry points the pinned HAL resolves by name, so the two
are interchangeable by symbol set (1.22 adds five the driver never resolves). Two things the digest alone does
not tell you, both measured:

* **Every binary in the set must be there.** `dlopen` of a single-file copy of those same bytes in a directory
  without its eight `librocm_sysdeps_*.so*` siblings fails (`librocm_sysdeps_elf.so.1: cannot open shared object
  file`) — the same digest, in a directory that cannot load it. The image installs the full 10-file flat set, and
  that one loads. Digest **plus closure** identifies a usable runtime, so check both (`sha256sum`, `ldd`).
* **A distro `libhsa-runtime64` cannot work.** ROCr `1.18.0` (`5b11324814140eb5…`, 3,902,416 B) lacks
  `hsa_amd_queue_create`, one of the 121, so an HRX process that resolves the bare soname from the system path gets
  no device (engine#713's acceptance table). A payload dir may also carry the pin **relocated** (`f5462dc3…`: the
  same 5,141,585 B with 185 bytes changed, `DT_RPATH`→`DT_RUNPATH $ORIGIN`).

The Lemonade pin is the sha (`ef0f3b49`), not the string around it: `tools/version.py` derives the dev version as
`<week>.0~<commit count>.<sha>`, and that count is the number of commits in the **checkout it was built in** —
`third_party/lemonade` is a `shallow = true` submodule, so a depth-1 clone reports `~1` for the same commit while a
full clone reports `~1752` (both measured on 2026-11-02; c18 printed `~5`, i.e. built from a deeper clone). The `~N` is
reproducible from the build tree, not from the release: it can differ between two candidates carrying the same code.

Artifacts of the gated c19 candidate — sizes and sha256 exactly as its own `SHA256SUMS` records them:

| file | size | sha256 |
|---|---|---|
| `1bit-os-v2026.44.img.zst` | 395,224,089 B | `c3a132e2a4d13fbdfac5350fedd63b59232e334a25d656e960fbba3ea64572dc` |
| `1bit-os-v2026.44.efi` | 105,110,144 B | `57023fb8143184db9840b7bdfd25f5134f846e516dd86ffbf6260f0697b66b7a` |
| `lemonade-onebit-v2026.44-linux-x86_64.tar.zst` | 25,061,222 B | `c6f55c73fd34406f92dcbbf2347b545686bd99dfdc52204953296fabb1c87e31` |

What ran on **c19** (Strix Halo and QEMU, 1 November) — every gate except the two hardware rows:

| check | result |
|---|---|
| build, `ctest`, package | PASS, **54/54** tests |
| hardware gate at the candidate's pins, on the box's GPU | **1959/1959** kernel ops (76 s, 0 ring timeouts), fa-decode 6/6, KLD 5 models (6 PASS rows), needles 5 PASS rows, placement all-on-HRX |
| NPU unit, on the box | **82.03 tok/s** (`oracle`, `greedy64`, `prefill`, `reproducible` all PASS) |
| QEMU `vmtest` (live boot → install → boot the installed disk) | **PASS**, including `MODE: INSTALLED on nvme0n1` and the engine's llama-server running from that disk |
| QEMU `vmtest-release` (`/bin/1bit-release-check` over SSH) | **PASS** (10 PASS, 9 SKIP, 0 FAIL — the SKIPs are the rows that need hardware a VM does not have) |
| `sha256sum -c SHA256SUMS`, `sbverify`, module/kernel identity | 3/3 OK, signature OK, one module tree for the shipped kernel |
| stick write | device readback `3c3c08c2…` **= the image**, `prepare-stick` 11/11 and `--check` exit 0, `/data` grown to the whole device |
| no DwarfStar | **0** `ds4/` files and **0** `hipblaslt` references; the image's catalogue has no `ds4` recipe |

**What did not run on c19: the hardware boot and the hardware install.** The only hardware evidence in this document is
candidate **c18**'s — a different, earlier candidate (`rc-2026-10-31c`): live USB boot with Secure Boot on,
`1bit-release-check` **18 checks / 0 failures** (HRX chat 299.5 tok/s, CPU chat 178.6 tok/s, TTFT 21.8 ms), then a full
install to the internal NVMe and a boot from it, **18/0 again** (HRX 301.8 tok/s, decode bench 178.3 tok/s). Those
numbers are c18's and must not be read as a hardware result for c19 or c20.

What c20 has to show before v2026.44 is published (the runbook is `C20-HARDWARE-TEST.md` in the release lane): the same
software gates above, then a **live USB boot on the board** with Secure Boot on — `1bit-release-check` 18 checks, 0
failures, with the nine hardware rows **PASS rather than SKIP** — and a **full install** with `1bit-install` (sha256
readbacks at step 6, the expected `#709` partial SKIP at step 9), a boot from the installed disk with the stick out
(`INSTALLED on nvme0n1`), and the same 18/0 check again.

What the os/v2 pass fixed in this image, against the c18 live-USB dry run: the boot screen is quiet while the serial
transcript keeps the kernel log (#703); the banner and `/etc/1bit/release-identity` carry the version from the shipped
package, not a constant (#704); one boot flow with a visible timer and live network/Lemonade progress (#705); a 16x32
console font and a full-width selection bar (#707); the sound modules are out and `mkimage` refuses unsigned modules
(#708); `prepare-stick` grows `/data` to the stick (#706); the installer shows every step with progress, sha256
readbacks and a `/data/install.log` (#710); the boot chain names its own data partition, so the banner says
`LIVE from USB /dev/sda` or `INSTALLED on nvme0n1` (#711).

Known issues in this image:

- **#709 is partial**: the shipped kernel has no `efivarfs` module, so the installer cannot register a firmware boot
  entry. It ships `efibootmgr`, does everything else, and says on screen that the disk boots as
  `\EFI\BOOT\BOOTX64.EFI` — pick the disk in the firmware menu once, or let the firmware fall back to it. The
  complete fix lands with 1bit OS v3 (#712).
- the installed appliance is still the live image on disk; the next release makes it a headless Ubuntu server (#712).

## v2026.43 (candidate, 23 Oct)

Built from `main` by the release run; **nothing is tagged or published yet** — the owner publishes.
The run records the artifacts, their checksums and every check's log beside the candidate build, and
the package rows below are filled from that record and are marked *pending* until it exists.

Release sha and pins (the submodules this candidate is built from):

| component | pin |
|---|---|
| engine | `7871c71d` (main when the last release-blocking PR merged; a later merge today re-runs only the stages it touches, and the run's record carries the final sha) |
| llama.cpp (HRX fork) | `851fa6c8d8f2` |
| hrx-system | `82804197bcbe` |
| Lemonade (fork) | `e58c9f111` (the fork's `main` tip; `bump-lemonade` re-pins `third_party/lemonade` to whatever it is at cut time, so re-read this row from `git ls-tree` then) |
| private kernel add-on (HIP) | `744bd713d968` |
| private NPU kernel set | `17ed5be7a28a` |
| private NPU runtime | `ce9fa59adf95` |
| Laya | source `6f0496f797dc`, model `7b928d828b7b` |
| amdxdna (out-of-tree NPU driver) | `526c608197` |
| tokenizers | `3012770722` |

| artifact | status |
|---|---|
| `lemonade-onebit-v2026.43-linux-x86_64.tar.zst` | pending |
| `1bit-os-v2026.43.img.zst`, `1bit-os-v2026.43.efi` | pending |
| `binary-drop-v2026.43.tar.zst` | pending |
| `changes.json`, `SHA256SUMS` | pending |

| check | status |
|---|---|
| `weekly.sh build test package` at the pins above | pending |
| `package_checks` (the packaging's own leak scan: no plaintext NPU artifacts, no private symbols) | pending |
| `1bit check` from the packed tarball on a clean prefix | pending |
| hardware gate, all phases (ops, fa-decode-q8, kat, kld, needle, placement, **npu**) | pending |
| OS image VMTEST (`os/mini/vmtest.sh`) | pending |
| binary drop built from a clean tree with no private parts | pending |

### What moved since the last rehearsal (18 Oct)

Every line is a merged PR; the numbers are measured, and the ones measured on Strix Halo carry the
power state they were taken in. **lemonade bench standard** = Lemonade's own CI benchmark (reload
before every run); **ours** = our sampler/batched standard (see the table legend in
the auto-opt database).

**HRX speed (ours).** Qwen3.8-27B UD-Q4_K_XL with MTP, greedy 256-token chats: **36.0 / 26.8 / 29.9
tok/s** on code / prose / short, from 25.9 / 20.5 / 21.9 before the speculative verify pass was
reworked — the step it took is 121 ms to 88 ms per step (`#487`). Decode without MTP stays 12.2 tok/s.

| change | PR | measured |
|---|---|---|
| speculative verify pass rework | `#487` | 27B `--mtp` **+31–39 %** |
| batched MoE decode | `#521`, `#526` | pin `744bd713d968`: batching pays from five tokens per window; the batched cells below were measured one pin earlier (90 W cap, PCM pad (23 Oct)) |
| decode attention for gated attention at long context | `#500` | Qwen3.5-4B 32K decode **23.7 → 47.1 tok/s**; 27B **7.5 → 11.0** |
| MoE formats for Qwen3.8-Flash-Next | `#494` | **5.1 → 26 tok/s**, 50 with MTP |
| HIP prompt flash attention, causal block skipping | `#491` | Qwen3-4B pp4096 **+45 %** |
| m128 prefill tiles + raster groups | `#490` | 27B prefill **+1.6…2.5 %** |
| query-chunk prompt FA, ROPE token chunks, int4 SwiGLU opt-in | `#493` | 27B KLD base re-recorded; 27B runs `-ub 4096` |
| a second warm-up at load builds the small-batch weight layout | `#498` | 27B chat-short TTFT **2.64 → 0.80 s** |
| warm the backend before the first client request | `#492` | removes the 5–23 % slower first request |
| MTP drafts with the model's own head on HRX by default | `#495` | `--print-backend` carries the warm-up |
| `--precise`: a follow-up turn bitwise identical to a cold run | `#504` | identity, not speed |

**Batched, ours standard** (N concurrent 256-token chat requests, `-c 8192` per slot, median of 3,
**90 W cap, PCM pad (23 Oct)**), aggregate tok/s at N=8: Qwen3.8-27B **50.7** (2.08x N=1, with the
model's own MTP draft), Qwen3-Coder-30B-A3B **138.1** (1.64x), Qwen3.5-4B **194.3** (3.39x). Single
stream in the same cells: Coder-30B 84.2, 27B 24.3 (MTP on; 12.2 without), Qwen3.5-4B 57.3. Lemonade's
Vulkan build reaches 245.7 on Coder-30B at N=8 in the same run — that gap is batching headroom, not a
model result, and it is the next speed row.

**Correctness and placement.** Strict device placement is the default: a graph with any node off HRX
refuses to load, and the hardware gate fails the same condition (`#482`, `#473`, `#442`). The census
closes **43 of 43** configurations with 0 nodes off the GPU and one graph split (`#476`, `#480`,
`#481`), the gate now covers q8_0 KV rows, a mixed K/V flash-attention case, an MoE model, an IQ
quant and a second needle (`#474`, `#477`, `#484`), and the KLD bases were re-recorded at the release
pin (`#456`). `#315` is **resolved**: the CWSR-on soak passed 37 arms over 9 h 03 min with 0 faults
(`#518`), after the CWSR save area moved into GTT (`#454`) and `hrx-system` gained its
`HRX_ALLOW_NO_SVM` gate (`#455`). A lost GPU now exits the server instead of serving from a broken
context (`#466`), allocation safety bounds the program cache and poisons a device after a fault
(`#486`), and the ZAYA1-8B NaN re-check closed on **120 fresh loads, 0 NaN** at the current pins
(`#433`).

**NPU (sealed).** The gate gained an NPU phase and the packages carry **only the sealed kernel
sets** — a plaintext artifact fails the packaging check, and the launcher refuses an unsealed one
(`#503`, `#505`, `#509`, `#512`). The 0.6B gate phase measures **90.33 tok/s**; the OS image proves
`amd_pmf` sensors and the NPU from the booted image (`#517`, `#523`). The runtime's public header and
prebuilt-library build are in the tree (`#467`, `#479`, `#469`).

**Lemonade.** One hop for GGUFs: Lemonade starts the engine's `llama-server` directly, no shim
(`#475`, fork `#15`), the engine's own warm-ups are sent in order (`#492`, fork `#16`, `#17`), and
the Ubuntu 26.04 PPA's `lemond` runs our HRX server unchanged (`#483`). Synced to upstream `#3771`
(fork `#21`, `#520`), and now to the GA release itself (`v2026.45.3`, fork `#29`) — the port carries our
onebit backend, catalog and FLM context clamp into upstream's rebuilt server, so the suite evidence for
this pin is a run against the **ported** `lemond`, not against a package built before it.

The catalog has **23 HRX entries**, each with a proven row on the box (strict placement, one request in
role, `lemonade bench ... chat-short`): fork `#23`'s 22 plus flint from `#24`. It also has **4 NPU
entries** — `Qwen3-0.6B`, `Qwen3-1.7B` and `Qwen3-4B` (fork `#19`, `#20`, `#23`) and
`Qwen3.6-35B-A3B` (`#8`). Chat-short spans 43.4 (Qwen3-8B) to 330.7 tok/s (Qwen3-0.6B), 0 faults in 16
holds.

Two behaviours are new on this pin, both from the fleet's NPU-exclusivity decision (2026-11-05): **an
NPU model switch evicts the previous NPU model before the new one starts**, because the array runs one
hardware context at a time (fork `#34`), and a *refused* context is queued on a budget instead of being
called a dead array (engine `#1100`, `1bit serve --load-queue-ms`). The 31-test suite through the
onebit recipe on HRX passes against the ported `lemond` (`Ran 31 tests in 110.6s`, `OK (skipped=2)`,
measured at load 28.5 with two foreign HRX clients resident), and the catalogue covers 27 onebit rows
in total (23 + 4).

**1bit OS.** The image carries the sealed NPU sets and the new Lemonade (`#512`, `#523`), the UKI is
signed with the enrolled MOK so a stick boots with Secure Boot on (`#513`), the USB4 v2 / Thunderbolt
stack is loaded (`#514`), and `amd_pmf` sensors come up from the booted image (`#517`). The install
step is not in this candidate (branch `1bit/os-install`).

**Tooling.** `scripts/gpu-hold`: a fair FIFO GPU queue, used by the hardware gate and the release
runs, which honours the box's thermal brake and cools down after a hot hold (`#501`, `#502`). Census
and pins kept current daily (`#506`, `#524`), and the kernel is now a written runbook with a
rehearsal at `v7.3` and linux-next `next-20261022` (`#525`).

## What a release carries

| file | what |
|---|---|
| `lemonade-onebit-<tag>-linux-x86_64.tar.zst` | Lemonade (`lemond`, `lemonade`, its web app) from our fork with the onebit recipe, and the engine in `1bit/`: `1bit` and every backend it ships: **HRX** (and the CPU, from the same llama-server) and the NPU engine's runtime. DwarfStar is out of the tree since 2026-11-02 (HIP kernels only, owner decision 2026-11-01); no Vulkan or ROCm llama.cpp since RFC #213 stage 3 |
| `1bit-os-<tag>.img.zst`, `1bit-os-<tag>.efi` | 1bit OS, a Lemonade appliance with the engine as a bootable USB image ([os/README.md](../os/README.md)) |
| `changes.json`, `SHA256SUMS` | what moved upstream since the last release, as data; checksums |

The engine is embedded in Lemonade, so it ships only inside those two: there is no standalone
engine tarball and no Windows package (Lemonade has its own Windows build). On the GPU the engine
is pure HRX with HIP kernels: no Vulkan, ROCm backend, Metal, ZINC or CUDA. Lemonade serves those
with its own backends.

The engine in `1bit/` keeps the build tree's layout (`scripts/package-linux.sh`): `1bit` finds each
backend at the same path relative to itself as in the build (`app/built_path.h`), and every program
finds its libraries beside itself. Unpack the package anywhere and start Lemonade through
`lemond-onebit`, which points Lemonade's onebit backend at the packaged `1bit`
(`LEMONADE_ONEBIT_BIN`):

```sh
tar --zstd -xf lemonade-onebit-v2026.40-linux-x86_64.tar.zst
lemonade-onebit-v2026.40/lemond-onebit           # lemond with the onebit recipe on the packaged engine
```

The builds target Strix Halo: the llama.cpp backends are built with `GGML_NATIVE` on its Zen 5. HRX builds with the
pinned toolchain (`config/hrx-toolchain.config`, `scripts/fetch-rocm-sdk.sh`), or TheRock in `/opt/rocm-therock` when
that is what the box has; the NPU needs the `amdxdna` kernel driver. The closed NPU add-ons
([npu.md](npu.md#add-on-routes)) are never in a release.

### The NPU engine

The release is the HRX runtime with our HIP kernels and the NPU engine, all inside Lemonade. The
release carries the NPU runtime, compiled into `1bit`, and the `onebit` recipe's `npu` backend.
The NPU kernels are closed too: they ship as sealed artifacts that `1bit` verifies and opens in
memory at load, failing cleanly on a
wrong key with no network callback of any kind; a lifted kernel file is ciphertext and does not run
([npu.md](npu.md)). The release package carries them, and serves dense Qwen3 (0.6B, 1.7B, 4B)
on the NPU.

`scripts/weekly.sh` packs them by default (`WEEKLY_NPU_KERNELS=0` builds without): the sealed
kernel sets go beside `1bit` in `npu/kernels`, the build loads sealed kernels only
(`ONEBIT_NPU_RELEASE`), the `lemond-onebit` launcher sets `ONEBIT_KCRYPT_REQUIRE_SEALED=1`, and the package checks refuse a plaintext kernel,
an ELF for anything but the host, a symbol table in `1bit`, or a private source name in it. The
release machine holds the kernel key in `ONEBIT_NPU_KEY_DIR` (default `~/.config/1bit-npu-key`).

Sealing is confidentiality; each set is signed too (#574). Before a set is packed it gets
`manifest.json` - every file the set ships, with the sha256 of its ciphertext - and `manifest.sig`,
Ed25519 over that manifest by the release signing key (`tools/seal_manifest.py write <set> --key
<release key>`, `ONEBIT_SEAL_KEY`). Packaging requires both: `scripts/package-linux.sh` and the
weekly `package_checks` refuse a set without them or a manifest that does not match the files
packed, and verify the signatures when `ONEBIT_SEAL_PUB` names the release public key.
`1bit-release-check` proves the same on a written stick, so a set that was re-sealed or swapped
fails the checks even though it still opens.

### Private components

Two components of the release binaries are proprietary and compiled in; their source is not in
this repository or in any release. They ship under the [1bit Binary License](../LICENSE-BINARY):
anyone may use them for any purpose, including commercial use, and pass on exact unmodified
copies with the license and notices; modifying them, reverse engineering them (except where the law
expressly allows it) and removing notices are not allowed. The third-party code inside them is listed
with its licenses in [NOTICE-BINARY](../NOTICE-BINARY). Every release package and the 1bit OS image
carry both files beside the engine's `LICENSE` and `NOTICE`:

| Component | Compiled into | Pin | Build option |
|---|---|---|---|
| The 1bit NPU runtime ([npu.md](npu.md#the-runtime)) | `1bit` (static) | `config/npu-runtime.config` | `ONEBIT_NPU_RT_DIR` (or a prebuilt `ONEBIT_NPU_RT_LIB`) |
| The private HIP kernels for HRX ([hrx.md](hrx.md), "Private HIP kernels") | `hrx/llama/bin/libggml-hrx.so` (code objects) | `config/gpu-kernels.config` | `ONEBIT_GPU_PRIVATE` |

The build stage fetches each with `scripts/fetch-private.sh <pin> <dir>`: the pinned path at the
pinned commit only, with the release machine's `gh` login (read access to the private
repositories). The run stops if either cannot be fetched or is not at its pinned sha. Move a pin
by hand, after the NPU check or the HRX checks on Strix Halo.

### The public binary drop

A source build outside the org has no private HIP kernels and no private NPU runtime, so it cannot
compile those parts itself. `scripts/make-binary-drop.sh <engine checkout> <out dir>` packs them
already compiled, for one engine pin, as `binary-drop-<version>.tar.zst` + `SHA256SUMS`:

```
binary-drop-<version>/
  hrx/llama/bin/        llama-server, llama-bench and the libraries beside them, libggml-hrx.so
                        with the private HIP kernels compiled in
  npu/libnpu_rt_api.a   the NPU runtime behind the engine's public npu/npu_rt_api.h
  npu/include/          npu_rt_api.h (the engine's own public header)
  LICENSE-BINARY NOTICE-BINARY LICENSE NOTICE
  MANIFEST              the pins below, the build host, the compilers, sha256 of every file
```

It takes the HRX tree from a release build (`--build <dir>`, packed the way `package-linux.sh`
packs it: the same `scripts/pack-elf.sh` steps copy the libraries it loads beside it, set RUNPATH
to `$ORIGIN` and strip it) or from an already packed release (`--from-package <tar.zst>`), and the NPU library either prebuilt (`--npu-lib`) or built from the
pinned private sources (`--npu-rt <dir>`, its `build-api-lib.sh`). It then runs the same class of
checks the release package runs: the NPU runtime compiled in, the kernels really are compiled in, no source files,
nothing named like the private sources, both binary licenses present, every library found. It publishes nothing; the
orchestrator publishes the tarball and adds its line to `config/binary-drop.config`.

To build against a drop, point CMake at it — a directory or the tarball's URL:

```
cmake -B build -DONEBIT_HRX=ON -DONEBIT_NPU=ON -DONEBIT_BINARY_DROP=/path/to/binary-drop-<version>
# or -DONEBIT_BINARY_DROP=https://…/binary-drop-<version>.tar.zst
```

A URL is only accepted when `config/binary-drop.config` pins its sha256 for the engine commit being
built (`drop=<engine sha> <version> <sha256>`); `file(DOWNLOAD)` refuses a body that does not match.
A directory is verified against its `MANIFEST` — every file's sha256, and the engine commit it was
built from — and a mismatch, or a drop built for another commit, stops the configure. With a drop in
place, `ONEBIT_GPU_PRIVATE` and `ONEBIT_NPU_RT_DIR` are not needed and not used: the build links
`hrx/llama/bin/llama-server` and `npu/libnpu_rt_api.a` from the drop, and does not build llama.cpp,
so the `llama.cpp` and `hrx-system` submodules are not required either. HRX still needs TheRock's
HSA runtime on the machine that runs it.

The NPU engine's runtime is compiled into `1bit`; no packed program needs a separate NPU runtime
library. `scripts/package-linux.sh` strips every packed program and library (`strip
--strip-unneeded`) and makes each find its libraries beside itself (`$ORIGIN`, `scripts/pack-elf.sh`),
with LLVM's OpenMP runtime (`libomp.so`, which the HRX llama.cpp libraries link when TheRock's
amdclang builds them; credited in `NOTICE-BINARY`) packed in `hrx/llama/bin`. After packing the engine tarball the package stage checks
(`logs/package-checks.log`; a failure stops the release):

- **Self-contained NPU runtime:** for the build's `1bit` and every program and library in the
  tarball, no separate NPU runtime library is needed or packed.
- **Kernels in:** the packed `libggml-hrx.so` exports the add-on's `ggml_hrx_hip_addon_register`.
- **No source:** no source file (`.hip`, `.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hpp`, `.inc`, `.cu`,
  `.cmake`) and no file named like one of the private checkouts' files. The three public GGUF-on-NPU
  helper scripts are the stated exception: `scripts/repack_gguf.py`, its declared-scale guard
  `scripts/check_repack_config.py` and the lane packer `tools/gguf_to_q4nx.py` are the engine's own
  Apache-2.0 source, published in this repository, and a packed install cannot serve a GGUF on the
  NPU without them ([npu.md](npu.md#from-a-gguf)). The check fails if a tarball carrying the NPU
  kernels is missing any of the three.
- **Licensed:** `LICENSE`, `NOTICE`, `LICENSE-BINARY` and `NOTICE-BINARY` are in the tarball.
- **Libraries found:** `ldd` (with `ROCR_VISIBLE_DEVICES=-1`) finds every library each packed
  program and library needs, except the TheRock runtime the package documents (HIP, hipBLAS(Lt),
  rocBLAS, HSA - the same libraries the reference DwarfStar build linked).
- **Links inside:** every symlink in the engine tarball and in the Lemonade + onebit tarball is
  relative and resolves inside the package (`pack_outside_links`); the `libggml*.so` ->
  `.so.0` links stay links. Lemonade's build links its web app's KaTeX fonts to the build
  machine's `/usr/share/fonts/truetype/katex`; the package stage and `os/mini/mkimage.sh` copy
  every link that points out of the tree as a file (`pack_localize_links`, `scripts/pack-elf.sh`).
- **Web app:** the Lemonade + onebit tarball carries Lemonade's web app
  (`bin/resources/web-app/index.html`), the UI `lemond` serves at `/`, as 1bit OS does.

`SHA256SUMS` is written again by the release stage, after `changes.json`, so it covers every file.

To exercise only the private builds and the packaging without releasing, on Strix Halo:

```sh
WEEKLY_DRY_RUN=1 WEEKLY_TAG=rehearsal-private scripts/weekly.sh ~/.cache/1bit-weekly build package
```

## The release run

`scripts/weekly.sh <workdir>` on Strix Halo, in stages:

1. **bumps**: each open `bump-*` PR is merged onto `main` locally, built with every backend, and
   tested: `ctest` (with the `serve` end-to-end tests on Qwen3-0.6B) plus the check that bump needs,
   below. A PR that passes is brought up to date, waits for CI, and is merged. One that fails stays
   open, and the release lists it as held back.
2. **build** and **test**: `main` at the merged pins, the same way.
3. **package**: the two packages above. `scripts/package-linux.sh` packs the engine first; its
   tarball goes into the Lemonade + onebit package as `1bit/`, not into the release on its own.
4. **release**: `tools/weekly_changes.py` lists, for every pin that moved since the last release,
   the upstream commits (authors, PRs) and releases in between; that becomes the release notes and
   `changes.json`, and `gh release create` publishes it all.

`WEEKLY_DRY_RUN=1` merges and publishes nothing; stages can be run one at a time
(`scripts/weekly.sh ~/.cache/1bit-weekly build test`).

Before the first stage runs, `weekly.sh` checks what the machine must have (the 2026-10-19
rehearsal found these only after the build) and stops with a message naming what is missing:

- **bumps, build:** the pinned HRX toolchain, which `preflight` itself fetches with
  `scripts/fetch-rocm-sdk.sh` at the pin in `config/hrx-toolchain.config` (`-DONEBIT_HRX_TOOLCHAIN=<dir>` in
  `WEEKLY_CMAKE_ARGS` uses another, e.g. TheRock in `/opt/rocm-therock`); `hipcc` from that toolchain or `/opt/rocm`,
  or `ROCM_PATH`. (The optional DwarfStar build that also wanted it is gone since 2026-11-02.)
- **package:** `patchelf` (`scripts/package-linux.sh`); `node` and `npm` for Lemonade's web app
  (`LEMONADE_WEB_APP=1 scripts/build-lemonade.sh`; its `npm ci` needs the network); a readable kernel image for 1bit OS,
  `/boot/vmlinuz-$(uname -r)` or `VMLINUZ=<copy>` (Ubuntu's `/boot/vmlinuz-*` is root-only).
  Which kernel that image should carry, and how a new one is built, signed and gated before it is
  allowed on the box, is [kernel.md](kernel.md).
  The image's licences step copies from the submodules of the run's own checkout (`<workdir>/src`,
  checked out by the build stage), so run `os/mini/mkimage.sh` from there, not from a plain
  worktree without submodules.

### Bumps

| PR | check beyond build + ctest |
|---|---|
| `bump-lemonade/*` | Lemonade's LLM suite through the onebit recipe on HRX ([lemonade.md](lemonade.md)) |
| `bump-laya/*` | `tests/laya_route_e2e.sh` and `tests/laya_route_serve.sh` on the pinned checkpoint |
| `bump-hrx/*` | every `hrx` row of `registry/check_models.tsv` through `tools/registry_check.py` (serve end to end per architecture, ZAYA1 and the Qwen MoE models included); any FAIL holds the bump |
| `bump-xdna/*`, `bump-tokenizers/*` | covered by build + ctest (serve end to end on each device); the xdna-driver pin is the `amdxdna` kernel driver, which the release does not build: check it on Strix Halo with the module loaded ([npu.md](npu.md), "The XDNA driver") |

A pin bump raised **by hand** (rather than by one of those workflows) has to do what the workflow does, or the
gates above silently do not apply to it (engine#672):

- **branch it `bump-<name>/<sha>`** or add the `hw-gate` label - `hw-gate.yml` runs on `bump-hrx/*` or on that label
  (it reads `hw_gate: skipping` otherwise, which looks like a pass), and `scripts/weekly.sh`'s `bumps` stage only picks
  up `bump-*` heads, so a differently named branch is invisible to the Sunday run as well;
- **carry the data derived from the pinned backends**: `registry/architectures.json` is generated
  (`tools/registry_build.py` with the submodules at their pins), and a class that starts being mapped needs its reviewed
  entry in `registry/significant.json` (`mapped_ok`, or `--check-gaps` refuses it) plus its expectation in
  `tests/census_watch_test.py` updated - otherwise `registry_pins` and `census_watch` fail at the new pin (engine#672:
  DeepSeek-V4.1-Flash went from "reviewed report-only" to mapped);
- run the checks in the table above, plus `hw-gate` (see [hw-gate.md](hw-gate.md), "When it runs").

Before the build, every bump also checks that each pin it moves is on the branch `.gitmodules`
names for that submodule (GitHub's compare API: the branch equals or contains the pin). A pin off
its branch is held: the shallow clones follow the branch and cannot fetch it, and GitHub may drop
an unreferenced commit.

## The announcement

After the release, the 1bit engine Discord gets a new post in #announcements, written from
`changes.json` in the layout of Lemonade's release posts: what the week brings, the notable upstream
changes with credit to their authors, the rest as a list, and the downloads (the two packages
above). Then two lists written
straight from the data: **New models** (repos published under our Hugging Face org, models added to
Lemonade's catalog, architectures the registry gained or that gained a backend) and **Updates** (every
upstream as pinned in the release, with its latest release and changelog). Each release gets its own post, and earlier ones stay.
