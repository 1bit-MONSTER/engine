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
# 1bit engine: local LLM inference on AMD Strix Halo (Ryzen AI MAX+ 395)

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Lemonade pin: last bump](https://img.shields.io/github/last-commit/1bit-MONSTER/engine?path=third_party%2Flemonade&label=Lemonade%20pin&color=f5c518)](https://github.com/1bit-MONSTER/engine/commits/main/third_party/lemonade)
[![Releases](https://img.shields.io/badge/releases-GitHub-24292f)](https://github.com/1bit-MONSTER/engine/releases)
[![Documentation: 1bit.gg](https://img.shields.io/badge/docs-1bit.gg-1779e1)](https://1bit.gg/)
[![Discord](https://img.shields.io/badge/chat-Discord-5865F2)](https://discord.gg/fa5m4Vawpa)

One binary to rule them all: the 1bit engine is a model-agnostic local LLM inference engine for the
AMD Ryzen AI MAX+ 395 (Strix Halo) on Linux. It serves models on the Radeon 8060S iGPU (`gfx1151`)
through AMD's HRX runtime, and models on the XDNA 2 NPU through our own NPU runtime, as a
backend of AMD's [Lemonade Server](https://github.com/lemonade-sdk/lemonade) and its OpenAI-compatible API.
Boot a USB stick, and the machine serves local LLMs on its GPU and NPU. That is the whole setup.

**GPU direction (2026-11-21): the GPU route is moving to Loom.** That is AMD's Loom model programs
on the HRX runtime, served by `loom-serve` behind `1bit serve`. On our Strix Halo box, straight Loom
beat our llama.cpp fork with its HIP kernels on every model we compared. A model moves only once it
passes the Loom gate: an interleaved A/B and an accuracy check against the current route
([#3158](https://github.com/1bit-MONSTER/engine/issues/3158)). The Loom route is in `1bit serve`
([#3166](https://github.com/1bit-MONSTER/engine/pull/3166), [docs/loom-route.md](docs/loom-route.md)):
a model listed in `config/loom-routes.json` is served by `loom-serve`. v2026.47 carries the Loom route in `1bit serve` and its first gated entry (Qwen3.8-27B UD-Q4_K_XL, [#3261](https://github.com/1bit-MONSTER/engine/pull/3261)). It activates only when that model's Loom card and `loom-serve` are installed, which this release does not ship, so every GPU model runs on HRX (with AMD's Loom kernels in ggml-hrx) by default; whole-model Loom programs arrive model by model. Which models have a Loom program is tracked in [#3156](https://github.com/1bit-MONSTER/engine/issues/3156). Until a
model moves, and for models with no Loom program, the llama.cpp fork with ggml-hrx (its Loom kernels)
and our HIP kernels serves it as the fallback. The NPU engine does not change ([docs/hrx.md](docs/hrx.md#direction-loom-on-the-gpu)).

## Get started

1. Download `1bit-os-v2026.47.img.zst` and `SHA256SUMS` from the
   [v2026.47 release](https://github.com/1bit-MONSTER/engine/releases/tag/v2026.47), which also carries the `lemonade-onebit` package and the release notes.
2. Verify it, unpack it, and write it to the whole USB disk (16 GB or more; `/dev/sdX` is the stick,
   not a partition):

   ```sh
   sha256sum -c SHA256SUMS --ignore-missing
   zstd -d 1bit-os-v2026.47.img.zst
   sudo dd if=1bit-os-v2026.47.img of=/dev/sdX bs=4M conv=fsync status=progress
   ```

3. Boot the machine from the stick. Where our key is already enrolled, Secure Boot can stay on: the chain
   is a Microsoft-signed shim → our MOK-signed grub → our MOK-signed kernel → our signed modules. A
   machine that has never trusted our key refuses the kernel, and the certificates are not on the stick,
   so enroll both public ones from the box that signed the image first — or turn Secure Boot off for now;
   see [docs/install.md](docs/install.md#6-secure-boot-and-our-mok).
4. Lemonade comes up with its catalog and web UI. Try it from the live session; to keep it, run
   `1bit-install <disk> --yes` there and the same system is on the disk. The install options, what each
   one erases and the Secure Boot notes are [docs/install.md](docs/install.md); connecting a client —
   `curl`, the web app, the phone app, the OpenAI SDKs — is [docs/clients.md](docs/clients.md).

**Updating an installed machine.** `sudo 1bit-update --check` says whether a newer release is published (add
`--channel test` for pre-releases), and `sudo 1bit-update --latest` downloads it, checks the release key's signature,
and installs it into the other slot; the running version stays as the rollback (`sudo 1bit-update --rollback`). The
console's item 5 does the same with a confirm, and the console header says when an update is available. In a browser:
`http://127.0.0.1:8081/` on the box itself, and `https://<box>:8444/` from the network only once
`sudo 1bit-update --web-password` has set a password ([the update pages](docs/os-v3.md#the-update-pages-local-and-lan-only-with-a-password)). These arrive
with the first release cut after v2026.49-rc1; on an earlier machine, run `sudo 1bit-update ./<image>.img` with the
unpacked image instead. How it decides what to trust: [docs/os-v3.md](docs/os-v3.md#in-system-updates-1bit-update---check-and---latest).

**1bit OS is a Puppy Linux appliance**: `ResolutePup64`, built with woof-CE on the Ubuntu 26.04
"resolute" base, with Lemonade Server and this engine embedded in it, running our MOK-signed
`7.4.0-rc1-1bit` kernel rather than the distribution's. It boots **headless** by default, to a
console of its own with no X, and a second boot entry, `1bit OS (desktop, JWM)`, starts Puppy's desktop
on the same system. The userland is an ordinary one - `bash`, `sudo` and `apt` work - and everything that has
to last (models, Lemonade's config and cache, SSH keys) lives on the `1BIT-DATA` partition
([os/README.md](os/README.md), [docs/os-v3.md](docs/os-v3.md)).

Already running Lemonade on Ubuntu? The `lemonade-onebit` package from the [releases page](https://github.com/1bit-MONSTER/engine/releases) adds the engine as a Lemonade backend; nothing else changes ([docs/lemonade.md](docs/lemonade.md)).

## What it is: one model-agnostic binary for the GPU and the NPU

One binary to rule them all. The 1bit engine is model agnostic: any Hugging Face architecture, any GGUF quantization, Ryzen AI NPU packs and ONNX graphs, served by one Lemonade backend on the AMD Ryzen AI MAX+ 395 (Strix Halo). It runs the Radeon 8060S iGPU through AMD's HRX runtime (today the llama.cpp fork with ggml-hrx and our HIP kernels; moving to Loom programs per model, once each passes the gate), and the XDNA 2 NPU through our own NPU runtime, behind the OpenAI-compatible API Lemonade already gives you. Every model is checked on the hardware before it ships; the models and their measured numbers are one page: [docs/models.md](docs/models.md).

## Milestones

| when | what |
|---|---|
| 2026-10-04 | The engine becomes HRX + our HIP kernels + the NPU engine, embedded in Lemonade; Vulkan, ROCm and CUDA builds removed ([RFC #213](https://github.com/1bit-MONSTER/engine/discussions/213), [#306](https://github.com/1bit-MONSTER/engine/pull/306)) |
| 2026-10-20 | One HTTP hop from Lemonade to the engine's server; the stock Ubuntu `lemond` runs the engine with no patch ([#475](https://github.com/1bit-MONSTER/engine/pull/475), [#483](https://github.com/1bit-MONSTER/engine/pull/483)) |
| 2026-10-21 | Every HRX pin change passes a hardware gate on Strix Halo before merge; Qwen3.8-27B decodes at 36 tok/s with MTP on our kernels ([#478](https://github.com/1bit-MONSTER/engine/pull/478), [docs/hw-gate.md](docs/hw-gate.md)) |
| 2026-10-23 | First release candidate built, booted from USB on the hardware with Secure Boot on ([docs/releases.md](docs/releases.md)) |
| 2026-10-25 | Microsoft BitNet b1.58 (TQ1_0 / TQ2_0) runs fully on HRX ([docs/models.md](docs/models.md)) |
| 2026-10-30 | The HRX pin carries the DeepSeek-V4.1 port; ONNX Runtime runs in-process with our own execution provider on the NPU; the NPU and the GPU are measured running at the same time ([#672](https://github.com/1bit-MONSTER/engine/pull/672), [#647](https://github.com/1bit-MONSTER/engine/pull/647), [docs/hybrid.md](docs/hybrid.md)) |
| 2026-11-01 | **v2026.44 is a draft**: candidate 19 is gated in software (1959/1959 kernel ops, NPU 82.03 tok/s, `VMTEST` pass through install and installed-disk boot, `VMTEST-RELEASE` 10 pass / 9 skip / 0 fail) but has **not** been booted or installed on hardware; candidate 20 takes the full gate, hardware boot and install included, before it is published ([docs/releases.md](docs/releases.md)) |
| 2026-11-02 | **1bit OS becomes a Puppy Linux appliance**: `ResolutePup64` (woof-CE, Ubuntu 26.04 base) with our MOK-signed kernel, Lemonade and the engine embedded, headless by default with a `desktop` boot entry, `apt` and a persistent `1BIT-DATA`; it boots on the board with Secure Boot on ([docs/os-v3.md](docs/os-v3.md), [os/README.md](os/README.md)) |
| 2026-11-06 | The installed appliance is a persistent system on the disk, not the live RAM image; it gets a normal account with `sudo` and takes the next appliance without losing its state ([#1291](https://github.com/1bit-MONSTER/engine/pull/1291), [#1302](https://github.com/1bit-MONSTER/engine/pull/1302), [#1334](https://github.com/1bit-MONSTER/engine/pull/1334)) |
| 2026-11-08 | A release installs every package it publishes before it is released ([#1368](https://github.com/1bit-MONSTER/engine/pull/1368)) |
| 2026-11-14 | The Bosgame BeyondMax's rear USB-A and Type-C ports work under Linux: a plug test on every external port passed once the keep-awake rule holds the xHCI controllers awake (USB4 devices still untested). The MTP drafter no longer crashes on a model's first request, and the HIP kernel and HRX pins moved forward through the hardware gate ([docs/axb35-usb-ports.md](docs/axb35-usb-ports.md), [#2431](https://github.com/1bit-MONSTER/engine/pull/2431), [#2453](https://github.com/1bit-MONSTER/engine/pull/2453), [#2461](https://github.com/1bit-MONSTER/engine/pull/2461)) |
| 2026-11-20 | Liquid AI LFM3-1.7B on HRX for smaller machines: prefill 7,925 tok/s, decode 153.7 tok/s, 2.05 GiB of GTT at 8k context; its NPU pack passes the registry check on the NPU ([docs/models.md](docs/models.md), [#2956](https://github.com/1bit-MONSTER/engine/pull/2956), [#3016](https://github.com/1bit-MONSTER/engine/pull/3016)) |
| 2026-11-21 | The GPU route starts moving to Loom: AMD's Loom model programs on HRX, served by `loom-serve` behind `1bit serve`, one model at a time once each passes an A/B and accuracy gate. The route is in `1bit serve`, chosen per model by `config/loom-routes.json`; a model gets an entry only after it passes, and Qwen3.8-27B UD-Q4_K_XL is the first ([#3261](https://github.com/1bit-MONSTER/engine/pull/3261)). The llama.cpp fork with our HIP kernels stays as the fallback; the NPU engine does not change ([#3166](https://github.com/1bit-MONSTER/engine/pull/3166), [docs/loom-route.md](docs/loom-route.md), [#3156](https://github.com/1bit-MONSTER/engine/issues/3156), [#3157](https://github.com/1bit-MONSTER/engine/issues/3157), [#3158](https://github.com/1bit-MONSTER/engine/issues/3158)) |
| 2026-11-22 | **[v2026.47](https://github.com/1bit-MONSTER/engine/releases/tag/v2026.47) is the first published release** ([release post](https://1bit.gg/blog-v2026.47-release.html)). The release build, c34, boots from the stick and installs to the internal NVMe on the board with Secure Boot on (33 of 38 release-check rows on each, the five open rows named for the next release); Qwen3-0.6B decodes at 91.7 tok/s on the NPU and Qwen3-4B ships on the NPU; the NPU and the GPU serve at the same time ([docs/releases.md](docs/releases.md#v202647-22-nov-2026--the-first-published-release)) |

The release goes out when every item on the release checklist is done ([docs/releases.md](docs/releases.md)); the weekly Lemonade pin moves with Lemonade's releases.

## Documentation and measured results

- [1bit.gg](https://1bit.gg/): the docs site, built from this repository's `docs/`; [the blog](https://1bit.gg/blog.html) is the build log.
- [HRX on the Radeon 8060S](https://1bit.gg/hrx.html) and [the NPU engine](https://1bit.gg/npu.html): the two device routes, on the site.
- [docs/architecture.md](docs/architecture.md): how the engine is built, what runs where, the hardware gate, the kernel, the direction.
- [docs/serve.md](docs/serve.md): `1bit serve` and its options.
- [docs/loom-route.md](docs/loom-route.md): the Loom GPU route behind `1bit serve`, its route table and its gate.
- [docs/zero-copy.md](docs/zero-copy.md): the copies left on the serving paths, what each one costs per token, and why it stays.
- [docs/lora.md](docs/lora.md): adapters (LoRA), changing a live adapter's factors in place, and the two traps around them.
- [Measured results](https://github.com/1bit-MONSTER/engine/wiki): every number with how it was taken.
- [Site repository](https://github.com/1bit-MONSTER/site): builds 1bit.gg from this repository.

## License

The source code in this repository is Apache-2.0. See [LICENSE](LICENSE).

The release binaries (the Linux package and 1bit OS) also contain two proprietary components,
compiled in: the 1bit NPU runtime and the 1bit HIP kernels for HRX. They are not in this
repository, their source is not published, and they are not under Apache-2.0: they ship under the
[1bit Binary License](LICENSE-BINARY) (free to use for any purpose and to pass on unmodified;
third-party parts in [NOTICE-BINARY](NOTICE-BINARY)). The HIP kernels serve the fallback GPU route
(the llama.cpp fork) and get no new work now that the GPU route is moving to Loom. A build from this
repository without them has no NPU engine and runs HRX on the public kernels only
([NOTICE](NOTICE), [docs/releases.md](docs/releases.md#private-components)).

## How to credit

Credit is welcome and, for the code, required: the Apache-2.0 license keeps the copyright
notices and the [NOTICE](NOTICE) file with every copy. Three lines are enough:

```
1bit engine - https://github.com/1bit-MONSTER/engine - Copyright 2026 bong-water-water-bong - Apache-2.0
```

[CITATION.cff](CITATION.cff) carries the same in citation form (GitHub shows it under
"Cite this repository"). If you build on a specific piece, name the file or the pull request it
came from; the history is public.

## Thank you

**[Osmantic / ODS](https://github.com/Osmantic/ODS) comes first.** ODS introduced me to
vibecoding, and that is where all of this started. Without it, this engine would not exist.

**The Lemonade team and AMD's developers.** [Lemonade](https://github.com/lemonade-sdk/lemonade)
is where this engine runs. AMD's developers left breadcrumbs all over the place: the XDNA driver
and XRT, IRON and Peano, HRX and Loom, their tested llama.cpp integration, their issues, their examples.
This engine is what following those breadcrumbs built. The GPU route's move to Loom follows the Loom
model programs that [loom-fun](https://github.com/loom-fun) (Jeremy Fowers and Daniel Holanda)
publish for Strix Halo.

The repositories this engine is built on, in order of importance:

| # | Repository | What it gives this engine | License |
|---|---|---|---|
| 1 | [amd/xdna-driver](https://github.com/amd/xdna-driver) | The XDNA 2 NPU kernel driver (`amdxdna`) the NPU engine runs on | GPL-2.0 (driver) |
| 2 | [Xilinx/XRT](https://github.com/Xilinx/XRT) | AMD's NPU runtime and its API, where the engine's NPU work started | Apache-2.0 (userspace) |
| 3 | [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) | IRON and aiecc: how our own NPU kernels are written and compiled | Apache-2.0 WITH LLVM-exception |
| 4 | [Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie) | Peano, the C++ compiler for the NPU's AI Engine cores | Apache-2.0 WITH LLVM-exception |
| 5 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF inference on the Radeon iGPU through HRX (AMD's tested pair), and on the CPU | MIT |
| 6 | [ROCm/hrx-system](https://github.com/ROCm/hrx-system) (our fork [1bit-MONSTER/hrx-system](https://github.com/1bit-MONSTER/hrx-system) carries two commits) | HRX, AMD's HIP Runtime Extended, behind the HRX0 device, and its Loom compiler | Apache-2.0 |
| 7 | [ROCm/hrx-loom-library](https://github.com/ROCm/hrx-loom-library), [loom-fun/loom-skills](https://github.com/loom-fun/loom-skills), [loom-fun/weaver](https://github.com/loom-fun/weaver) | Loom: the kernel corpus for the HRX Loom compiler, and the tools and model cards for writing, tuning and serving Loom model programs on Strix Halo, where the GPU route is moving (in progress, [#3156](https://github.com/1bit-MONSTER/engine/issues/3156)) | MIT |
| 8 | [torvalds/linux](https://github.com/torvalds/linux) | The kernel, with `amdxdna` and `amdgpu` in-tree | GPL-2.0 WITH Linux-syscall-note |
| 9 | [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | Every model's `tokenizer.json`, byte-exact, behind our C ABI | Apache-2.0 |
| 10 | [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | The router that decides where each request runs | Apache-2.0 |
| 11 | [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM) | The Q4NX NPU model format and its models on Hugging Face (`FastFlowLM/*-NPU2`), which the engine's NPU route runs on its own kernels | MIT |
| 12 | [antirez/ds4](https://github.com/antirez/ds4) (DwarfStar) | **removed from the engine** (owner 2026-11-01, out of the tree 2026-11-02): antirez's CUDA kernels through a HIP shim plus hipBLAS are not our HIP kernels, and `--device ds4` now says it is gone. The fork [1bit-MONSTER/ds4](https://github.com/1bit-MONSTER/ds4) stays in the open as the reference for that work | MIT |

Also built on [nlohmann/json](https://github.com/nlohmann/json) (MIT)
and [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT).

Every third-party copyright and license is listed in [NOTICE](NOTICE). Each project keeps its
own license; nothing here relicenses anyone's work.
