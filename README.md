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
# 1bit engine

**Documentation:** [1bit.gg](https://1bit.gg/) · **Measured results:** [wiki](https://github.com/1bit-MONSTER/engine/wiki) · **Community:** [Discord](https://discord.gg/fa5m4Vawpa)

**Site:** [github.com/1bit-MONSTER/site](https://github.com/1bit-MONSTER/site) (builds 1bit.gg from this repository's README.md and docs/)

**The 1bit engine is AMD's HRX runtime, our own HIP kernels on the Strix Halo GPU, and our NPU
engine, all inside [Lemonade](https://github.com/lemonade-sdk/lemonade).** On a Ryzen AI Max+ 395,
every graph node of every model and mode we test runs on the GPU: 43 of 43 configurations in the
census, with 0 nodes on the CPU ([#476](https://github.com/1bit-MONSTER/engine/pull/476),
[#480](https://github.com/1bit-MONSTER/engine/pull/480),
[#481](https://github.com/1bit-MONSTER/engine/pull/481)). Qwen3.8-27B (UD-Q4_K_XL) with MTP
speculative decoding decodes at 36.0 / 26.8 / 29.9 tok/s on code / prose / short chats
([#487](https://github.com/1bit-MONSTER/engine/pull/487)).

**Where it stands on 23 October 2026** (candidate `v2026.43`; numbers carry their standard and their
power state — *lemonade bench standard* is Lemonade's own CI benchmark, *ours* is our sampler/batched
standard, and everything measured after the 23 Oct 01:38 rebuild is *90 W cap, PCM pad (23 Oct)*):

| what | number | standard |
|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL with MTP, code / prose / short chat | **36.0 / 26.8 / 29.9 tok/s** | lemonade bench standard |
| Qwen3-Coder-30B-A3B, single stream | **84.2 tok/s** (85.9 in the lemonade-bench cells) | ours / lemonade standard |
| batched, aggregate at 8 concurrent 256-token chats: 27B / Coder-30B / Qwen3.5-4B | **50.7 / 138.1 / 194.3 tok/s** | ours, 90 W cap, PCM pad (23 Oct) |
| NPU, 0.6B gate phase | **90.33 tok/s** | ours (gate) |
| census on HRX: configurations with 0 nodes off the GPU | **43 of 43** | ours (gate) |
| Lemonade catalog: HRX entries proven on the box | **22 of 22** (43.4–330.7 tok/s chat-short) | lemonade bench standard |

The engine is in the release candidate now; the tag and the packages are the owner's
([docs/releases.md](docs/releases.md#v202643-candidate-23-oct)).

Lemonade stays the server you talk to (its catalog, downloads, router and UI). For the models it
hands to 1bit, it runs the engine as one of its backends, behind an OpenAI-compatible API:

- **HRX on the Radeon iGPU** ([docs/hrx.md](docs/hrx.md)): AMD's ggml-hrx with its Loom kernels
  and our HIP kernels, the engine's one GPU route. For a GGUF, Lemonade starts the engine's
  `llama-server` directly, one HTTP hop
  ([#475](https://github.com/1bit-MONSTER/engine/pull/475),
  [docs/serve.md](docs/serve.md#one-hop-from-lemonade---print-backend)). Placement is strict by
  default: the whole graph runs on HRX, or the load stops with the node that would not
  ([#482](https://github.com/1bit-MONSTER/engine/pull/482),
  [docs/serve.md](docs/serve.md#off-hrx-placement-strict-by-default)). `--device cpu` runs the
  same build on the CPU.
- **The NPU engine** on the XDNA 2 NPU, in progress: correctness first, speed next
  ([docs/npu.md](docs/npu.md)).
- **Hybrid**: the prompt on the NPU engine, the tokens on HRX. `1bit serve --device hybrid` hands
  the prompt's KV cache from a prefill provider to the HRX llama-server
  ([docs/hybrid.md](docs/hybrid.md)).
- **DwarfStar**, for DeepSeek V4 Flash, GLM 5.x and Qwen3.8-Flash-Next in its own GGUFs, with SSD
  expert streaming ([docs/dwarfstar.md](docs/dwarfstar.md)).
- **Laya**, opt-in, which picks where each request runs ([docs/laya.md](docs/laya.md)), and
  **the Project Manager**, ZAYA1-8B delegating to expert models ([docs/pm.md](docs/pm.md)).
- **Every Hugging Face model architecture**, mapped to a backend by a registry and kept current
  by a daily census ([docs/registry.md](docs/registry.md)).

Vulkan, ROCm, CUDA, ZINC, ComfyUI, Metal/MLX and Windows were cut from the engine's core
([#306](https://github.com/1bit-MONSTER/engine/pull/306)) and are not routes of the engine.
Lemonade serves them with its own backends and builds ([docs/lemonade.md](docs/lemonade.md)).

**Checked on the hardware.** Every HRX pin change and every PR labelled `hw-gate` passes the
hardware gate on Strix Halo before merge: it builds our HIP kernels into the server as the release
does, runs kernel tests, known answers, KLD against the release pin and long-prompt needles, and
checks that every graph of every gate model is on the GPU
([#478](https://github.com/1bit-MONSTER/engine/pull/478),
[#484](https://github.com/1bit-MONSTER/engine/pull/484), [docs/hw-gate.md](docs/hw-gate.md)). The
runtime keeps one memory budget for every buffer, and after a GPU fault the server refuses new work
and exits cleanly, so Lemonade can restart the model
([#486](https://github.com/1bit-MONSTER/engine/pull/486)). Serve numbers in these docs come from
`tools/bench.py`, which A/B-measures configurations in the same run ([docs/bench.md](docs/bench.md)).

The box's kernel is ours too: Linux `v7.3-rc4` plus the IOMMU PerfOpt backports, built from our
config, MOK-signed for Secure Boot, and gated like a pin bump. Moving it to the next upstream tag is
a written step list ([docs/kernel.md](docs/kernel.md)).

**Install:** through Lemonade. The packages are Lemonade with the `onebit` recipe and the engine
(unpack it and run `lemond-onebit`), and 1bit OS, a Lemonade appliance on a bootable USB image.
On Ubuntu 26.04 the stock `lemond` from the Lemonade PPA runs the engine through a systemd drop-in,
with no patch to Lemonade ([#483](https://github.com/1bit-MONSTER/engine/pull/483),
[docs/lemonade.md](docs/lemonade.md#ubuntu-2604-the-ppas-lemond)). The engine has no standalone
package of its own. The first release goes out when every item on the release checklist is done
([docs/releases.md](docs/releases.md)).

> **Direction:** HRX with our HIP kernels, plus the NPU engine, embedded into Lemonade rather than
> Lemonade embedded into the engine.[^geramyl] HRX met the decode gates in
> [RFC #213](https://github.com/1bit-MONSTER/engine/discussions/213), which removed the engine's
> Vulkan and ROCm builds. HRX is pinned to AMD's hrx-system and ggml-hrx with our commits on top,
> kept current by `bump-hrx` ([docs/hrx.md](docs/hrx.md#pinned-sources-kept-current)). ZAYA1-8B
> (Zyphra) runs from our llama.cpp on HRX at about 90 tok/s decode in Q4_K_M, matching
> transformers. Zyphra's Zamba, Zamba2 and BlackMamba ran only on the removed Vulkan build and are
> to be ported to HRX. The working engine is being ported from our private development repository;
> see [docs/PORTING.md](docs/PORTING.md). Measured results are on the
> [wiki](https://github.com/1bit-MONSTER/engine/wiki). This repository holds the verified code
> without the development history.

## License

The source code in this repository is Apache-2.0. See [LICENSE](LICENSE).

The release binaries (the Linux package and 1bit OS) also contain two proprietary components,
compiled in: the 1bit NPU runtime and the 1bit HIP kernels for HRX. They are not in this
repository, their source is not published, and they are not under Apache-2.0: they ship under the
[1bit Binary License](LICENSE-BINARY) (free to use for any purpose and to pass on unmodified;
third-party parts in [NOTICE-BINARY](NOTICE-BINARY)). A build from this
repository without them has no NPU engine and runs HRX on the public kernels only
([NOTICE](NOTICE), [docs/releases.md](docs/releases.md#private-components)).

## Thank you

**[Osmantic / ODS](https://github.com/Osmantic/ODS) comes first.** ODS introduced me to
vibecoding, and that is where all of this started. Without it, this engine would not exist.

**The Lemonade team and AMD's developers.** [Lemonade](https://github.com/lemonade-sdk/lemonade)
is where this engine runs. AMD's developers left breadcrumbs all over the place: the XDNA driver
and XRT, IRON and Peano, HRX, their tested llama.cpp integration, their issues, their examples.
This engine is what following those breadcrumbs built.

The repositories this engine is built on, in order of importance:

| # | Repository | What it gives this engine | License |
|---|---|---|---|
| 1 | [amd/xdna-driver](https://github.com/amd/xdna-driver) | The XDNA 2 NPU kernel driver (`amdxdna`) the NPU engine runs on | GPL-2.0 (driver) |
| 2 | [Xilinx/XRT](https://github.com/Xilinx/XRT) | AMD's NPU runtime and its API, where the engine's NPU work started | Apache-2.0 (userspace) |
| 3 | [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) | IRON and aiecc: how our own NPU kernels are written and compiled | Apache-2.0 WITH LLVM-exception |
| 4 | [Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie) | Peano, the C++ compiler for the NPU's AI Engine cores | Apache-2.0 WITH LLVM-exception |
| 5 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF inference on the Radeon iGPU through HRX (AMD's tested pair), and on the CPU | MIT |
| 6 | [ROCm/hrx-system](https://github.com/ROCm/hrx-system) (our fork [1bit-MONSTER/hrx-system](https://github.com/1bit-MONSTER/hrx-system) carries two commits) | HRX, AMD's HIP Runtime Extended, behind the HRX0 device | Apache-2.0 |
| 7 | [torvalds/linux](https://github.com/torvalds/linux) | The kernel, with `amdxdna` and `amdgpu` in-tree | GPL-2.0 WITH Linux-syscall-note |
| 8 | [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | Every model's `tokenizer.json`, byte-exact, behind our C ABI | Apache-2.0 |
| 9 | [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | The router that decides where each request runs | Apache-2.0 |
| 10 | [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM) | The Q4NX NPU model format and its models on Hugging Face (`FastFlowLM/*-NPU2`), which the engine's NPU route runs on its own kernels | MIT |
| 11 | [antirez/ds4](https://github.com/antirez/ds4) (DwarfStar) | DeepSeek V4 Flash, GLM 5.x and Qwen3.8-Flash-Next on its own kernels (ROCm on Strix Halo, CUDA, Metal) | MIT |

Also built on [nlohmann/json](https://github.com/nlohmann/json) (MIT)
and [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT).

Every third-party copyright and license is listed in [NOTICE](NOTICE). Each project keeps its
own license; nothing here relicenses anyone's work.

[^geramyl]: The HRX + Loom + NPU direction, and embedding the engine into Lemonade rather than Lemonade into the engine, were proposed by [geramyL](https://github.com/Geramy), a moderator on Lemonade's Discord. Thank you.
