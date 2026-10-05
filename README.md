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

**Site:** [github.com/1bit-MONSTER/site](https://github.com/1bit-MONSTER/site) (builds 1bit.gg from this repository's README.md and docs/) · **Kernel build:** [github.com/1bit-MONSTER/kernel](https://github.com/1bit-MONSTER/kernel)

**The 1bit engine runs inside [Lemonade](https://github.com/lemonade-sdk/lemonade).** Lemonade stays
the server you talk to (its catalog, downloads, router and UI), and for the models it hands to 1bit,
Lemonade runs the engine as one of its backends, the same way it runs `llama-server`. The engine
serves each model behind an OpenAI-compatible API (`1bit serve`), whatever device runs it:

- the XDNA 2 NPU engine
- HRX on the Radeon iGPU (AMD's ggml-hrx with our Loom kernels), the engine's one GPU route ([docs/hrx.md](docs/hrx.md)), and the CPU from the same llama.cpp build
- MoE experts streamed from the drive (`1bit serve --moe-slots N`), for MoE models larger than memory: kept, but not in this build until it moves to HRX with Qwen3.8-Flash-Next ([docs/moe-streaming.md](docs/moe-streaming.md))
- DwarfStar, for DeepSeek V4 Flash, GLM 5.x and Qwen3.8-Flash-Next in its own GGUFs, with SSD expert streaming
- MLX on Apple Silicon, through lemon-mlx-engine
- ONNX Runtime GenAI models (Lemonade's ONNX format) on the CPU ([docs/onnx.md](docs/onnx.md))
- Laya, which decides where each request runs
- every Hugging Face model architecture, kept current by a daily census

Every tuned setting `1bit serve` gives a backend is a recipe with its measurement attached
([docs/recipes.md](docs/recipes.md)), and the serve numbers in these docs come from
`tools/bench.py`, which A/B-measures configurations against a baseline in the same run
([docs/bench.md](docs/bench.md)).

Packages ship every Sunday, rebuilt at that week's upstream pins: Linux, Lemonade with the engine,
Windows and 1bit OS ([docs/releases.md](docs/releases.md)). The first release ships on Sunday,
4 October 2026.

> **Direction:** the engine is HRX (AMD's ggml-hrx, kernels in Loom or HIP) plus the NPU.[^geramyl] HRX met the decode gates in
> [RFC #213](https://github.com/1bit-MONSTER/engine/discussions/213), and its stage 3 removed the
> engine's Vulkan and ROCm llama.cpp builds, with `--lean`, `--adaptive`, `--long-model` and
> `--prefill-device`. What HRX does not run, Lemonade serves with its own llamacpp backends.
>
> **Status:** the engine runs inside Lemonade through `1bit serve` ([docs/lemonade.md](docs/lemonade.md),
> [docs/serve.md](docs/serve.md)); the NPU and HRX each pass its end-to-end test on
> Strix Halo, and the Lemonade recipe that runs it (`onebit`, in our fork
> [1bit-MONSTER/lemonade](https://github.com/1bit-MONSTER/lemonade)) passes Lemonade's own LLM test
> suite on HRX. Following a review,[^geramyl] the engine no longer
> vendors Lemonade: Lemonade is the host, 1bit is the engine inside it. Ported so far: HRX on AMD's live ggml-hrx
> ([docs/hrx.md](docs/hrx.md); its decode-split race, #123/#140, is fixed and the kernel is on by default; Q2_K, IQ2 and IQ3_XXS GGUFs run on HRX instead of the CPU, and `--mtp` on Qwen3.8-27B runs NaN-free since #257), the NPU engine on full ELFs with the
> upstream XDNA stack pinned ([docs/npu.md](docs/npu.md); its layer kernel and lm-head now decode from
> source, ELF to ELF, with no xclbin at any point, verified on Qwen3-0.6B/1.7B/4B against an fp64
> reference — the kernel sources stay private and are not yet merged here, docs/npu.md Step 3d) and
> MLX ([docs/apple.md](docs/apple.md)). ZAYA1-8B (Zyphra)
> runs from our llama.cpp on HRX, matching transformers, at about 90 tok/s decode in Q4_K_M
> ([docs/hrx.md](docs/hrx.md)). The rest of Zyphra's family (Zamba, Zamba2, BlackMamba) and
> Qwen3.8-Flash-Next ran only on the removed Vulkan build; they are to be ported to HRX. Experimental,
> and closed source: Qwen3.6-35B-A3B on the NPU through a private add-on, parity against fp64 passes,
> 16.3-16.5 tok/s decode ([docs/npu.md](docs/npu.md#private-routes)). GGUFs of six architectures
> (Qwen2.5, Qwen3 MoE, Qwen3.6-35B-A3B, MiniCPM4/5, GLM-4.7-Flash) also answer on the NPU through
> `1bit serve --device npu`: correct, not yet fast (0.006-0.17 tok/s;
> [docs/npu.md](docs/npu.md#from-a-gguf)). DwarfStar reads 1BP packages from our fork
> ([docs/dwarfstar.md](docs/dwarfstar.md)). Step 4, the Laya router, has landed as an opt-in:
> `1bit serve --laya` classifies each conversation (code, prose, short, long document; 95.5% on
> 200 labelled requests) and a measured policy picks the device; the scorer runs on HRX at
> 15-16 ms a decision ([docs/laya.md](docs/laya.md)). Step 5, the model registry, has landed: of 334,413
> HF text-generation models with an architecture, 94.71% are mapped to a backend and 64.2%
> have an architecture checked end to end on Strix Halo; a daily census keeps the counts
> current ([docs/registry.md](docs/registry.md)). The working engine is being ported from 1bit-MONSTER,
> our private development repository; see [docs/PORTING.md](docs/PORTING.md).
> Measured results are on the [wiki](https://github.com/1bit-MONSTER/engine/wiki). This repository
> holds the verified code without the development history.

## License

Apache-2.0. See [LICENSE](LICENSE).

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
| 1 | [amd/xdna-driver](https://github.com/amd/xdna-driver) | The XDNA 2 NPU driver and its XRT shim | Apache-2.0 (shim) |
| 2 | [Xilinx/XRT](https://github.com/Xilinx/XRT) | The runtime every NPU kernel runs through: full ELFs, hardware contexts, buffers | Apache-2.0 (userspace) |
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
