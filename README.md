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

**The 1bit engine runs inside [Lemonade](https://github.com/lemonade-sdk/lemonade).** Lemonade stays
the server you talk to (its catalog, downloads, router and UI), and for the models it hands to 1bit,
Lemonade runs the engine as one of its backends, the same way it runs `llama-server`. The engine
serves each model behind an OpenAI-compatible API (`1bit serve`), whatever device runs it:

- the XDNA 2 NPU engine
- HRX on the Radeon iGPU (AMD's ggml-hrx with our Loom and HIP kernels), the default GPU route ([docs/hrx.md](docs/hrx.md))
- Vulkan and ROCm builds, leaving the engine in stages (RFC #213): `--device auto` means HRX, and Lemonade's own backends serve what the engine does not ([docs/vulkan.md](docs/vulkan.md))
- MoE experts streamed from the drive on Vulkan (`1bit serve --moe-slots N`), for MoE models larger than memory ([docs/moe-streaming.md](docs/moe-streaming.md#streaming-in-the-inference-path))
- a lean option, ROCmFPX's ROCmFP4 and ROCmI4 formats: faster, less accurate ([docs/lean.md](docs/lean.md))
- ZINC, which also reaches NVIDIA GPUs (CUDA) and Apple GPUs (Metal)
- DwarfStar, for DeepSeek V4 Flash, GLM 5.x and Qwen3.8-Flash-Next in its own GGUFs, with SSD expert streaming
- MLX on Apple Silicon, through lemon-mlx-engine
- ONNX Runtime GenAI models (Lemonade's ONNX format) on the CPU ([docs/onnx.md](docs/onnx.md))
- Laya, which decides where each request runs
- ComfyUI.cpp: ComfyUI workflows (Stable Diffusion 1.5 text-to-image and image-to-image) in C++, matching ComfyUI's output to 50 dB ([docs/comfyui.md](docs/comfyui.md))
- every Hugging Face model architecture, kept current by a daily census

Every tuned setting `1bit serve` gives a backend is a recipe with its measurement attached
([docs/recipes.md](docs/recipes.md)), and the serve numbers in these docs come from
`tools/bench.py`, which A/B-measures configurations against a baseline in the same run
([docs/bench.md](docs/bench.md)).

Packages ship every Sunday, rebuilt at that week's upstream pins: Linux, Lemonade with the engine,
Windows and 1bit OS ([docs/releases.md](docs/releases.md)). The first release ships on Sunday,
4 October 2026.

> **Direction:** the engine is HRX (AMD's ggml-hrx, kernels in Loom or HIP, whichever measures faster)
> plus the NPU[^geramyl]
> ([RFC #213](https://github.com/1bit-MONSTER/engine/discussions/213)). `--device auto` means HRX. What the
> engine does not run on HRX or the NPU is Lemonade's job: Lemonade ships its own llama.cpp backends
> (Vulkan, ROCm, CPU), and `1bit serve` hands those models back. The Vulkan and ROCm builds still in
> this repository leave in stages.
>
> **Status (first release, 4 October 2026):**
> - **Inside Lemonade.** `1bit serve` is a Lemonade backend ([docs/lemonade.md](docs/lemonade.md),
>   [docs/serve.md](docs/serve.md)); the `onebit` recipe in our fork
>   [1bit-MONSTER/lemonade](https://github.com/1bit-MONSTER/lemonade) passes Lemonade's LLM test suite on HRX.
> - **HRX on the Radeon iGPU** ([docs/hrx.md](docs/hrx.md)), on AMD's live ggml-hrx with our kernels:
>   Qwen3.8-27B UD-Q4_K_XL decodes at 97% of the old Vulkan figure and reads prompts at pp512 335 tok/s
>   (a 14K-token prompt at 265 tok/s); ZAYA1-8B (Zyphra) decodes at about 90 tok/s; GGUFs from Q2_K and
>   IQ1/IQ2/IQ3 up run on the GPU, not the CPU; prompts past 32K context work; `--mtp` on Qwen3.5/3.8 is
>   NaN-free.
> - **Ternary on HRX.** PrismML's Ternary Bonsai runs from its own PTQ1_0 / PQ2_0 files: the 27B in
>   5.5 GiB, 14-16 tok/s decode, logits identical to an exact Q4_0 copy.
> - **Hadamard-rotated Q4_0** files (`tools/hadamard_q4_0.py`) run on HRX.
> - **Laya** picks the device for each conversation (`1bit serve --laya`, 95.5% on 200 labelled
>   requests; the scorer runs on HRX at 15-16 ms a decision; [docs/laya.md](docs/laya.md)); long
>   documents go to HRX.
> - **The NPU engine** on full ELFs with the upstream XDNA stack pinned ([docs/npu.md](docs/npu.md)); its
>   layer kernel is not yet built from source. GGUFs of six architectures also answer on the NPU,
>   correct but not yet fast. Qwen3.6-35B-A3B on the NPU is a closed-source add-on (16.3-16.5 tok/s).
> - **The model registry** maps 94.88% of 332,726 HF text-generation models to a backend, with a daily
>   census ([docs/registry.md](docs/registry.md)).
> - **Still on Vulkan in this build**, until they are ported to HRX: `--moe-slots` (MoE experts streamed
>   from the drive, [docs/moe-streaming.md](docs/moe-streaming.md)), `--mmproj`, `--parallel` on gated
>   delta-net models, the RAG servers, Qwen3.8-Flash-Next, and Zyphra's Zamba, Zamba2 and BlackMamba.
>
> Measured results are on the [wiki](https://github.com/1bit-MONSTER/engine/wiki). The working engine
> is being ported from 1bit-MONSTER, our private development repository ([docs/PORTING.md](docs/PORTING.md));
> this repository holds the verified code without the development history.

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
| 5 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF inference on the Radeon iGPU: Vulkan (upstream release) and HRX (AMD's tested pair) | MIT |
| 6 | [ROCm/hrx-system](https://github.com/ROCm/hrx-system) | HRX, AMD's HIP Runtime Extended, behind the HRX0 device | Apache-2.0 |
| 7 | [torvalds/linux](https://github.com/torvalds/linux) | The kernel, with `amdxdna` and `amdgpu` in-tree | GPL-2.0 WITH Linux-syscall-note |
| 8 | [zolotukhin/zinc](https://github.com/zolotukhin/zinc) | Its own GPU kernels, and the engine's route to NVIDIA through CUDA | MIT |
| 9 | [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | Every model's `tokenizer.json`, byte-exact, behind our C ABI | Apache-2.0 |
| 10 | [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | The router that decides where each request runs | Apache-2.0 |
| 11 | [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM) | The Q4NX NPU model format and its models on Hugging Face (`FastFlowLM/*-NPU2`), which the engine's NPU route runs on its own kernels | MIT |
| 12 | [antirez/ds4](https://github.com/antirez/ds4) (DwarfStar) | DeepSeek V4 Flash, GLM 5.x and Qwen3.8-Flash-Next on its own kernels (ROCm on Strix Halo, CUDA, Metal) | MIT |

Also built on [nlohmann/json](https://github.com/nlohmann/json) (MIT)
and [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT).

Every third-party copyright and license is listed in [NOTICE](NOTICE). Each project keeps its
own license; nothing here relicenses anyone's work.

[^geramyl]: The HRX + Loom + NPU direction, and embedding the engine into Lemonade rather than Lemonade into the engine, were proposed by [geramyL](https://github.com/Geramy), a moderator on Lemonade's Discord. Thank you.
