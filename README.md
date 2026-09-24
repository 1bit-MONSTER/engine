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

**Documentation:** [1bit.gg](https://1bit.gg/) · **Measured results:** [wiki](https://github.com/1bit-MONSTER/engine/wiki)

**The 1bit engine runs inside [Lemonade](https://github.com/lemonade-sdk/lemonade).** Lemonade stays
the server you talk to (its catalog, downloads, router and UI), and for the models it hands to 1bit,
Lemonade runs the engine as one of its backends, the same way it runs `llama-server`. The engine
serves each model behind an OpenAI-compatible API (`1bit serve`), whatever device runs it:

- the XDNA 2 NPU engine
- HRX on the Radeon iGPU, on the llama.cpp + hrx-system pair AMD tests
- Vulkan on the Radeon iGPU, from upstream llama.cpp's latest release, so new architectures land the day upstream ships them
- a lean option, ROCmFPX's ROCmFP4 and ROCmI4 formats: faster, less accurate ([docs/lean.md](docs/lean.md))
- ZINC, which also reaches NVIDIA GPUs (CUDA) and Apple GPUs (Metal)
- MLX on Apple Silicon, through lemon-mlx-engine
- Laya, which decides where each request runs
- every Hugging Face model architecture, kept current by a daily census

> **Status:** the engine runs inside Lemonade through `1bit serve` ([docs/lemonade.md](docs/lemonade.md),
> [docs/serve.md](docs/serve.md)); the NPU, Vulkan, HRX and ZINC each pass its end-to-end test on
> Strix Halo, and the Lemonade recipe that runs it (`onebit`, in our fork
> [1bit-MONSTER/lemonade](https://github.com/1bit-MONSTER/lemonade)) passes Lemonade's own LLM test
> suite on Vulkan and HRX. Following geramyL's review the engine no longer
> vendors Lemonade: Lemonade is the host, 1bit is the engine inside it. Ported so far: HRX on AMD's live ggml-hrx
> ([docs/hrx.md](docs/hrx.md)), Vulkan from upstream llama.cpp's latest release ([docs/vulkan.md](docs/vulkan.md)), the NPU engine on full ELFs with the
> upstream XDNA stack pinned ([docs/npu.md](docs/npu.md); its layer kernel is not yet built from
> source), ZINC ([docs/zinc.md](docs/zinc.md)) and MLX ([docs/apple.md](docs/apple.md)). Next is step 4,
> the Laya router. The working engine is being ported from
> [1bit-MONSTER](https://github.com/1bit-MONSTER/1bit-MONSTER); see [docs/PORTING.md](docs/PORTING.md).
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
| 5 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF inference on the Radeon iGPU: Vulkan (upstream release) and HRX (AMD's tested pair) | MIT |
| 6 | [ROCm/hrx-system](https://github.com/ROCm/hrx-system) | HRX, AMD's HIP Runtime Extended, behind the HRX0 device | Apache-2.0 |
| 7 | [torvalds/linux](https://github.com/torvalds/linux) | The kernel, with `amdxdna` and `amdgpu` in-tree | GPL-2.0 WITH Linux-syscall-note |
| 8 | [zolotukhin/zinc](https://github.com/zolotukhin/zinc) | Its own GPU kernels, and the engine's route to NVIDIA through CUDA | MIT |
| 9 | [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | Every model's `tokenizer.json`, byte-exact, behind our C ABI | Apache-2.0 |
| 10 | [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | The router that decides where each request runs | Apache-2.0 |

Also built on [nlohmann/json](https://github.com/nlohmann/json) (MIT)
and [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT).

Every third-party copyright and license is listed in [NOTICE](NOTICE). Each project keeps its
own license; nothing here relicenses anyone's work.
