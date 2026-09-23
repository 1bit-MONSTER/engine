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

One binary that runs [Lemonade](https://github.com/lemonade-sdk/lemonade) completely,
with AMD Ryzen AI hardware behind it:

- the XDNA 2 NPU engine
- HRX and Vulkan on the Radeon iGPU, compiled together in one llama.cpp build
- ZINC, which also reaches NVIDIA GPUs (CUDA) and Apple GPUs (Metal)
- Laya, which decides where each request runs
- every Hugging Face model architecture, kept current by a daily census

> **Status:** steps 1–3 of 5 have landed: embedded Lemonade ([docs/lemonade.md](docs/lemonade.md)),
> HRX + Vulkan in one build on AMD's live ggml-hrx ([docs/hrx.md](docs/hrx.md)), and the NPU engine on
> full ELFs with the upstream XDNA stack pinned ([docs/npu.md](docs/npu.md); its layer kernel is
> not yet built from source). Also: MLX on Apple Silicon through Lemonade's `mlx` recipe
> ([docs/apple.md](docs/apple.md)), and ZINC, which reaches NVIDIA GPUs through its CUDA backend
> ([docs/zinc.md](docs/zinc.md)). Next is step 4, the Laya router. The working engine is being ported from
> [1bit-MONSTER](https://github.com/1bit-MONSTER/1bit-MONSTER) in five steps; see
> [docs/PORTING.md](docs/PORTING.md). Measured results are on the
> [wiki](https://github.com/1bit-MONSTER/engine/wiki). This repository holds the verified code
> without the development history.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Thank you

**[Osmantic / ODS](https://github.com/Osmantic/ODS) comes first.** ODS introduced me to
vibecoding, and that is where all of this started. Without it, this engine would not exist.

**The Lemonade team and AMD's developers.** [Lemonade](https://github.com/lemonade-sdk/lemonade)
runs inside this binary. AMD's developers left breadcrumbs all over the place: the XDNA driver
and XRT, IRON and Peano, HRX, their tested llama.cpp integration, their issues, their examples.
This engine is what following those breadcrumbs built.

The repositories this engine is built on, in order of importance:

| # | Repository | What it gives this engine | License |
|---|---|---|---|
| 1 | [amd/xdna-driver](https://github.com/amd/xdna-driver) | The XDNA 2 NPU driver and its XRT shim | Apache-2.0 (shim) |
| 2 | [Xilinx/XRT](https://github.com/Xilinx/XRT) | The runtime every NPU kernel runs through: full ELFs, hardware contexts, buffers | Apache-2.0 (userspace) |
| 3 | [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) | IRON and aiecc: how our own NPU kernels are written and compiled | Apache-2.0 WITH LLVM-exception |
| 4 | [Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie) | Peano, the C++ compiler for the NPU's AI Engine cores | Apache-2.0 WITH LLVM-exception |
| 5 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF inference on the Radeon iGPU: Vulkan and HRX in one build | MIT |
| 6 | [ROCm/hrx-system](https://github.com/ROCm/hrx-system) | HRX, AMD's HIP Runtime Extended, behind the HRX0 device | Apache-2.0 |
| 7 | [torvalds/linux](https://github.com/torvalds/linux) | The kernel, with `amdxdna` and `amdgpu` in-tree | GPL-2.0 WITH Linux-syscall-note |
| 8 | [zolotukhin/zinc](https://github.com/zolotukhin/zinc) | Its own GPU kernels, and the engine's route to NVIDIA through CUDA | MIT |
| 9 | [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | Every model's `tokenizer.json`, byte-exact, behind our C ABI | Apache-2.0 |
| 10 | [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | The router that decides where each request runs | Apache-2.0 |

Also built on [nlohmann/json](https://github.com/nlohmann/json) (MIT),
[yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT), and the libraries Lemonade
builds with: curl, zstd, Mbed TLS, libwebsockets, CLI11 and Brotli.

Every third-party copyright and license is listed in [NOTICE](NOTICE). Each project keeps its
own license; nothing here relicenses anyone's work.
