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
