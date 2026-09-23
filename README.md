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
> HRX + Vulkan in one build ([docs/hrx.md](docs/hrx.md)), and the NPU fast lane on full ELFs
> ([docs/npu.md](docs/npu.md)). Also: MLX on Apple Silicon ([docs/apple.md](docs/apple.md)), and
> ZINC with NVIDIA through its CUDA backend ([docs/zinc.md](docs/zinc.md)). The working engine is
> being ported from [1bit-MONSTER](https://github.com/1bit-MONSTER/1bit-MONSTER); see
> [docs/PORTING.md](docs/PORTING.md). Measured results are on the
> [wiki](https://github.com/1bit-MONSTER/engine/wiki). This repository holds the verified code
> without the development history.

## License

Apache-2.0. See [LICENSE](LICENSE).
