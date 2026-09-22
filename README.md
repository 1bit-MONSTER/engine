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
- Laya, which decides where each request runs

> **Status:** porting the working engine from
> [1bit-MONSTER](https://github.com/1bit-MONSTER/1bit-MONSTER) in four steps; see
> [docs/PORTING.md](docs/PORTING.md). This repository holds the verified code
> without the development history.

## License

Apache-2.0. See [LICENSE](LICENSE).
