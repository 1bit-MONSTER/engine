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
# Vulkan from upstream llama.cpp

`1bit serve --device vulkan` runs a `llama-server` built from upstream
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), pinned to its latest
release in `third_party/llama.cpp-vulkan`. It is a separate tree from
`third_party/llama.cpp`, which stays on the llama.cpp + hrx-system pair AMD
tests together for HRX ([hrx.md](hrx.md)).

Two pins, because the two routes stay current in different ways:

| Route | Source | Pinned to | Moved by |
|---|---|---|---|
| `--device vulkan` | `third_party/llama.cpp-vulkan` | ggml-org's latest release | `bump-llama-vulkan.yml`, daily |
| `--device hrx` | `third_party/llama.cpp` + `third_party/hrx-system` | AMD's tested pair (ROCm/ggml-staging-automation) | `bump-hrx.yml`, daily |

A new model architecture reaches the Vulkan route the day upstream releases it,
without waiting for AMD's pair to move. Qwen3.8-Flash-Next (`qwen4exp`) is the
case that prompted this: upstream added it on 2026-08-27, and AMD's pinned
llama.cpp (`f1a0aca141de`) cannot load it.

## Build

Needs the Vulkan headers, loader and `glslc`.

```
git submodule update --init --depth 1 third_party/llama.cpp-vulkan
cmake -B build -G Ninja -DONEBIT_VULKAN=ON
cmake --build build
```

`ONEBIT_VULKAN` and `ONEBIT_HRX` build side by side: `--device vulkan` then uses
the upstream build and `--device hrx` the HRX build. Without `ONEBIT_VULKAN`,
`--device vulkan` falls back to the HRX build's Vulkan backend, as before.
`ONEBIT_LLAMA_SERVER` still overrides both.

## Verified (Strix Halo, v0.5.0 `7fe450e19305`)

| Model | Result |
|---|---|
| `tests/serve_e2e.sh` (Qwen3-0.6B Q4_K_M) | PASS (`serve_e2e_vulkan_upstream`) |
| Qwen3.8-27B UD-Q4_K_XL | "The capital of France is Paris.", 12.2 tok/s |
| Qwen3.8-Flash-Next UD-Q4_K_XL (`--ctx-size 8192`) | "The capital of France is Paris.", 22.0 tok/s; the HRX pin fails to load it |

Large models need `--ctx-size`: without it llama-server allocates the KV cache
for the model's full trained context (262,144 tokens for Qwen3.8), which does not
fit next to Flash-Next's 104 GiB of weights.

Measured quant sweet spots for Qwen3.8 on this route are on the
[wiki](https://github.com/1bit-MONSTER/engine/wiki).
