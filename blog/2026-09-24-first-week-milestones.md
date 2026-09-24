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
tags: milestones, npu, hrx, vulkan, zinc

# The first week of 1bit engine

The clean repository started on 2026-09-22. In its first three days, the engine came up on every device it targets, inside Lemonade, with each step verified on real hardware before it merged. These are the milestones, with the numbers and where they come from.

## The NPU, on full ELFs generated in C++

The XDNA 2 NPU now runs without an xclbin and without per-context kernel files. The engine generates each kernel's full ELF (the design plus its control code) in C++ when the model loads, and XRT opens it directly; for each token, the host submits every layer and the lm head as one runlist ([#8](https://github.com/1bit-MONSTER/engine/pull/8), [#9](https://github.com/1bit-MONSTER/engine/pull/9)).

- Qwen3-0.6B decodes at **91 tok/s** (11.0 ms per token), and **87 tok/s** served through Lemonade.
- The XDNA driver and its XRT are pinned to upstream and built privately ([#12](https://github.com/1bit-MONSTER/engine/pull/12)).
- The layer kernel itself is being rebuilt from scratch as our own open kernel ("dx", on IRON and Peano). RMSNorm, RoPE and attention pass their tests so far, correctly rounded to bf16.

## HRX and Vulkan in one build, then both at once

The Radeon iGPU has two devices in one llama.cpp build: `HRX0` on AMD's HRX runtime and `Vulkan0`. HRX follows AMD's live ggml-hrx, pinned to the pair AMD tests and bumped daily ([#11](https://github.com/1bit-MONSTER/engine/pull/11)); Vulkan has its own pin on upstream llama.cpp's latest release ([#28](https://github.com/1bit-MONSTER/engine/pull/28)).

- **HRX went from 3-5x slower than Vulkan to close behind it.** Its prefill is now 1.3-1.7x Vulkan's, and its decode within 10% on 4-bit files.
- **Our patches make HRX correct on more models** ([#29](https://github.com/1bit-MONSTER/engine/pull/29)): an IQ3_XXS kernel, and op claims that only take the nodes HRX can actually run. `test-backend-ops` on HRX0 went from about 1150 failures to **791 passing, 0 failing**, and Unsloth's sub-4-bit files, which failed on HRX, now come within 1% of Vulkan's perplexity.
- **Prefill on HRX, decode on Vulkan, with zero copies** ([#30](https://github.com/1bit-MONSTER/engine/pull/30)): a whole 8192-token request on Qwen2.5-7B finishes **26% sooner**. [How it works](2026-09-24-hrx-prefill-vulkan-decode.md).

## Beyond AMD

- **NVIDIA, through ZINC's CUDA backend** ([#14](https://github.com/1bit-MONSTER/engine/pull/14)): Qwen3.5-9B at **167-173 tok/s** on an RTX 5090, and 295 tok/s through ZINC's Vulkan backend on Strix Halo.
- **Apple Silicon**, through Lemonade's `mlx` recipe and lemon-mlx-engine, verified on an M4 ([#10](https://github.com/1bit-MONSTER/engine/pull/10)).

## Underneath

- **Tokenizers:** Hugging Face `tokenizers` v0.23.2 behind our C ABI reads any model's `tokenizer.json`, byte-exact on 18 models ([#19](https://github.com/1bit-MONSTER/engine/pull/19)).
- **Quantization:** on the GPU, Unsloth's Dynamic UD-Q4_K_XL beats Q4_K_M: 17% lower KL divergence for 2% more size. On the NPU, a ternary model converts to 4-bit exactly, and XDNA 2 multiplies int8 by int4 natively ([wiki](https://github.com/1bit-MONSTER/engine/wiki/Quantization)).
- **The Linux kernel** is pinned to upstream v7.3-rc4 with `amdxdna` in-tree ([#17](https://github.com/1bit-MONSTER/engine/pull/17)), and the **Laya router**'s source and checkpoints are pinned and hash-verified ([#18](https://github.com/1bit-MONSTER/engine/pull/18)).

## Next

Step 4 is the Laya router, which decides where each request runs. After it: every Hugging Face architecture, kept current by a daily census. The [porting map](../docs/PORTING.md) tracks it, and the [wiki](https://github.com/1bit-MONSTER/engine/wiki) keeps every measured number with how it was taken.
