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
tags: milestone, release, hrx, ternary, lemonade
summary: The first real release of the 1bit engine: HRX is the GPU route, the engine runs inside Lemonade, ternary Bonsai runs from PrismML's own files, and every number below was measured on Strix Halo.

# The first release

Sunday's weekly job published the engine's first real release: a Linux package, Lemonade with the
engine inside it, a Windows zip and the 1bit OS image, all built at this week's pins
([docs/releases.md](../docs/releases.md)).

Two decisions shaped it.
- **HRX is the GPU route.** [geramyL](https://github.com/Geramy), a moderator on Lemonade's Discord,
  proposed it: HRX (AMD's ggml-hrx) for the GPU, with kernels in Loom (or HIP where that measures
  faster), plus the NPU. `--device auto` means HRX.
- **Lemonade is the host.** The engine runs inside Lemonade as one of its backends. What the engine
  does not run on HRX or the NPU, Lemonade's own llama.cpp backends serve.

## HRX

All figures below: Strix Halo in balanced power mode (85 W), `llama-bench` or `1bit serve` as noted
([docs/hrx.md](../docs/hrx.md)).

| | |
|---|---|
| Qwen3.8-27B UD-Q4_K_XL decode | 97% of the previous Vulkan figure |
| Qwen3.8-27B prompt, pp512 | 98 -> 335 tok/s (q8_1 x4 prompt kernel, fork #55) |
| a 14,435-token prompt through `1bit serve` | 91 -> 265 tok/s |
| prompts that are not a multiple of 256, pp400 | 67 -> 103 tok/s (fork #61) |
| pp512 at 32K context | failed -> 227 tok/s (fork #59) |
| UD-IQ2_S decode (mixed-format FFN pairs) | 3.1 -> 8.3 tok/s (fork #60) |
| ZAYA1-8B Q4_K_M decode | about 90 tok/s |

Sub-4-bit GGUFs (Q2_K, IQ1, IQ2, IQ3) run on the GPU instead of the CPU, and `--mtp` on Qwen3.5/3.8 is
NaN-free since the delta-net snapshot fix (fork #52).

## Ternary Bonsai from PrismML's own files

PrismML's Ternary-Bonsai-2-27B is Qwen3.8-27B trained to weights of -1, 0 and +1. Until this week it
ran on HRX only after conversion to Q4_0 (14.1 GiB). Now PTQ1_0 and PQ2_0 are native types in our HRX
build (fork #62), decoded bit for bit as PrismML's own code decodes them:

| Ternary-Bonsai-2-27B | weights | decode |
|---|---|---|
| Q4_0 copy + packed ternary decode | 14.13 GiB | 15.3 tok/s |
| PQ2_0 | 6.70 GiB | 15.8 tok/s |
| PTQ1_0 | 5.53 GiB | 14.3 tok/s |

All three give the same logits (KLD 0.000129 against the CPU).

## What is still on Vulkan

These keep the Vulkan build for now and move to HRX next:
- `--moe-slots`, which streams MoE experts from the drive;
- images (`--mmproj`);
- `--parallel` on gated delta-net models;
- the RAG servers;
- Qwen3.8-Flash-Next, and Zyphra's Zamba, Zamba2 and BlackMamba.

## Thank you

To geramyL for the direction, to the Lemonade and AMD developers whose work the engine runs on, and to
PrismML for the Bonsai models and their reference code.
