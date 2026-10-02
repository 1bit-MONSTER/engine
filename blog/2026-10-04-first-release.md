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
tags: milestone, release, hrx, loom, kernels, ternary, mxfp4, lemonade, zyphra, qwen
summary: The 1bit engine's first release: every GGUF quant type decodes on HRX with Loom kernels, written by AI agents and checked against the CPU. Qwen3.8-27B reads prompts at 335 tok/s, ternary Bonsai-27B runs in 5.5 GiB, all inside Lemonade.

# 1bit engine v2026.40: the first release

The 1bit engine's first release is out. It is LLM inference for AMD Ryzen AI Max (Strix Halo),
running inside [Lemonade](https://github.com/lemonade-sdk/lemonade), with one direction: the GPU
runs on HRX with kernels we write and measure, the NPU runs the rest, and every number on this page
was measured on the machine it describes.

## What you get

One command serves a model behind an OpenAI-compatible API, and `--device auto` puts a GGUF on HRX:

```sh
tar --zstd -xf 1bit-v2026.40-linux-x86_64.tar.zst
1bit-v2026.40-linux-x86_64/1bit serve -m Qwen3.8-27B-UD-Q4_K_XL.gguf --port 8000
```

Or install Lemonade with the engine inside it (`lemonade-onebit-v2026.40`) and pick a model there:
Lemonade stays the server you talk to, and runs the engine as one of its backends. Four packages
ship: Linux, Lemonade with the engine, Windows, and 1bit OS, the engine as a bootable USB image
([releases](../docs/releases.md)).

## HRX: the GPU route

HRX is AMD's ggml-hrx runtime. Its kernels are written in Loom; we write ours in Loom too, or in HIP
where that measures faster, and each one is checked against the CPU before it ships. Strix Halo,
balanced power mode (85 W):

| | before | now |
|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL, 512-token prompt | 98 tok/s | **335 tok/s** |
| a 14,435-token prompt through `1bit serve` | 91 tok/s | **265 tok/s** |
| a 400-token prompt (not a multiple of 256) | 67 tok/s | **103 tok/s** |
| a 512-token prompt at 32K context | failed | **227 tok/s** |
| Qwen3.8-27B UD-IQ2_S decode | 3.1 tok/s | **8.3 tok/s** |
| ZAYA1-8B Q4_K_M decode | 47.9 tok/s | **about 90 tok/s** |

Qwen3.8-27B decodes at 97% of the Vulkan figure it replaced. Unsloth's sub-4-bit GGUFs (Q2_K,
IQ1, IQ2, IQ3) run on the GPU instead of the CPU, and `--mtp` speculative decoding on Qwen3.5/3.8 is
NaN-free ([HRX](../docs/hrx.md)).

## Every quant type, on HRX and Loom alone

As of this release every GGUF quant type we test decodes on the Strix Halo iGPU through HRX and Loom
kernels alone: no Vulkan, no ROCm, no CPU fallback at decode. The kernels that got it there are our own,
written by AI agents (Claude, working with the engine's owner): each one profiled, written in Loom,
checked against the CPU (KLD or bit for bit), measured on this machine and reviewed before it merged.
Prompt / decode, tok/s:

| kernel | measured on | before | now |
|---|---|---|---|
| MXFP4 experts, ADD_ID, clamped SwiGLU, attention sinks | gpt-oss-20b MXFP4 | 25.8 / 12.6 | **1011 / 39** |
| Hadamard rotation | Ternary-Bonsai-2-27B | 13.5 / 15.8 | **90.9 / 19.0** |
| TQ1_0 / TQ2_0 ternary | Ternary-Bonsai-1.7B TQ2_0 | CPU only | **4100 / 156** |
| IQ2_XS / IQ2_XXS | Qwen3-4B IQ2_XXS | 59 / 23.0 | **715 / 39.5** |
| IQ3_XXS / IQ2_S | Qwen3-4B, mixed IQ3_XXS | 2.8 / 1.5 | **250 / 8.9** |
| Q2_K | Qwen3-4B Q2_K | 686 / 47 | **1300 / 75** |
| separate gate/up codebooks | Qwen3.8-27B UD-IQ2_S | - / 3.1 | **- / 8.3** |

gpt-oss-20b runs entirely on HRX, about 1% from the CPU's perplexity (mean KLD 0.027): close, not bit
for bit, and we are working on the difference ([HRX](../docs/hrx.md)).

## Ternary Bonsai in 5.5 GiB

PrismML's Ternary-Bonsai-2-27B is Qwen3.8-27B trained to weights of -1, 0 and +1. Its PTQ1_0 and
PQ2_0 formats are now native types in our HRX build, decoded bit for bit as PrismML's own code
decodes them, so the file you download is the file that runs:

| Ternary-Bonsai-2-27B | weights in GPU memory | decode |
|---|---|---|
| converted to Q4_0 (before) | 14.13 GiB | 15.3 tok/s |
| PQ2_0, PrismML's file | 6.70 GiB | 15.8 tok/s |
| PTQ1_0, PrismML's file | **5.53 GiB** | 14.3 tok/s |

All three give the same logits (KLD 0.000129 against the CPU). A Hadamard kernel on HRX takes
Bonsai's 1024-point rotation off the CPU: its prompts went from 13 to 92 tok/s.

## Laya picks the device

`1bit serve --laya` classifies each conversation (code, prose, short, long document; 95.5% on 200
labelled requests) and a measured policy picks where it runs. The classifier itself runs on HRX at
15-16 ms a decision, and long documents now go to HRX ([Laya](../docs/laya.md)).

## Next

Still on the Vulkan build, and moving to HRX next: MoE experts streamed from the drive
(`--moe-slots`), images (`--mmproj`), several sequences on gated delta-net models,
Qwen3.8-Flash-Next, and Zyphra's Zamba family. Releases ship every Sunday from here on.

## Thank you

To the Lemonade and AMD developers whose work the engine runs on, and to PrismML for the Bonsai models
and their reference code.[^geramyl] Questions and results are welcome on
[Discord](https://discord.gg/fa5m4Vawpa).

[^geramyl]: The HRX + Loom + NPU direction, and embedding the engine into Lemonade rather than Lemonade into the engine, were proposed by [geramyL](https://github.com/Geramy), a moderator on Lemonade's Discord. Thank you.
