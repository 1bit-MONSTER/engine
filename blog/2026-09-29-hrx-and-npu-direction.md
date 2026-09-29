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
tags: milestone, hrx, npu, laya, rocm
summary: The engine is heading to HRX plus the NPU. Laya's scorer already runs on HRX at 15-16 ms a decision, W4A4 reaches MoE experts, six GGUF architectures answer on the NPU, and DwarfStar reads 1BP packages.

# Where the engine is going: HRX and the NPU

Reviewing the Laya work, geramyL, a moderator on Lemonade's Discord, put it plainly: we want it on the NPU, a Vulkan version
already exists, and the stack is HRX + Loom + NPU. They were right, and not only for Laya. The
engine is heading to AMD's HRX backend, whose kernels we write in Loom, plus the NPU. Vulkan
stays the default until HRX meets the gates in
[RFC #213](https://github.com/1bit-MONSTER/engine/discussions/213): decode within 10% of Vulkan
on the showcase models, every architecture that runs on Vulkan checked on HRX, and speculative
decoding, several sequences per batch and vision on HRX. Here is what landed since the last
posts, and how far HRX already is.

## Laya's scorer runs on HRX

Laya classifies each conversation as code, prose, short or a long document, and a measured
policy picks the device ([previous post](2026-09-28-laya-routes-by-request-class.md)). The
classifier is a ModernBERT encoder. It now runs through ggmlc (our fork) on HRX, with Loom kernels
for what the encoder needs: norms, rotate-half RoPE, GEGLU and the attention scores
([#206](https://github.com/1bit-MONSTER/engine/pull/206),
[#208](https://github.com/1bit-MONSTER/engine/pull/208)).

| Scorer | Accuracy | Per decision |
|---|---|---|
| C++ on the CPU | 95.5% | 538 ms |
| ggmlc on Vulkan | 95.5% | 24 ms |
| **ggmlc on HRX** | **95.5%** | **15-16 ms** (25 ms at 128 tokens) |

Through `1bit serve --laya` that is about 28 ms per request at the median. Routing no longer costs
a noticeable first-turn delay.

## 4-bit activations reach the MoE experts

The Hadamard route rotates a model's weights so 4-bit activations lose less. It now covers the
expert matmuls too ([#199](https://github.com/1bit-MONSTER/engine/pull/199),
[#201](https://github.com/1bit-MONSTER/engine/pull/201)). Qwen3-Coder-30B-A3B, ROCm, rotated Q4_0:

| Route | Prompt (pp512) | Decode | KLD |
|---|---|---|---|
| exact 8-bit activations | 1,218 | 79.0 | 0.048 |
| **4-bit activations, attention and experts** | **2,156** | 79.2 | 0.081 |

Prompt speed only matters when prompts are long, so `tools/e2e_bench.py` measures what a user
waits for: time to first token plus decode, as effective tokens per second
([#202](https://github.com/1bit-MONSTER/engine/pull/202)). `1bit serve --long-model` sends
conversations with 2K or more prompt tokens to the rotated file and shorter ones to the default
device. On Qwen3-Coder-30B-A3B, 256 tokens out:

| Prompt tokens | Vulkan alone | `--long-model` |
|---|---|---|
| 512 | 70.7 | 71.3 |
| 2,048 | 52.1 | 53.1 |
| 8,192 | 20.3 | 25.5 |
| 16,384 | 8.6 | **13.1** |

Running rotated files in 1,024-token micro-batches added another 6-10% to prompt speed for free
([#205](https://github.com/1bit-MONSTER/engine/pull/205)).

## GGUFs on the NPU: correct first

`1bit serve -m <gguf> --device npu` repacks a GGUF for the NPU and runs it on the per-op forward
([#209](https://github.com/1bit-MONSTER/engine/pull/209)). Six architectures answer " Paris":
MiniCPM5-1B, Qwen2.5-7B, Qwen3-Coder-30B-A3B, Qwen3.6-35B-A3B (GDN + MoE), MiniCPM4-8B (LongRoPE
and its embedding and residual scales) and GLM-4.7-Flash (MLA). Getting there meant
matching llama.cpp's conventions exactly: rotary pairing, MiniCPM's scales, and the layout of
GLM's MLA weights. It is not fast yet: 0.006-0.17 tok/s, because the host sequences every
operation. The next step is the fused path, which the NPU work already has for Qwen3.6-35B-A3B
(16.3-16.5 tok/s decode through the private add-on).

## DwarfStar reads 1BP packages

DwarfStar now comes from our fork, which reads 1BP packages: a GGUF's tensors and metadata,
aligned for streaming ([#196](https://github.com/1bit-MONSTER/engine/pull/196)).
`gguf_to_1bp.py` converts a local GGUF or one on Hugging Face through range requests, and
`1bit serve` recognises a `.1bp` by its first bytes. DeepSeek V4 Flash Q2, converted straight from
Hugging Face, gives the same answers and token counts as its GGUF.

## HRX so far

The gates in RFC #213 are measured, not assumed. On ZAYA1-8B, Vulkan decodes at 93 tok/s and
HRX at 47.9. On long prompts, HRX already prefills faster than Vulkan: handing an 8K prompt's
prefill to HRX cuts a Qwen2.5-7B request by 26%. The decode-split race that made HRX unreliable
is fixed ([#123](https://github.com/1bit-MONSTER/engine/issues/123)), and its multi-pass output
step got 15-22% faster on long contexts ([#180](https://github.com/1bit-MONSTER/engine/pull/180)).
Each gate that closes gets its own post.
