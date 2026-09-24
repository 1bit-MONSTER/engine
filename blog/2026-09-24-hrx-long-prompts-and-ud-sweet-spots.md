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
tags: hrx, quantization, unsloth

# HRX answers long prompts correctly, and where Unsloth quants pay

Two results from the end of the day: a bug fixed in HRX decoding, and one table that answers "which Unsloth file should I download?" for five popular models.

## HRX: decoding past 256 tokens

`1bit serve --device hrx` gave garbled answers to prompts of about 400 to 1,700 tokens, while short prompts and very long ones read fine. Prompt processing was right at every length; what went wrong was decoding one token at a time once the KV cache held more than 256 tokens.

HRX's decode attention splits the cache into 64-token blocks and then combines the per-block results. Up to 256 tokens it combines them one way; past that it switches to a cooperative path in which four groups of threads each take two query rows. That path only checked that a row's query head exists, not that the row belongs to the key/value head doing the work. Qwen3-0.6B has two query heads per key/value head, so the spare groups combined stale rows and wrote them over their neighbours' output. The fix adds that one bound, in both copies of the kernel.

After the fix, fed the same tokens as Vulkan, HRX's predictions differ by 1-3% at every length we tried (248 to 2,040 tokens), and chat answers match Vulkan's from 404 to 2,844 tokens. The details are in the [HRX docs](../docs/hrx.md).

The same day, the sub-4-bit Unsloth files started working on HRX: our pinned HRX build has an IQ3_XXS matmul, and HRX and Vulkan now give the same perplexity within 0.3% on UD-Q2_K_XL and UD-IQ2_M. They decode slowly there (13-17 tok/s, where Vulkan does about 390 on Qwen3-0.6B) because parts of those files still run on the CPU, so Vulkan remains the device for them.

## Which Unsloth file to download

For each model we measured every Unsloth Dynamic quant from 2 to 5 bits against a reference (BF16 when it fits, else Q8_0): KL divergence over 40 x 512 tokens of wikitext-2, and decode speed on Vulkan on Strix Halo. Two picks come out of it on every model:

| Model | Lean pick | KL | Decode | Accurate pick | KL | Decode |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | UD-Q4_K_XL, 2.7 GiB | 0.018 | 60.0 tok/s | UD-Q5_K_XL, 3.0 GiB | 0.0099 | 55.4 tok/s |
| Qwen3.5-9B | UD-Q4_K_XL, 5.6 GiB | 0.016 | 36.0 | UD-Q5_K_XL, 6.3 GiB | 0.0092 | 32.7 |
| Qwen3-Coder-30B-A3B | UD-Q4_K_XL, 16.5 GiB | 0.027 | 84.7 | UD-Q5_K_XL, 20.2 GiB | 0.012 | 76.4 |
| Qwen3.6-27B | UD-Q4_K_XL, 16.4 GiB | 0.018 | 12.0 | UD-Q5_K_XL, 18.7 GiB | 0.0081 | 10.0 |
| Qwen3.6-35B-A3B | UD-Q4_K_XL, 20.8 GiB | 0.014 | 61.0 | UD-Q5_K_XL, 24.8 GiB | 0.0091 | 53.7 |

- **UD-Q4_K_XL is the lean pick everywhere:** the smallest file that stays within a KL of 0.03.
- **UD-Q5_K_XL is the accurate pick:** under 0.01 on four of the five models, for 8-17% slower decode. Qwen3-Coder-30B-A3B is the exception: 0.012 is as close as any Unsloth file gets it.
- **Skip the 2-bit files at these sizes:** KL 0.11-0.23, with the top prediction changing on 13-21% of tokens.

One popular model is missing: gemma-4-12b-it does not run correctly on the llama.cpp build we measured with, so there was nothing fair to compare. (MiniMax-H3 was on our download list too, by mistake: it generates video, it is not a text model.) Every row, with prompt speeds and ROCm, is on the [Quantization page](https://github.com/1bit-MONSTER/engine/wiki/Quantization) of the wiki.
