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
tags: qwen, speculative-decoding, quantization, rocm, vulkan
summary: The DFlash2 drafter decodes Qwen3.8-27B at 45.7 tok/s on code through 1bit serve --dflash, 44% faster than MTP.

# Qwen3.8-27B at 45.7 tok/s: DFlash2 in `1bit serve`, and 4-bit activations with a Hadamard rotation

Qwen3.8-27B is a dense 27B model, and on Strix Halo its decode speed is set by memory
bandwidth: every token reads the whole 15 GB file, so plain decode sits near 11-12 tok/s.
Prompt processing is set by the GPU's matrix units instead. This week we pushed on both.
Every number below was measured on Strix Halo (Ryzen AI Max+ 395, Radeon 8060S), with the
method in the linked docs.

## Decode: DFlash2 drafts a whole block

Speculative decoding breaks the bandwidth limit by guessing several tokens and checking them
all in one pass of the big model. `1bit serve --mtp` already does this with the model's own
MTP head. [DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), from z-lab and Inco,
is a separate 2B draft model trained for Qwen3.8-27B. It emits a whole block of 8 tokens in
one pass, reading the big model's hidden states, and keeps the top 16 candidates at each
position so it can pick a coherent path through them. Upstream llama.cpp added it as
`draft-dflash`, and the Vulkan build the engine pins already carries it.

The new `1bit serve --dflash <draft.gguf>` wires it in. One detail mattered more than any
other: llama-server's default draft length is 3, and DFlash only pays when it drafts full
blocks. At length 3 it matched MTP; at full blocks it pulled well ahead. So `--dflash` now
drafts full blocks unless you say otherwise.

Qwen3.8-27B Q4_0 through `1bit serve --device vulkan`, decode tok/s on three chat prompts
(code / prose / short), back to back:

| Drafter | tok/s |
|---|---|
| none | 11.1 / 11.8 / 12.4 |
| `--mtp` | 31.7 / 25.8 / **28.2** |
| `--dflash` (DFlash2, Q8_0) | **45.7 / 28.3** / 17.8 |

That is 4.1x on code, and 44% over MTP. On prose DFlash2 edges ahead. On the short prompt,
a translation that ends after about 16 tokens, MTP still wins: there are too few tokens to
fill blocks. The drafter runs faster in Q8_0 than in BF16 (42.5 against 38.9 tok/s on code).
A drafted token is kept only when the model itself agrees, so the output is the model's own.
Converting the drafter needs only the target's config and tokenizer, not its weights
([docs/serve.md](../docs/serve.md#dflash-draft-models-dflash)).

## Prompt processing: 4-bit activations, rotated first

On the exact int8 path, prompt processing for this model tops out near 400 tok/s. The GPU
is not power-limited: it runs at about 158 W and 2.7 GHz through the whole prompt, and the
matmul kernel reaches about half of the int8 matrix rate. The matrix units run int4 × int4 at
twice the int8 rate, though, and ROCmFPX's W4A4 kernel uses exactly that instruction.

We pointed that kernel at Unsloth's plain Q4_0 file. A Q4_0 weight is already a 4-bit value
with an offset of 8, and flipping its top bit gives the same value in two's complement. So
the weights stay exactly what Unsloth shipped, and only the activations are rounded to 4 bits.
That is fast (461-488 tok/s) but lossy: mean KLD against the full model rose from 0.029 to
0.084. Leaving any single group of tensors at int8 did not fix it, because the error comes from
all of them.

A Hadamard rotation fixes much of it. A few large activation values in a block of 32 force a
coarse 4-bit scale on the other 31. Rotating each block by a 32-point Walsh-Hadamard transform
spreads that energy evenly before rounding. We rotate the weights the same way once, offline,
so the product is unchanged: on the exact int8 path the rotated file matches the plain one
(KLD 0.031 against 0.029). The rotation itself runs inside the activation quantizer, as three
lane shuffles and two in-register butterfly steps.

| Qwen3.8-27B Q4_0 on ROCm | pp512 | Mean KLD |
|---|---|---|
| exact int8 | ~400 | 0.029 |
| W4A4 | 461-488 | 0.084 |
| **W4A4, Hadamard-rotated** | **503** | **0.055** |

The rotation removes a third of the 4-bit error. Perplexity ends up 1.2% above the exact
path, and the error is still nearly double. So this is measured, not shipped: the patches are
not in the engine's pin yet ([docs/lean.md](https://github.com/1bit-MONSTER/engine/blob/0baf286/docs/lean.md)).

## What's next

DFlash2 is in `1bit serve` today. For prompts, the next steps are a better exact int8 matmul
for this GPU (about 15% would reach the W4A4 numbers without the error), and rotations
larger than 32, which spread outliers further. Packages ship on Sunday with the weekly
release.

**Update, later on 2026-09-28:** the Hadamard route has shipped. `tools/hadamard_q4_0.py` makes the
rotated file, `1bit serve` recognises it and runs it on the lean ROCm build by itself, and the
Qwen3.8-27B file is published as
[1bit-MONSTER/Qwen3.8-27B-Q4_0-H32-GGUF](https://huggingface.co/1bit-MONSTER/Qwen3.8-27B-Q4_0-H32-GGUF):
pp512 509, KLD 0.055, and 440-470 tok/s on a 1,838-token prompt through `serve`
([docs/lean.md](https://github.com/1bit-MONSTER/engine/blob/0baf286/docs/lean.md)).
