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
tags: rocm, quantization, qwen, kernels
summary: Hadamard-rotated Q4_0 with 4-bit activations reads Qwen3.8-27B prompts at 509 tok/s (from ~400) at KLD 0.055, and 1bit serve runs it by itself.

# 509 tokens a second of prompt: the Hadamard route ships

Earlier today we measured a way to make Qwen3.8-27B read prompts faster on Strix Halo: 4-bit
activations on the GPU's matrix units, with a Hadamard rotation to keep them accurate. It is no
longer an experiment. The engine now has a tool that makes the rotated file, `1bit serve`
recognises the file and runs it on the right build by itself, and the Qwen3.8-27B file is on
Hugging Face as
[1bit-MONSTER/Qwen3.8-27B-Q4_0-H32-GGUF](https://huggingface.co/1bit-MONSTER/Qwen3.8-27B-Q4_0-H32-GGUF).

## Why prompts are a matrix-unit problem

Decoding reads the whole model once per token, so memory bandwidth sets its speed. Reading a
prompt is different: hundreds of tokens go through each weight at once, and the matrix units set
the pace. On the Radeon 8060S those units multiply 8-bit integers at one rate and 4-bit integers
at twice that rate. The usual path turns activations into 8-bit numbers. On this model it reads
prompts at about 400 tokens a second.

A Q4_0 file already stores its weights as 4-bit numbers. A code `q` means `q - 8`, and flipping
its top bit gives exactly that value in two's complement. So the 4-bit kernel can read Unsloth's
Q4_0 weights bit for bit, and only the activations have to go down to 4 bits. That runs at 461-488
tokens a second, but it costs accuracy: KLD against the full model rises from 0.029 to 0.084.

## The rotation

Activations are rounded in blocks of 32 values that share one scale. A few large values in a
block force a coarse scale on the other 31, and most of the 4-bit error comes from there. A
32-point Walsh-Hadamard transform spreads each block's energy evenly before rounding, so the
scale fits all 32 values better.

The same transform is applied to the weights once, when the file is made. Because the transform
is orthogonal, rotating both sides leaves every dot product unchanged: `x . w = (Hx) . (Hw)`. The
check is the exact 8-bit path: the rotated file gives KLD 0.031 there, against 0.029 for the same
model unrotated. On the GPU the rotation runs inside the step that quantizes activations: two
butterfly steps within each thread and three lane shuffles.

## The numbers

Qwen3.8-27B on Strix Halo, KLD against BF16 over wikitext-2, prompt speed from llama-bench (512
tokens):

| | Prompt, tok/s | Perplexity | KLD | Same top token |
|---|---|---|---|---|
| Q4_0, 8-bit activations | ~400 | 6.021 | 0.029 | 91.9% |
| Q4_0, 4-bit activations | 461-488 | 6.241 | 0.084 | 87.4% |
| **Q4_0-H32, rotated, 4-bit activations** | **509** | **6.096** | **0.055** | **89.3%** |

The rotation removes a third of the 4-bit error. Against the 8-bit path, the rotated file reads
prompts 28% faster for 1.2% more perplexity. Through `1bit serve`, a 1,838-token prompt runs at
440-470 tokens a second.

## Making the file is part of the engine

A rotated file only works when the runtime rotates the activations too. Run it anywhere else and
it produces garbage. So the engine treats the file and the route as one thing.

- `tools/hadamard_q4_0.py` rotates the attention, FFN and delta-net projections of a Q8_0 or BF16
  source, adjusts the imatrix to match, and quantizes. It then stamps the file and checks that
  every Q4_0 tensor in it is a rotated one; a file that fails is deleted. Qwen3.8-27B took about
  20 minutes.
- `1bit serve` reads the stamp. With no device flag it runs the file on the lean ROCm build with
  the rotation and the 4-bit path turned on. It refuses the file on any other device.
- A test covers the routing without a GPU, so CI checks it on every change.

```sh
tools/hadamard_q4_0.py Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-Q4_0-H32.gguf --imatrix imatrix.gguf
1bit serve -m Qwen3.8-27B-Q4_0-H32.gguf
```

## Two more kernels on the way

Qwen3.8's linear-attention layers had two slow steps on ROCm, and the same update fixes both. A
new prefill kernel keeps each state column in registers and stages each token's inputs in shared
memory. A tiled transpose replaces the strided copy that fed their convolution. Both match the CPU
reference in the backend tests, and every model with these layers gets them on ROCm.

## What it is for

This route reads prompts. It decodes at about 11 tokens a second, so long answers belong
elsewhere: the Vulkan route with the DFlash2 drafter decodes code at 45.7 tokens a second on the
same model. Use the rotated file where prompts are long and answers short, like summaries, code
review over whole files, or retrieval with large contexts. The details are in
[docs/lean.md](../docs/lean.md).
