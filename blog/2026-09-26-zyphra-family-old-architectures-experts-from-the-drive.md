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
tags: zyphra, models, moe, hrx, onnx

# Zyphra's whole family, four older architectures, and experts from the drive

A lot landed this week. Zyphra's model family now runs end to end, including its vision
models. Four older architectures that upstream llama.cpp never shipped are in. `1bit serve` can
run a MoE model whose experts do not fit in memory. And the HRX decode path became deterministic.
Every number below was measured on Strix Halo (Ryzen AI Max+ 395, Radeon 8060S), with the
method in the linked docs.

## Zyphra, all of it

ZAYA1-8B already ran on Vulkan, HRX and ROCm. The rest of the family followed:

- **ZAYA1-74B-preview**, with its sliding-window layers: Q4_K_M decodes at 35.4 tok/s on Vulkan.
- **ZAYA1-base and ZAYA1-reasoning-base**, in Zyphra's older checkpoint layout. Our converter
  reads that layout directly, and the tensors it writes are byte-identical to ZAYA1-8B's.
- **ZAYA1-VL-8B**, the vision model. It applies a vision-only LoRA to image tokens and attends to
  each image bidirectionally. Against Zyphra's own FP32 code it agrees at 100 of 101
  teacher-forced positions. F16 decodes at 38-51 tok/s. It reads "HELLO 42" off a synthetic
  image, and it recognises the New York Times front page of the moon landing.
- **Zamba, Zamba2 and BlackMamba**, plus **Zamba2-VL**. Zamba v1 and BlackMamba match the
  reference at 96/96 positions. A new Vulkan scan for Mamba-1 layers took Mamba-370M from 25.8
  to 173.5 tok/s; those layers used to run on the CPU.

Adding the vision model left ZAYA1-8B's text path unchanged. Its wikitext perplexity is
identical on Vulkan (21.5731, and 21.6213 with four sequences per batch) and HRX (21.5518), and
it still agrees with transformers at 95/96. Serve the vision models with `1bit serve --mmproj`
([docs/vulkan.md](../docs/vulkan.md#zaya1-vl-8b-images)).

## Four older architectures

OPT, GPT-Neo, CodeGen and GPT-J now run from our llama.cpp. We checked each port against
transformers FP32 before it merged. CodeGen started at 0/96: its fused projection stores query,
value and key in that order, not query, key, value. After the fix all four agree at every
checked position ([docs/vulkan.md](../docs/vulkan.md#opt-gpt-neo-codegen-and-gpt-j-from-our-llamacpp)).
With them, the model registry maps 323 Hugging Face architectures to a backend. "Mapped" means
a backend's code accepts the architecture; the census reports separately how many have been
checked ([docs/registry.md](../docs/registry.md)).

## MoE experts from the drive

`1bit serve --moe-slots N` keeps a MoE model's routed experts in the file on disk. It holds N of
them in GPU memory and streams in the rest as the router asks for them. Perplexity is exactly
that of the all-GPU model. On Qwen3-Coder-30B-A3B, with 75% of the experts resident and the
prefetch moved off the decode path, it decodes at 43-45 tok/s (39 through `serve`). With 25%
resident it still decodes at 8-10. The resident model is faster (79-91 tok/s), so streaming
is for models that do not fit
([docs/moe-streaming.md](../docs/moe-streaming.md#streaming-in-the-inference-path)).

## HRX decode, now deterministic

Repeating the same request exposed a problem in HRX's split flash-attention decode kernel: it
gave different answers to identical requests, and on Qwen3-Coder-30B it faulted the GPU in two
of three runs. `1bit serve --device hrx` now decodes without it by default. That path is
deterministic, and on most models faster: Qwen3-0.6B goes from 169 to 322 tok/s, and ZAYA1-8B
from 23.5 to 25.5. Coder-30B runs 5 of 5 clean at 66-71 tok/s.
`ONEBIT_HRX_DECODE_SPLIT=1` turns the kernel back on for testing a fix
([docs/hrx.md](../docs/hrx.md)).

## Also

- **ONNX on the GPU on Linux**: ONNX Runtime's WebGPU provider, over Vulkan. Qwen3-4B int4
  decodes at 52.9 tok/s on the Radeon, against 26.8 on the CPU ([docs/onnx.md](../docs/onnx.md)).
- **GPU matrix units**: a short reference on the Radeon's matrix instructions. int8 runs at the
  f16 rate; only int4 × int4 doubles it ([docs/hrx.md](../docs/hrx.md#the-gpus-matrix-units-wmma)).
- **Issues and security**:
  - issue forms, and RFCs for larger changes in Discussions;
  - a CI check that stops a pin from moving backwards;
  - CodeQL's extended suite, with no open alerts.

Packages for all of this ship on Sunday with the weekly release.
