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
tags: hrx, vulkan, zero-copy

# Prefill on HRX, decode on Vulkan, one KV cache

On Strix Halo's Radeon iGPU, AMD's HRX runtime prefills a prompt faster than Vulkan, and Vulkan decodes faster than HRX. `1bit serve --device vulkan --prefill-device hrx` now uses both on the same request, and the key-value cache never gets copied between them.

## The idea

A request has two phases. **Prefill** reads the whole prompt at once and is compute-bound; **decode** makes one token at a time and is bound by memory bandwidth. Our one llama.cpp build carries both GPU backends, `HRX0` and `Vulkan0`, so the engine loads the model on each, runs the prompt prefix on HRX, and hands the rest of the request to Vulkan.

The hand-off is the hard part. Prefill's output is the KV cache: every layer's keys and values for every prompt token. Copying it from one backend to the other would cost the time the split saves.

## Zero copies

So both contexts use **one** KV cache:

- HRX allocates it in its own device memory and exports it as a **dma-buf** (through HSA).
- Vulkan imports the same memory with `VK_EXT_external_memory_dma_buf` and reads it in place, at 110.6 GB/s against 113.2 GB/s for its own allocations.
- The only thing that moves between them is the cache's bookkeeping: which cells hold which positions of which sequence.

We tried sharing host memory first. It worked, but it cost Vulkan half its decode speed; device memory through a dma-buf does not.

Both sides compute the cache's layout independently, so they must agree on it exactly: tensors on 4 KiB boundaries, split into chunks of at most 1 GiB (Vulkan reads garbage from a single buffer past about 4 GiB), and a hash of the layout checked on both sides. HRX's prefill is fastest in whole batches, so only whole ubatches of the prompt go to HRX, and short prompts skip the split: below 1024 tokens (`--prefill-min-tokens`) it does not pay.

## Measured

llama-server on Strix Halo, prefill and whole request:

| Model | Prompt | Vulkan alone | HRX prefill + Vulkan |
|---|---|---|---|
| Qwen2.5-7B Q4_K_M | 2048 | 1541 / 4314 ms | **1078** / 3890 ms (-10%) |
| Qwen2.5-7B Q4_K_M | 8192 | 7347 / 10376 ms | **4646** / 7668 ms (**-26%**) |
| Qwen3-0.6B Q4_K_M | 8192 | 1289 / 2267 ms | **1089** / 2032 ms (-10%) |

Decoding on the shared cache runs within 3% of Vulkan with its own cache. The output is the same model: against Vulkan alone, teacher-forced over 64 tokens, the mean KL divergence is 0.0002-0.0006 nats and the top token agrees on 64-65 of 65 positions. For scale, Q4_K_M quantization itself is 0.063 against BF16.

## Where it lives

The patches are in our llama.cpp fork, on top of AMD's tested HRX commit, and are rebased onto every new AMD pair automatically: dma-buf export in ggml-hrx, dma-buf import in ggml-vulkan, the cache sharing in llama, and the prefill hand-off in llama-server. The weights are loaded twice, once per device, so it suits machines with memory to spare; Strix Halo's unified memory is one. The details are in the [HRX docs](../docs/hrx.md).
