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
tags: npu, speculative-decoding

# The NPU drafts, the GPU checks

Speculative decoding makes a large model faster by letting a small one guess ahead: the small model proposes a few tokens, the large model checks them all in one pass, and every guess it agrees with is a token it did not have to produce one at a time. Usually both models run on the GPU. On Strix Halo there is a second processor sitting idle during chat: the XDNA 2 NPU. So we gave it the small model.

## How it works

The NPU runs Qwen3-0.6B on the engine's fast lane (91 tokens per second on its own). The GPU runs the target, a Qwen3 model from 14B to 32B, through llama.cpp on Vulkan. Each round:

1. The NPU proposes k tokens, one after another.
2. The GPU scores all k in a single batch and predicts the token after each.
3. The longest run of proposals the GPU agrees with is kept, plus the GPU's own next token.
4. Both key/value caches roll back to that point, and the next round starts.

Both models decode greedily, so the result should equal the GPU decoding alone, token for token. That is the correctness check we ran every time.

One constraint: the two models must share a tokenizer. Qwen3-0.6B can draft for the Qwen3 family (151,936 tokens), including Qwen3-14B and Qwen3-32B; the newer Qwen3.5, 3.6 and 3.8 models use a different vocabulary.

## Measured

Unsloth's UD-Q4_K_XL files on Strix Halo, a coding prompt, 192 tokens (136 for the 14B, which finished its answer):

| Target | k | GPU alone | NPU drafts, GPU checks | Speed-up | Drafts accepted |
|---|---|---|---|---|---|
| Qwen3-32B | 2 | 11.0 tok/s | 17.9 tok/s | 1.63x | 72% |
| Qwen3-32B | 4 | 10.2 | 21.3 | 2.09x | 59% |
| **Qwen3-32B** | **6** | **10.7** | **23.4** | **2.19x** | **56%** |
| Qwen3-32B | 8 | 10.7 | 12.9 | 1.20x | 53% |
| Qwen3-14B | 2 | 24.1 | 31.6 | 1.31x | 71% |

- **The larger the target, the more it pays.** Qwen3-32B more than doubles; the faster 14B gains 31%.
- **k = 8 is past the knee.** Checking nine tokens at once costs the GPU far more than checking seven.
- **The output matched GPU-only decoding in six of eight runs.** The other two differed at a single token, where the model's top two choices were nearly tied and the GPU's batched arithmetic broke the tie the other way.

## Against drafting on the GPU

The same 0.6B drafter can also run on the GPU next to the target, which llama.cpp does built in. We measured that too, same prompt and files:

| Target | GPU alone | 0.6B drafts on the GPU | 0.6B drafts on the NPU |
|---|---|---|---|
| Qwen3-32B | 10.4 tok/s | 24.9 tok/s | 23.4 tok/s |
| Qwen3-14B | 22.5 | 43.5 | 31.6 |

Drafting on the GPU is still a little faster: a 0.6B model costs the GPU very little, and in this first version the NPU and the GPU take turns. What the NPU gives is the drafting done on silicon that was otherwise idle, with the GPU never running the small model at all. Making each NPU draft step cheaper is the next piece of work.

The benchmark is [`tests/npu_spec_bench.cpp`](https://github.com/1bit-MONSTER/engine/blob/npu-draft/tests/npu_spec_bench.cpp) on the `npu-draft` branch.
