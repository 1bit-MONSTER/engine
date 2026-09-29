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
tags: hrx, milestone, kernels
summary: One barrier that fenced only shared memory caused HRX's NaNs and GPU faults; the fix took Qwen3-Coder-30B from 1 of 8 correct runs to 8 of 8.

# Milestone: the HRX decode race is solved

Yesterday's post ended with a workaround. HRX's split flash-attention decode kernel gave
different answers to identical requests, and on Qwen3-Coder-30B it sometimes produced NaNs that
faulted the GPU ([#123](https://github.com/1bit-MONSTER/engine/issues/123),
[#140](https://github.com/1bit-MONSTER/engine/issues/140)). So `1bit serve --device hrx` stopped
using it. Today we found the cause and fixed it, and the kernel is back on by default. It is
correct again, and faster than the path we used without it.

## What it was

The decode-split kernel attends over the KV cache in blocks, and a final workgroup combines the
blocks. That workgroup writes the attention output to global memory. Other waves then read it
back to quantize it (`next_q8`) for the next projection. Between the write and the read sat a
workgroup barrier that fences shared memory (LDS) only. It does not wait for global stores or
invalidate the cache the readers use. So a reading wave could pick up the output slot's
*previous* contents. The f32 output was right, the quantized copy was wrong, and when the stale
bytes happened to decode as NaN the MoE router saw NaN logits.

The window is narrow on a quiet box (a few percent of requests diverged there) and wide when the
GPU preempts the dispatch in the middle. Any process's KFD queue eviction does that: page
compaction, KSM, memory pressure. That is why the failure rate followed those system settings. For a while it looked like a page-migration bug, but migration was only
the trigger. The fix fences both memory spaces at all five places the kernel does this:
`barrier<global>`, then `barrier<workgroup>`
([docs/hrx.md](../docs/hrx.md#moe-router-expert-id-fault-fix-engine123)).

## How we know

A separate process forced queue evictions while the engine served, to make the rare case common:

- **Qwen3-Coder-30B-A3B** (2113-token prompt): 1 of 8 correct before, 8 of 8 after, with no NaN.
- **Qwen3-0.6B**, greedy: 13-34 divergent answers per ~177 requests before, none in 354 after.

The last clue came from hashing every kernel's output in serial execution. In every bad request,
the first command whose output diverged was this kernel, with the f32 output identical and the
quantized output different.

## Faster with it on

Decode tok/s (Q4_K_M, llama-bench, interleaved on a shared box):

| model | ctx 2100, split on | ctx 2100, split off |
|---|---|---|
| Qwen3-0.6B | 140-158 | 118-122 |
| Qwen3-Coder-30B-A3B | 42-62 | 47-49 |
| ZAYA1-8B | 38-42 | 34-36 |

The kernel's multi-pass output step was also vectorised
([#180](https://github.com/1bit-MONSTER/engine/pull/180)): on Qwen3-Coder-30B-A3B, +15% at
depth 2100, +20% at 3000 and +22% at 4800 (median of 6 interleaved rounds), with bit-identical
perplexity. `ONEBIT_HRX_DECODE_SPLIT=0` turns the kernel off if you need to compare.

## Also this week

- **Architecture gaps closed.** OPT, GPT-Neo, CodeGen and GPT-J, which upstream llama.cpp has no
  model code for, and Zyphra's whole family, from ZAYA1-74B to the vision models, all run from our
  llama.cpp. The registry now maps 323 Hugging Face architectures, up from 265 at step 5
  ([docs/registry.md](../docs/registry.md)).
- **GGUF on the NPU**: a repacked GGUF serves on the NPU. Qwen2.5-7B, MiniCPM4-8B and
  MiniCPM5-1B answer there
  ([#179](https://github.com/1bit-MONSTER/engine/pull/179), [docs/npu.md](../docs/npu.md#from-a-gguf)).

Packages with the fix ship on Sunday with the weekly release.
