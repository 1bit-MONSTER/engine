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

# MoE streaming from NVMe

The goal: run mixture-of-experts models larger than Strix Halo's 128 GB. Only the experts a
token routes to are needed. The hot ones stay in RAM, and the rest come from the NVMe drive
as the router asks for them. The work splits three ways:
- the dense layers (attention, shared experts, embeddings) on the GPU (HRX or Vulkan);
- the routed experts wherever they run fastest: NPU (native int8 x int4) or GPU;
- the NVMe feeding the cache.

This page holds the measurements, the expert cache (`moe/`, `1bit moe-cache`) and the
tools that produced them (`tools/moe_trace.cpp`, `moe_policy.cpp`, `moe_expert_bench.cpp`).

## What one token needs

Qwen3.8-Flash-Next (`qwen4exp`) UD-Q4_K_XL, from its GGUF metadata:

| | |
|---|---|
| layers | 48 |
| experts per layer / routed per token | 512 / 10 (+ 1 shared) |
| routed expert weights | 71.7 GiB, 3.06 MiB per expert per layer |
| per-layer token embeddings | 26.8 GiB (llama.cpp's `--lazy-mode` already reads their rows on demand) |
| everything else | 4.9 GiB |

A token reads 48 × 10 × 3.06 MiB = 1.47 GB of routed experts. The shared weights stay
resident.

## The drive

Strix Halo's NVMe (YMTC PC41Q 2 TB), `O_DIRECT` reads from a model file:

| read | throughput |
|---|---|
| sequential, 16 MiB | 4.91 GB/s |
| random, 3 MiB (one expert), one at a time | 3.62 GB/s (0.87 ms each) |
| random, 3 MiB, 8 in flight | 3.87 GB/s |

With nothing cached, 1.47 GB per token at about 3.7 GB/s caps decode at about 2.5 tok/s. So
the cache decides the speed.

On 2026-09-26, with other tenants active (load average 6-9), the same drive gave 1.2-4.2
GB/s for random 1-3 MiB reads, 8 in flight, depending on the moment. Every measurement below
that reads the drive carries that spread.

## How often experts repeat

`tools/moe_trace.cpp` records each token's routed experts per layer, then replays them
through an LRU cache per layer. The cache is warmed by the prompt, and the hit rate counts
decode tokens only. Greedy decode, 300 tokens (Qwen3.6: 400), on Vulkan:

| experts cached per layer | Flash-Next, prose | Flash-Next, code | Qwen3.6-35B-A3B, code (of 256) |
|---|---|---|---|
| 64 | 68.1% | 69.6% | 82.5% |
| 128 | 85.7% | 84.5% | 93.8% |
| 192 | 92.2% | 91.4% | 97.4% |
| 256 | 94.5% | 94.5% | all |
| 384 | 96.5% | 97.7% | |

For Flash-Next, the per-token read from the drive and the ceiling that sets are:

| cache | RAM for experts | misses | read per token | NVMe ceiling |
|---|---|---|---|---|
| 128 / layer | 18.4 GiB | ~15% | ~0.22 GB | ~17 tok/s |
| 192 / layer | 27.5 GiB | ~8% | ~0.12 GB | ~31 tok/s |
| 256 / layer | 36.7 GiB | ~5.5% | ~0.08 GB | ~46 tok/s |

These are ceilings from the drive alone, before compute, and from two prompts. The next
sections add traces, compare policies and measure the real cache.

## More traces: three families, three kinds of use

`tools/moe_trace.cpp --chat` runs a conversation through the model's chat template and
records every token's routed experts per layer. Prompt and decoded tokens are marked
separately. All traces use greedy decoding on Vulkan with the engine's llama.cpp pin
(2026-09-25), in three shapes:
- **code:** one coding request, then 400 tokens;
- **chat:** six user turns on unrelated topics, up to 256 reply tokens each;
- **long:** about 30,000 tokens of the engine's own docs, a question, then 400 tokens.

| model | family | layers | experts / routed |
|---|---|---|---|
| Qwen3-Coder-30B-A3B Q4_K_M | qwen3moe | 48 | 128 / 8 |
| Qwen3.6-35B-A3B UD-Q5_K_XL | qwen35moe (DeltaNet hybrid) | 40 | 256 / 8 |
| GLM-4.7-Flash Q4_K_M | deepseek2 (MLA, sigmoid router) | 47 (46 MoE) | 64 / 4 + shared |

`tools/moe_policy.cpp` replays a trace through several cache policies. Prompt tokens warm
the cache, and only decode tokens count.
- **lru, lfu, slru:** one cache per layer (slru is segmented: 20% probation).
- **g-lru, g-lfu:** one budget shared by all layers (g-lfu halves its counts every 64 tokens).
- **opt:** Belady's policy, which evicts the expert used furthest in the future. It needs
  the future, so it is the bound.

Decode hit rate with a cache of a quarter and of half of each layer's experts:

| trace | cache / layer | lru | lfu | slru | g-lru | g-lfu | opt |
|---|---|---|---|---|---|---|---|
| Coder-30B code | 32 of 128 | 68.4 | 65.6 | 66.9 | 68.2 | 68.1 | 83.5 |
| | 64 | 87.8 | 88.5 | 88.1 | 89.3 | 90.4 | 94.9 |
| Coder-30B chat | 32 | 85.2 | 73.3 | 85.7 | 86.2 | 83.9 | 93.0 |
| | 64 | 95.8 | 92.8 | 96.2 | 96.5 | 96.2 | 98.6 |
| Coder-30B long | 32 | 80.3 | 81.4 | 81.1 | 81.3 | 82.0 | 90.8 |
| | 64 | 95.0 | 95.6 | 95.3 | 96.3 | 96.5 | 98.3 |
| Qwen3.6-35B code | 64 of 256 | 80.5 | 78.7 | 80.8 | 82.0 | 81.1 | 91.0 |
| | 128 | 92.6 | 92.2 | 91.3 | 94.0 | 94.1 | 97.1 |
| Qwen3.6-35B chat | 64 | 79.9 | 66.1 | 80.3 | 80.5 | 78.1 | 90.8 |
| | 128 | 91.7 | 88.5 | 92.2 | 93.1 | 93.6 | 97.7 |
| Qwen3.6-35B long | 64 | 79.0 | 77.4 | 80.5 | 79.2 | 79.1 | 89.5 |
| | 128 | 92.4 | 92.6 | 92.8 | 94.2 | 93.8 | 96.8 |
| GLM-4.7-Flash code | 16 of 64 | 63.7 | 62.3 | 63.1 | 64.0 | 65.2 | 80.1 |
| | 32 | 83.7 | 84.1 | 83.2 | 85.1 | 86.5 | 92.9 |
| GLM-4.7-Flash chat | 16 | 65.0 | 52.6 | 65.0 | 65.8 | 63.5 | 81.6 |
| | 32 | 85.2 | 80.0 | 85.6 | 87.1 | 86.6 | 94.1 |
| GLM-4.7-Flash long | 16 | 59.7 | 61.0 | 60.2 | 60.2 | 63.0 | 78.2 |
| | 32 | 81.2 | 83.9 | 83.4 | 82.5 | 85.1 | 92.5 |

What the traces show:
- **One budget shared across layers** (g-lru) is the best simple policy, or within a point
  of it, on every trace. Some layers need more slots than others, and a shared budget gives
  them those slots.
- **LFU fails on chat.** A new topic makes last topic's counts stale: Coder chat at 32 per
  layer is 73.3% against 85.2% for LRU, and Qwen3.6 chat at 64 is 66.1% against 79.9%.
  Frequency with decay (g-lfu) recovers most of it, and is marginally the best on long
  context and at half-size caches.
- **Chat and long context repeat experts more than a single request.** Coder at 32 per
  layer: code 68%, long 80%, chat 85%.
- **The bound is 5-16 points higher** at quarter-size caches, so a smarter policy still has
  room. The traces are kept, so any policy can be tried on the same data.

### Predicting the next layer's experts

A trace also records a gate-ahead prediction for every decoded token. It applies layer l's
router (with layer l's FFN norm, and GLM's selection bias) to the output of layer l-1: the
engine has that vector while layer l's attention runs. The two-ahead prediction does the
same from layer l-2's output, one layer earlier.

| trace | routing predicted (top-k) | misses covered, top-k | two-ahead: predicted | two-ahead: misses covered |
|---|---|---|---|---|
| Coder-30B code / chat / long (32) | 87.3 / 89.9 / 89.1% | 84.0 / 84.1 / 83.0% | 79.8 / 83.9 / 83.4% | 71.4 / 70.3 / 70.0% |
| Qwen3.6-35B code / chat / long (64) | 83.6 / 83.5 / 82.5% | 74.1 / 76.5 / 75.2% | 75.7 / 75.6 / 74.3% | 57.4 / 60.5 / 59.3% |
| GLM-4.7-Flash code / chat / long (16) | 86.2 / 87.5 / 83.1% | 82.2 / 85.2 / 81.1% | 74.2 / 77.1 / 71.2% | 63.1 / 69.0 / 65.1% |

Predicting 2 x top-k covers 93-98% of misses, but reads 5-10 times as many experts that are
never used. Predictors that need only the trace (the previous token's experts,
co-occurrence with the previous layer) cover under 30%.

## The expert cache in the engine

`moe/` holds the cache (Linux only, built into `1bit` as `ONEBIT_MOE`):
- **`gguf_index`** reads a GGUF header, including every shard of a split file, and finds each
  routed expert's bytes: `blk.N.ffn_{gate,up,down}_exps.weight`, one slice per expert.
- **`expert_cache`** keeps experts in pinned slots. One `mmap` region is locked with `mlock`,
  and each slot holds all of one expert's parts, each part aligned to 4 KiB for `O_DIRECT`.
  - Misses are read with `O_DIRECT`, straight from the drive into the slot, by a pool of
    reader threads, so several reads are in flight. Demand reads go ahead of prefetches.
  - `acquire(layer, experts)` blocks until the experts are resident and pins them until
    `release`. `prefetch(layer, experts)` only queues reads.
  - Eviction is LRU, over one budget shared by all layers (the replays favour it) or per layer.
  - Slots come in size classes. Unsloth's UD quants use bigger types for some layers' experts,
    so layers are grouped by the bytes one expert needs, and each group gets its own slots
    (same share of experts per layer). Flash-Next's 9,216 slots pin 27.0 GiB this way,
    instead of 34.4 GiB with one slot size.
- **`1bit moe-cache`** replays a trace through the cache against the real model file, with
  real reads. Compute is simulated per layer:
  1. at the layer's start, the prefetch is issued (lookahead 1: this layer's gate-ahead
     prediction; lookahead 2: the next layer's two-ahead prediction);
  2. attention runs (`--attn-frac` of the layer's compute time);
  3. `acquire` loads the routed experts, stalling on any read still in flight;
  4. the experts compute (the rest of the layer's time).

  It prints the decode rate the drive and cache allow, the stall per token, hits, prefetch
  hits, demand misses, wasted prefetches and bytes read.

### What the cache delivers

`1bit moe-cache` sweep, 2026-09-25/26:
- **Traces:** all nine, 120 decode tokens each, the cache warmed by the last 64 prompt tokens.
- **Compute per token:** the model's Vulkan0 decode rate with every expert resident. That is
  12.1 ms for Coder-30B, 21 ms for Qwen3.6-35B and 14.1 ms for GLM, split half before and half
  after the experts are needed.
- **Box:** Strix Halo was shared, with a load average of 6-9, other tenants' GPU servers, and
  18 GB of their memory swapped out. A run can only be slowed by that, so the table takes the
  better of two runs.
- **Noise:** small caches still vary by up to about 30% between runs, so the table shows the
  operating point that matters, 75% of the experts in RAM.
- **Drive ceiling:** `moe_policy` computes it from the trace alone, with the shared LRU's
  misses at 3.7 GB/s. It is free of noise, and it covers the whole trace, so the cache is
  warmer than in the sweep's 120 tokens.

| decode tok/s, 75% of experts in RAM | trace | no prefetch | gate-ahead | two-ahead | drive ceiling, serial | drive ceiling, overlapped | all resident |
|---|---|---|---|---|---|---|---|
| Coder-30B, 4,608 slots (13.2 GiB) | code | 28.8 | 32.1 | **33.9** | 53.3 | 82.6 | 82.6 |
| | chat | 45.3 | 46.6 | **48.4** | 58.1 | 82.6 | |
| | long | 24.6 | 28.2 | **31.6** | 61.4 | 82.6 | |
| Qwen3.6-35B UD-Q5, 7,680 slots (≈16.5 GiB) | code | 30.3 | **33.2** | 31.8 | 37.4 | 47.6 | 47.6 |
| | chat | 35.3 | 37.4 | **39.3** | 36.5 | 47.6 | |
| | long | 33.4 | 35.8 | **37.6** | 40.0 | 47.6 | |
| GLM-4.7-Flash, 2,208 slots (≈12.6 GiB) | code | 25.6 | 30.2 | **38.3** | 37.9 | 70.9 | 70.9 |
| | chat | 38.6 | 39.6 | **39.6** | 39.3 | 70.9 | |
| | long | 28.7 | 26.8 | **32.9** | 37.3 | 70.9 | |

What the sweep shows:
- **Prefetch pays at a large cache.** Two layers ahead is the best mode, or ties it, on 8 of
  9 traces: 5-50% over no prefetch. It turns demand misses (2.6-6.2%) into prefetch hits
  and leaves 0.7-2.5% of reads on the critical path. It reads 10-50% more bytes, so it only
  pays when the drive has slack.
- **Prefetch hurts at small caches.** At 25-50% of experts in RAM, the drive is the limit.
  Prefetch adds bytes (10-30% of its reads are wasted) and pushes out experts that were still
  needed. Two layers ahead predicts worse but gives a whole layer of lead, while one ahead
  gives only half of a layer's compute, about 0.1 ms: less than one 3 MiB read (0.9 ms).
- **The cache decides the speed.** Coder-30B chat goes from 15-18 tok/s with 25% of experts
  (4.4 GiB) to 45-48 with 75% (13.2 GiB). The all-resident rate (93 tok/s) needs all
  17.3 GiB.
- **Reads come in 3-6 MiB units at 3-3.9 GB/s.** Four to sixteen reads in flight make no
  measurable difference on this drive.

## The three-way split, measured

Each MoE layer runs its attention and shared part (dense), then its routed experts. Batch-1
decode is serial across layers, so a split only pays if the experts' device is faster than
the dense device would be. On Strix Halo every device reads the same LPDDR5X, so the
question is which one reads weights fastest.

**The expert block alone.** `tools/moe_expert_bench.cpp` times one layer's routed experts for
one decode token: gate and up MUL_MAT_ID, SwiGLU, then down. It uses the model's real layer-10
weights and fresh random top-k experts every run (median of 40). The NPU row comes from the
closed-source Qwen3.6-35B NPU route's on-device stage profile (2026-09-24). That route
requantizes to q4_1, and its block includes the shared expert.

| expert block per layer | Vulkan0 | HRX0 | CPU (32 threads) | NPU |
|---|---|---|---|---|
| Coder-30B Q4_K, top-8 (21.2 MB) | 0.137 ms, 155 GB/s | 0.511 ms, 42 GB/s | 0.337 ms, 63 GB/s | |
| Qwen3.6-35B UD-Q5_K_XL, top-8 (18.4 MB) | 0.136 ms, 135 GB/s | 0.432 ms, 43 GB/s | 0.287 ms, 64 GB/s | |
| Qwen3.6-35B Q8_0, top-8 (26.7 MB) | 0.156 ms, 171 GB/s | 0.366 ms, 73 GB/s | 0.322 ms, 83 GB/s | 0.459 ms, 39 GB/s (q4_1 + shared, 17.9 MB) |
| GLM-4.7-Flash Q4_K/Q6_K, top-4 (24.5 MB) | 0.155 ms, 158 GB/s | 0.540 ms, 45 GB/s | 0.369 ms, 66 GB/s | |

HRX0 runs Qwen MoE layers through fused router and FFN dispatches in a real model, so its
standalone number understates it for Qwen. In whole models, HRX0 decodes Coder-30B at 90.5
tok/s against Vulkan0's 92.4.

**Whole models, split both ways.** `llama-bench -fa 1 -n 64 -r 3`, pin 96f6b89. The routed
expert tensors go to a second device with `-ot exps=<device>`, and everything else stays on
the first (`-ts 1/0`):

| decode tok/s | all Vulkan0 | all HRX0 | dense HRX0, experts Vulkan0 | dense Vulkan0, experts HRX0 | dense Vulkan0, experts CPU |
|---|---|---|---|---|---|
| Qwen3-Coder-30B Q4_K_M | **92.4** | 90.5 | fails (#108) | 27.0 | 43.5 |
| Qwen3.6-35B UD-Q5_K_XL | **52.9** | 31.5 | fails (#108) | 25.5 | 30.2 |
| GLM-4.7-Flash Q4_K_M | **71.1** | 22.1 | 25.2 | 25.9 | 36.0 |

Every split loses to one GPU device. Each layer hands its activations to the other device and
back, and the other device reads experts no faster than Vulkan0. The NPU reads experts at
39-45 GB/s (the 35B route's MoE waves run at 80-84% of its ~53-56 GB/s streaming peak).
Vulkan0 reads them at 135-171 GB/s. So for batch-1 decode:
- the experts belong on Vulkan0;
- HRX0 is an alternative for the whole model, not for half of it;
- the NPU is worth using for a second model running at the same time, not for half of this
  one. On 2026-09-24, once both were loaded, the 35B NPU route decoded at 11.0 tok/s (11.1
  solo) while a Vulkan decode ran at about 50. That is roughly 61 tok/s across two
  independent streams, with no visible interference.

Dense on HRX0 with the experts elsewhere fails for Qwen MoE models: HRX's fused Qwen router
assumes the experts are on HRX0 too (#108). GLM, whose sigmoid router is not fused, runs.

## The sweet spot per model

Four axes: quantization (quality against speed), split, cache size (the RAM given to
experts; the rest streams from the drive) and MTP.
- **Quality:** KL divergence of each quant's token distribution from Q8_0's, and how often
  its top token matches, over 40 chunks of 512 tokens (Unsloth's UD quants, measured
  2026-09-24).
- **Speed:** Vulkan0 decode (tg64) and prompt (pp512), `llama-bench -fa 1 -r 3`.
- Quants that are not on the box (the drive has 18 GB free) show sizes and quality only.
  Decode speed does not follow file size closely enough to estimate it: Qwen3.6-35B's Q8_0 is
  39% larger than its UD-Q5_K_XL, yet only 5-10% slower.

**Qwen3-Coder-30B-A3B** (48 layers, 128 experts, top-8)

| quant | GiB | KLD vs Q8_0 | same top token | decode tok/s | prompt tok/s |
|---|---|---|---|---|---|
| UD-Q2_K_XL | 10.98 | 0.113 | 87.1% | not on disk | |
| UD-Q3_K_XL | 12.86 | 0.058 | 90.4% | not on disk | |
| UD-Q4_K_XL | 16.45 | 0.027 | 93.4% | not on disk | |
| Q4_K_M | 17.28 | not measured | | **92.4-93.1** | 1290 |
| UD-Q5_K_XL | 20.25 | 0.012 | 95.6% | 75.3 | 1052 |
| TQ2_0 (ternary) | 7.6 | not measured | | 29.5 (no fast ternary path on Vulkan) | 129 |

**Qwen3.6-35B-A3B** (40 layers, 30 DeltaNet + 10 attention, 256 experts, top-8)

| quant | GiB | KLD vs Q8_0 | same top token | decode tok/s | prompt tok/s |
|---|---|---|---|---|---|
| UD-Q2_K_XL | 11.45 | 0.113 | 85.8% | not on disk | |
| UD-Q3_K_XL | 15.69 | 0.045 | 90.8% | not on disk | |
| UD-Q4_K_XL | 20.82 | 0.014 | 94.6% | not on disk | |
| UD-Q5_K_XL | 24.77 | 0.009 | 95.8% | **52.3-52.9** | 889 |
| Q8_0 | 34.37 | reference | | 47.9-49.8 | 1269 (2026-09-24) |
| NPU route (closed source, q4_1) | | parity vs fp64 | | 16.3-16.5 | ~21 |

**GLM-4.7-Flash** (47 layers, MLA, 64 experts, top-4 + shared): Q4_K_M only, 71.0-71.1 tok/s decode
on Vulkan0 and 22.1-22.4 on HRX0. No other quant is on the box, so there is no quality axis.

**Split:** all on Vulkan0 for every model (see the three-way split). HRX0 is the whole-model
alternative for Coder-30B (90.5 tok/s).

**MTP:** Unsloth ships no MTP head for these three models. Qwen3.8-Flash-Next has one. With
its UD-Q4_K_XL fully resident and the MTP head drafting 3 tokens, Vulkan decodes at 40-49 tok/s
against 25-26 without (2026-09-24, on Unsloth's `qwen4exp/mtp` branch; engine #110 carries the
upstream MTP fix). With MTP, each verification step routes several tokens at once, so a layer
needs the union of their experts. The Flash-Next section below measures what that costs a
streamed cache.

### The picks

On Strix Halo with the whole box, every model here fits in RAM, so the pick is the fastest
quant whose quality is acceptable, all on Vulkan0. When the RAM must be shared (other models
resident, as in a fleet), the cache sets the speed:

| model | whole box | decode tok/s | quality (KLD vs Q8_0) | experts in RAM (75%) + two-ahead prefetch | decode tok/s, measured |
|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | Q4_K_M, all on Vulkan0 | 92-93 | UD-Q4_K_XL: 0.027 at a similar size | 13.2 GiB of 17.5 GiB | 32-48 (code 34, chat 48, long 32) |
| Qwen3.6-35B-A3B | **UD-Q5_K_XL**, all on Vulkan0 (faster than Q8_0 at nearly its quality) | 52-53 | 0.009 | ≈16.5 GiB of ≈21.9 GiB | 32-39 |
| GLM-4.7-Flash | Q4_K_M, all on Vulkan0 | 71 | no other quant measured | ≈12.6 GiB of ≈16.8 GiB | 33-40 |
| Qwen3.8-Flash-Next | UD-Q4_K_XL + MTP n = 3, resident on Vulkan (about 77 GiB) | 40-49 | | 27.5 GiB of 71.7 GiB (27.0 GiB pinned) | 7-12 on this drive (ceiling 24-34 at 3.7 GB/s) |

MTP is available only for Flash-Next, and it is worth it both ways: 1.6-1.9 times resident,
and it leaves the drive's reads per token unchanged when streamed.

## Qwen3.8-Flash-Next: the model that needs streaming

Flash-Next UD-Q4_K_XL is 111 GB: 71.7 GiB of routed experts, 26.8 GiB of per-layer token
embeddings and 4.9 GiB of everything else. It has 48 layers of 512 experts, 10 routed per
token, at 3.06 MiB each. With the whole box to itself it fits (about 77 GiB resident, with the
embeddings read lazily). On a box shared with other models it does not.

Traces: `MOE_CPU_EXPS=1` keeps the experts in the memory-mapped file on the CPU, so tracing needs no
free memory for them, with the dense part on Vulkan0 (llama.cpp with `qwen4exp`, e71b805).
Its layers mix four 2560-wide residual streams (hyper-connections), so the gate-ahead
predictor does not apply yet, and these traces carry routes only.

| decode hit rate | cache / layer (GiB) | lru | g-lru | g-lfu | slru | opt |
|---|---|---|---|---|---|---|
| code | 128 (18.4) | 85.9 | 87.5 | 87.1 | 84.2 | 93.3 |
| | 192 (27.5) | 91.7 | 92.8 | 92.5 | 89.4 | 96.9 |
| | 256 (36.7) | 94.5 | 95.1 | 94.3 | 91.4 | 98.9 |
| | 384 (55.1) | 97.5 | 97.6 | 96.9 | 92.6 | 99.8 |
| chat (6 turns) | 128 | 82.6 | 84.1 | 84.0 | 83.8 | 93.2 |
| | 192 | 88.7 | 89.9 | 92.1 | 90.9 | 96.4 |
| | 256 | 91.9 | 92.3 | 94.9 | 94.3 | 97.4 |
| | 384 | 96.8 | 97.2 | 97.5 | 96.8 | 98.4 |

**MTP costs the drive nothing extra.** A verification step routes the draft window, n + 1 = 4
tokens, and advances by the accepted ones (2.2 assumed, from the measured 40-49 against 25-26
tok/s). Replayed through the shared LRU, the reads per generated token are the same with and
without MTP, within 0.5%, at every cache size, because each token's experts are read once
either way. MTP groups them into steps of 2.2 times as many reads, which keeps more reads in
flight. So MTP lowers the compute per token and leaves the drive term unchanged.

Drive ceilings (shared LRU, 3.7 GB/s), compute 38.5 ms per token without MTP (26 tok/s
resident) and 22 ms with it (45.5 tok/s):

| cache / layer | reads per token, code / chat | no MTP, overlapped | MTP, serial | MTP, overlapped |
|---|---|---|---|---|
| 128 (18.4 GiB) | 193 / 244 MB | 19.2 / 15.2 | 13.5 / 11.4 | 19.2 / 15.2 |
| 192 (27.5 GiB) | 110 / 156 MB | 26.0 / 23.7 | 19.3 / 15.6 | 33.5 / 23.7 |
| 256 (36.7 GiB) | 76 / 119 MB | 26.0 / 26.0 | 23.6 / 18.5 | 45.5 / 31.2 |
| 384 (55.1 GiB) | 37 / 43 MB | 26.0 / 26.0 | 31.3 / 29.8 | 45.5 / 45.5 |

**Measured on the drive** (`1bit moe-cache`, chat and long traces, 192 per layer = 9,216
slots: 27.5 GiB of experts in 34.4 GiB of pinned slots. At the time, every slot was sized for
the largest layer's expert. Size classes (below) now pin 27.0 GiB for the same slots):

| mode | compute per token | decode tok/s | stall per token |
|---|---|---|---|
| no prefetch (reads on demand) | 38.5 ms / 22 ms (MTP) | 7.3-7.7 / 8.3-8.8 (2.4 in one later run) | 91-99 ms (412 ms in that run) |
| oracle: every layer's experts queued when a token starts | 38.5 ms / 22 ms | 10.2-11.2 / 10.7-11.5 | 51-71 ms |
| oracle, compute ~0 (the cache's read path alone) | | 8.9-9.8 | 102-112 ms |

This is far below the ceilings above, for two reasons:
- **The drive was slower than its fresh-box figure.** Random 1 MiB reads, 8 in flight, gave
  3.5 GB/s at one point in the session and 1.2-3.2 GB/s later, with the drive at 53 °C (limit 82)
  and the box's load average at 6-9 from other tenants. Paired runs (raw reads, the cache, raw
  reads again) put the cache's read path at the drive's rate of the moment. The cache is not
  the bottleneck.
- **Reads on demand are latency-bound.** A layer misses 1-2 of its 10 experts (3 reads of
  about 1 MiB each: gate, up, down). They go in flight together, then compute waits. At that
  depth each read pays the drive's latency under load. With the oracle, all of a token's reads
  are in flight together: 1.4-4.8 times faster, from the same bytes.

So the design point for Flash-Next is **queue many layers' reads at once**. That needs
predictions several layers ahead: its gate-ahead predictor, then two- and three-ahead. MTP's
bigger steps help the same way, and a quieter drive helps most.

## Running these measurements safely

A model larger than free memory must never be loaded blind. llama.cpp's CPU backend copies
quantized weights into anonymous memory (repacking), so `n_gpu_layers=0` doesn't stream from
the file. On 2026-09-25 that OOM-killed other services on the box. On Strix Halo, GPU
allocations also escape cgroup memory limits. So every big run goes through
`scripts/mem-guard.sh <floor GB> <cmd>`, which kills the run itself when `MemAvailable` falls
below the floor.

## Next

1. **Streaming in the inference path.** Route llama.cpp's MUL_MAT_ID through `ExpertCache`
   slots, with the gate-ahead prediction computed on the GPU and the cache on the host, and
   measure it against the `1bit moe-cache` numbers above.
2. **A smarter eviction policy.** Belady's bound is 5-16 points above the shared LRU at small
   caches. Candidates: frequency with decay per layer, and hints from the router's scores.
3. **Flash-Next's gate-ahead prediction.** Its layers keep four 2560-wide residual streams
   (hyper-connections), so the predictor needs that model's mixing step before the router.
4. **The quants not on the box** (UD-Q2/Q3/Q4_K_XL): their decode speed, once there is room
   on the drive.
