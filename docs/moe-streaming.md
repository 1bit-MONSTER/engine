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

This page holds the measurements the design rests on. The engine code comes next.

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

These are ceilings from the drive alone, before compute, and from two prompts. LRU per
layer is the simplest policy. The traces are kept, so better policies (frequency, a budget
shared across layers, prefetching the next layer's experts while this one runs) can be
compared on the same data.

## Running these measurements safely

A model larger than free memory must never be loaded blind. llama.cpp's CPU backend copies
quantized weights into anonymous memory (repacking), so `n_gpu_layers=0` doesn't stream from
the file. On 2026-09-25 that OOM-killed other services on the box. On Strix Halo, GPU
allocations also escape cgroup memory limits. So every big run goes through
`scripts/mem-guard.sh <floor GB> <cmd>`, which kills the run itself when `MemAvailable` falls
below the floor.

## Next

1. More traces: long context, chat, other MoE families. Then cache policies, replayed on
   those traces.
2. The expert cache in the engine: pinned RAM slots, `O_DIRECT` reads with several in flight,
   and the next layer's experts prefetched while this layer computes.
3. The three-way split. Dense on HRX or Vulkan, experts on the NPU or the GPU, measured
   both ways.
4. The sweet spot per model: quantization × split × cache size × MTP, on tok/s and quality.
