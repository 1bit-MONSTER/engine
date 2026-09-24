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
tags: serving, batching, vulkan, rocm

# One GPU, many users

A chat with one person and a server for many want different things from the same GPU. We measured Qwen3.8-27B (UD-Q4_K_XL) on Strix Halo from 1 to 24 simultaneous requests, and the engine now grows with the load.

## Batching is the multi-user lever

Decode reads every weight once per step. With one request, one token comes out of that read; with eight batched together, eight do. `1bit serve --parallel N` gives the model N slots that decode together:

| Requests | Vulkan | ROCm |
|---|---|---|
| 1 | 11.8 | 11.6 |
| 4 | 36.9 | 30.2 |
| 8 | **50.9** | 35.6 |
| 16 | 42.9 | **68.0** |

Total tok/s. Vulkan is the better batcher up to 8 requests and then falls back; ROCm keeps climbing to 16, where it serves 5.8x one stream. On the smaller-active Qwen3-Coder-30B-A3B the same pattern holds at a higher level: 228 tok/s on Vulkan at 8 requests, 319 on ROCm at 16.

Two backends on one GPU also help a little on their own: Vulkan and ROCm decoding at the same time made 14.3 tok/s together against 12.2 for either alone (+18%), two Vulkan processes only +4%. They are one GPU on one memory bus, so it is a small gain, not a doubling.

## MTP and batching do not mix

Multi-token prediction gives one user 2.3-3.4x ([the previous post](2026-09-24-speculative-decoding-where-it-pays.md)). Under load it costs: with the MTP head loaded, Vulkan at 4 requests made 24.9 tok/s against 36.9 without it, and turning drafting off per request (draft length 0) did not win it back. Having MTP loaded changes how the server batches.

## Growing with the load

So `1bit serve --adaptive` runs two backends side by side, the model loaded on each, and routes every request by how busy they are:

| Requests | 1 | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| `--adaptive --adaptive-at 8` | 12.0 | 21.9 | 36.3 | **49.9** | 43.5 | 44.7 | **63.5** |

Vulkan takes the first eight requests, batched; ROCm takes the overflow. It matches the best single setup up to 8 and keeps going past it: 63.5 tok/s at 24, where Vulkan alone falls back. Between 12 and 16 the two backends compete for the one GPU and it dips below ROCm alone; that is the next thing to tune.

The first version of the router had a bug worth recording: a burst of requests all checked "is Vulkan free?" before any of them had claimed it, and all piled onto one slot. Choosing a backend and reserving its slot is now one locked step.

## Which to use

- **One person at a time:** `1bit serve --mtp` (35-42 tok/s on Qwen3.8-27B).
- **Many people:** `1bit serve --adaptive --adaptive-at 8`.
- **A fixed crowd of 16 or more:** `--device rocm --parallel 16`.

Details and the full tables are in [1bit serve](../docs/serve.md).
