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
# Models measured on a Ryzen AI MAX+ 395

The models the engine is checked against, with the number each one measured on the Strix Halo box
at the current pins. One row is one model file on one route; the standard each number was taken in
is in [bench.md](bench.md), and the routes are described in [hrx.md](hrx.md) and [npu.md](npu.md).
Every ggml weight type runs on HRX, including NVFP4, Q2_0, MXFP4 and the ternary TQ, PTQ and PQ
formats.

| model | file, route | measured |
|---|---|---|
| Qwen3.8-27B | UD-Q4_K_XL, HRX | decode 36.0 tok/s with MTP on code prompts; 56.6 tok/s with the DFlash2 drafter |
| Qwen3.8-Flash-Next (MoE) | HRX | prefill 287 tok/s, decode 24.0 tok/s |
| Zyphra ZAYA1-8B | Q4_K_M, HRX | decode 157 tok/s |
| PrismML Ternary-Bonsai-2-27B | PQ2_0, 6.7 GiB, HRX | decode 43.3 tok/s with MTP |
| Microsoft BitNet b1.58 2B-4T | TQ2_0 ternary GGUF (TQ1_0 runs too), 1.11 GiB, 2.06 bits per weight, HRX | end to end on the Radeon 8060S, no CPU fallback; decode tg128 123.4 tok/s after the TQ2_0 load-time repack onto the packed-ternary decode kernel (101.6 before it, same binary); perplexity 102.42 on HRX vs 102.56 on the CPU, same file |
| Qwen3-0.6B | NPU route | decode 88.6 tok/s in the release gate |
| gpt-oss-20b | MXFP4, HRX | runs on HRX |

The numbers move with the pins; a row changes when its measurement is retaken. The Lemonade
release the engine runs inside is the weekly pin in `third_party/lemonade` ([lemonade.md](lemonade.md)).
