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
| Qwen3.8-27B | UD-Q4_K_XL, HRX | decode 36.0 tok/s with MTP on code prompts; 56.6 tok/s with the DFlash2 drafter ([post](https://1bit.gg/blog-qwen38-27b-dflash2-and-hadamard.html)) |
| Qwen3.8-Flash-Next (MoE) | HRX | prefill 287 tok/s, decode 24.0 tok/s |
| Zyphra ZAYA1-8B | Q4_K_M, HRX | decode 157 tok/s |
| PrismML Ternary-Bonsai-2-27B | PQ2_0, 6.7 GiB, HRX | decode 43.3 tok/s with MTP ([NPU post](https://1bit.gg/blog-ternary-bonsai-on-the-npu.html)) |
| Microsoft BitNet b1.58 2B-4T | TQ2_0 ternary GGUF (TQ1_0 runs too), 1.11 GiB, 2.06 bits per weight, HRX | end to end on the Radeon 8060S, no CPU fallback; decode tg128 123.4 tok/s after the TQ2_0 load-time repack onto the packed-ternary decode kernel (101.6 before it, same binary); perplexity 102.42 on HRX vs 102.56 on the CPU, same file ([post](https://1bit.gg/blog-bitnet-b158-on-hrx.html)) |
| Qwen3-0.6B | NPU route | decode 88.6 tok/s in the release gate |
| gpt-oss-20b | MXFP4, HRX | runs on HRX |

The numbers move with the pins; a row changes when its measurement is retaken. The Lemonade
release the engine runs inside is the weekly pin in `third_party/lemonade` ([lemonade.md](lemonade.md)).

## Our own models on Hugging Face

Some of these files are ours, published under the **`1bit-MONSTER`** organization:
[huggingface.co/1bit-MONSTER](https://huggingface.co/1bit-MONSTER). It carries re-hosted quantizations of
upstream models, our own NPU packs, and the prebuilt kernels a pack needs beside it — `ZR1-1.5B-GGUF`, for
one, holds its Q8 container's kernels in `npu-container-q8/npu/`. The model repositories are **public**: a
download needs no account, no token, and no request to us. (The appliance image is separate: it is a
release artifact and is not published yet — [releases.md](releases.md).)

The organization is a catalogue, not an endorsement list. It holds more repositories than this page has
measurements for, and publishing a file is not a claim that it runs well here. The rows above are that
claim; the org is where the bytes come from.

The **Zyphra** family — Zamba2, Zamba v1 and BlackMamba — has its own page, because those files carry a
conversion property worth knowing before you download one: a Zamba2 GGUF is now the size of its checkpoint
(the shared block and the per-layer MLP adapters are no longer expanded), the two Zamba2 models changed
quantization pick as a result, and a quantizer older than that change drops the adapter off HRX. Sizes, picks
and the `--tensor-type` note are in [zyphra-models.md](zyphra-models.md).
