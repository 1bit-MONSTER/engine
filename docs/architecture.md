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
# How the engine is built

The 1bit engine runs inside AMD's [Lemonade Server](https://github.com/lemonade-sdk/lemonade) as
its `onebit` backend, pinned to Lemonade's weekly release (`third_party/lemonade`, moved by the
`bump-lemonade` workflow; [lemonade.md](lemonade.md)). Lemonade stays the server you talk to: its
catalog, downloads, router and UI. For the models it hands to 1bit, it runs the engine as one of
its backends, behind an OpenAI-compatible API. This page is the technical map of what runs where;
the README is the short version.

## Routes

- **HRX on the Radeon iGPU** ([hrx.md](hrx.md)): AMD's ggml-hrx with its Loom kernels and our HIP
  kernels, the engine's one GPU route. Every ggml weight type runs on HRX, including NVFP4, Q2_0,
  MXFP4 and the ternary TQ, PTQ and PQ formats. For a GGUF, Lemonade starts the engine's
  `llama-server` directly, one HTTP hop
  ([#475](https://github.com/1bit-MONSTER/engine/pull/475),
  [serve.md](serve.md#one-hop-from-lemonade---print-backend)). Placement is strict by default: the
  whole graph runs on HRX, or the load stops with the node that would not
  ([#482](https://github.com/1bit-MONSTER/engine/pull/482),
  [serve.md](serve.md#off-hrx-placement-strict-by-default)). `--device cpu` runs the same build on
  the CPU.
- **The NPU engine** on the XDNA 2 NPU, on our own NPU runtime with full-ELF kernel sets:
  correctness first, speed next ([npu.md](npu.md)).
- **Hybrid**: the prompt on the NPU engine, the tokens on HRX. `1bit serve --device hybrid` hands
  the prompt's KV cache from a prefill provider to the HRX llama-server ([hybrid.md](hybrid.md)).
  The NPU and the GPU run at the same time; the measured cost of a busy GPU to the NPU's weight
  stream is recorded there.
- **ONNX Runtime**, linked in-process behind `-DONEBIT_ORT`, with our own execution provider that
  claims MatMul and runs it on the NPU runtime ([onnx.md](onnx.md), [onnx-pack.md](onnx-pack.md)).
- **DwarfStar**, for DeepSeek V4 Flash, GLM 5.x and Qwen3.8-Flash-Next in its own GGUFs, with SSD
  expert streaming ([dwarfstar.md](dwarfstar.md)).
- **Laya**, opt-in, which picks where each request runs ([laya.md](laya.md)), and **the Project
  Manager**, ZAYA1-8B delegating to expert models ([pm.md](pm.md)).
- **Every Hugging Face model architecture**, mapped to a backend by a registry and kept current by
  a daily census ([registry.md](registry.md)).

Vulkan, ROCm, CUDA, ZINC, ComfyUI, Metal/MLX and Windows were cut from the engine's core
([#306](https://github.com/1bit-MONSTER/engine/pull/306)) and are not routes of the engine.
Lemonade serves them with its own backends and builds ([lemonade.md](lemonade.md)).

## Checked on the hardware

Every HRX pin change and every PR labelled `hw-gate` passes the hardware gate on Strix Halo before
merge: it builds our HIP kernels into the server as the release does, runs kernel tests, known
answers, KLD against the release pin and long-prompt needles, and checks that every graph of every
gate model is on the GPU ([#478](https://github.com/1bit-MONSTER/engine/pull/478),
[#484](https://github.com/1bit-MONSTER/engine/pull/484), [hw-gate.md](hw-gate.md)). The runtime
keeps one memory budget for every buffer, and after a GPU fault the server refuses new work and
exits cleanly, so Lemonade can restart the model
([#486](https://github.com/1bit-MONSTER/engine/pull/486)). Serve numbers in these docs come from
`tools/bench.py`, which A/B-measures configurations in the same run ([bench.md](bench.md)).

## The kernel

The box's Linux kernel is ours too: `v7.3-rc4` plus the IOMMU PerfOpt backports, built from our
config, MOK-signed for Secure Boot, and gated like a pin bump. Moving it to the next upstream tag is
a written step list ([kernel.md](kernel.md)).

## Pinned sources

HRX is pinned to AMD's hrx-system and ggml-hrx with our commits on top, kept current by `bump-hrx`
([hrx.md](hrx.md#pinned-sources-kept-current)); Lemonade, Laya, the tokenizers library and
DwarfStar each have their own bump workflow ([releases.md](releases.md#bumps)). A hand-raised bump
carries the branch name and the derived data the checks expect ([releases.md](releases.md)).

## Direction

HRX with our HIP kernels, plus the NPU engine, embedded into Lemonade rather than Lemonade embedded
into the engine.[^geramyl] HRX met the decode gates in
[RFC #213](https://github.com/1bit-MONSTER/engine/discussions/213), which removed the engine's
Vulkan and ROCm builds. ZAYA1-8B (Zyphra) runs from our llama.cpp on HRX at about 90 tok/s decode
in Q4_K_M, matching transformers. Zyphra's Zamba, Zamba2 and BlackMamba ran only on the removed
Vulkan build and are to be ported to HRX. The working engine is being ported from our private
development repository; see [PORTING.md](PORTING.md). This repository holds the verified code
without the development history.

[^geramyl]: The HRX + Loom + NPU direction, and embedding the engine into Lemonade rather than Lemonade into the engine, were proposed by [geramyL](https://github.com/Geramy), a moderator on Lemonade's Discord. Thank you.
