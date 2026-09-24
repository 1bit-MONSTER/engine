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
tags: announcement, engine

# 1bit.MONSTER is now 1bit engine

1bit.MONSTER grew fast: an inference engine, a model zoo, benchmarks, a store, a voice assistant and a lot of experiments, all in one repository with all of its history. The engine that came out of it is now its own project, **1bit engine**, and this site is its home.

## What changed

**A clean repository.** The engine lives in [1bit-MONSTER/engine](https://github.com/1bit-MONSTER/engine). It holds verified code only, without the development history. Each piece is ported over once it passes its end-to-end test on Strix Halo, and the [porting map](../docs/PORTING.md) shows what has moved and what is next.

**Apache-2.0.** The old repository was GPL-3.0. The engine is Apache-2.0, with a copyright and license notice on every file, enforced in CI, and every third-party project it builds on credited in its NOTICE.

**Inside Lemonade, not around it.** The old engine vendored its own copy of Lemonade. Now [Lemonade](../docs/lemonade.md) is the server you talk to (its catalog, downloads, router and UI), and it runs the engine as one of its backends, through `1bit serve`, the same way it runs `llama-server`.

**Upstream, pinned and kept current.** The XDNA driver and XRT, AMD's HRX pair, llama.cpp for Vulkan, ZINC, the Linux kernel, Hugging Face tokenizers and the Laya router are each pinned to upstream, and the pins are bumped automatically, so new models and fixes arrive soon after upstream ships them.

## What it runs on

One `1bit serve` drives:

- the [XDNA 2 NPU](../docs/npu.md), on full ELFs generated in C++;
- [HRX and Vulkan](../docs/hrx.md) on the Radeon iGPU, one llama.cpp build with both;
- [ZINC](../docs/zinc.md), which also reaches NVIDIA GPUs through CUDA;
- [MLX](../docs/apple.md) on Apple Silicon.

Measured numbers live on the [wiki](https://github.com/1bit-MONSTER/engine/wiki). [The first week of 1bit engine](2026-09-24-first-week-milestones.md) lists what the first three days produced.

## The old site

The 1bit.MONSTER site and its posts stay online in the [1bit.MONSTER archive](https://1bit-monster.github.io/1bit-MONSTER/), and old links land on a page that points at their archived copy. The blog page here lists the old posts worth keeping. 1bit JARVIS, the voice assistant, is not part of the engine for now; it will come back later as its own piece.
