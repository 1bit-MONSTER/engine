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

# 1bit.MONSTER is now 1bit engine

1bit.MONSTER grew fast: an inference engine, a model zoo, benchmarks, a store and a lot of
experiments, all in one repository with all of its history. The engine that came out of it is
now its own project, [1bit engine](../README.md), and this site is its home.

The engine is a clean repository. It holds verified code only, without the development history,
and it is Apache-2.0. Each piece is ported over once it passes its end-to-end test on Strix Halo;
the [porting map](../docs/PORTING.md) shows what has moved and what is next.

What it is: one engine behind an OpenAI-compatible API, running inside
[Lemonade](../docs/lemonade.md) as one of its backends. The same `1bit serve` drives the
[XDNA 2 NPU](../docs/npu.md), [HRX and Vulkan](../docs/hrx.md) on the Radeon iGPU,
[ZINC](../docs/zinc.md) for NVIDIA and Apple GPUs, and [MLX](../docs/apple.md) on Apple Silicon.
Measured numbers live on the [wiki](https://github.com/1bit-MONSTER/engine/wiki).

The old 1bit.MONSTER site and its posts stay online in the
[1bit.MONSTER archive](https://1bit-monster.github.io/1bit-MONSTER/). Old links to it land on a
page that points at their archived copy.

New posts will show up here, and feed readers can follow the blog's Atom feed.
