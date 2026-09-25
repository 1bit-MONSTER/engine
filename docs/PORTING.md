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
# Porting map

This repository is the working 1bit-MONSTER engine, ported without its history.
Each step below is one PR (or a short series) that builds and runs on Strix Halo before the next one starts.

| Step | Component | Source in 1bit-MONSTER | Done when |
|---|---|---|---|
| 1 | **The engine runs inside Lemonade.** Lemonade runs `1bit serve` as a backend: one model per process behind an OpenAI-compatible API | `tools/unified_server.cpp` | **landed** ([docs/serve.md](serve.md)): NPU, Vulkan, HRX and ZINC pass `serve_e2e` on Strix Halo; `smoke_serve` runs in CI. The engine first embedded Lemonade (v11.9.0 plus local recipes); after geramyL's review it is embedded *into* Lemonade instead: the vendored copy is removed, and the Lemonade recipe that runs `1bit serve` (`onebit`) lives in our fork [1bit-MONSTER/lemonade](https://github.com/1bit-MONSTER/lemonade) ([docs/lemonade.md](lemonade.md)) |
| 2 | **HRX with Vulkan.** llama.cpp with `GGML_HRX=ON` and `GGML_VULKAN=ON` in one build, on AMD's tested pair | `1bit-MONSTER/llama.cpp` `1bit/hrx-vulkan-patched` (AMD's `hrx-graph-develop-v2` plus our commits) + `ROCm/hrx-system`, pinned from `ROCm/ggml-staging-automation` | **landed** ([docs/hrx.md](hrx.md)): `1bit serve` runs the same checkpoint on `Vulkan0` and `HRX0`; kept current by `bump-hrx.yml` |
| 3 | **NPU engine.** Full ELFs only, and the open 16-tile layer kernel built from source | `engine/npu` (`npu_engine_universal.cpp`, `I8Ctx::init_elf`); ELF dispatch table on `backup/iso-build-elf-native-2026-09-22`; kernel on `bench/fastlane-16tile-corrections-2026-09-22` | **3a–3c landed** ([docs/npu.md](npu.md)). 3a (#8): full ELFs generated in C++, reproducing all 5,153 captured contexts. 3b/3c (#9): the lane runtime (logits bit-identical to the reference lane, 24/24 steps, 11.0 ms/token), tokenizer, `1bit unified`, and Qwen3-0.6B served on the NPU (now through `1bit serve`). The XDNA driver and XRT are pinned upstream and built privately (#12). **Open:** the layer kernel and lm-head artifacts are not yet built from source (npu.md, "Open") |
| 3x | **35B MoE on the NPU (experimental, closed source).** Qwen3.6-35B-A3B on the NPU | the private `1bit-MONSTER/npu-kernels` repository | **landed as a private add-on** ([docs/npu.md](npu.md#private-routes)): built into `1bit` with `-DONEBIT_NPU_PRIVATE`; parity against the fp64 reference passes (3 positions, argmax 846 / 198 / 3710), 16.3-16.5 tok/s decode, and `1bit serve --device npu` answers "The capital of France is Paris." Without the add-on, `serve` says the route is not part of the build |
| + | **Linux kernel.** The kernel that provides `amdxdna` and `amdgpu`, pinned to upstream | `torvalds/linux` release tags; config from the Strix Halo kernel of 2026-09-23 | **pinned** ([docs/kernel.md](kernel.md)): v7.3-rc4 builds into Debian packages with `amdxdna` in-tree; kept current by `bump-linux.yml`. Installing it on Strix Halo is a separate, deliberate step |
| 4 | **Laya router.** A non-autoregressive scorer that picks where each request runs | `src/laya_scorer.cpp`, `include/laya_scorer.h` on `backup/laya-and-results-2026-09-22`; model at `~/models/laya` | **landed, opt-in** ([docs/laya.md](laya.md)): source `NandhaKishorM/laya` + the three HF checkpoints pinned, hash-verified fetch, `bump-laya.yml` (#18); the C++ scorer matches the Python reference on the root and `typed-decisions/` checkpoints (max logit diff 8.6e-6, same argmax) and `1bit serve --device auto --laya-model <dir>` routes each request (#90). Load 2.8 s once, then 0.38 s per decision (was 8.75 s, #91). Next: `multilingual/` (mmBERT-base) |
| 5 | **Every HF model, kept current.** The architecture registry and the daily HF census that finds new architectures and ranks the unmapped ones by model count (1bit-MONSTER's registry mapped 2,030 HF arch strings to its own kernels; here the backends' pinned code decides) | `src/model_registry*.cpp`, `Testing/census_*.py` and `.json`, `.github/workflows/census-{watch,sweep,autopr}.yml` | **landed** ([docs/registry.md](registry.md)): `registry/architectures.json` generated from the pinned llama.cpp, HRX fork and ZINC (265 HF architectures); `census.yml` sweeps HF daily. First sweep: 415,414 text-generation models, mapped 93.28%, checked 64.18% of those with an architecture (reported separately). All seven checked GGUF architectures pass on HRX since the llama.cpp pin `96f6b89` (#95) |
| 5z | **ZAYA1 (Zyphra) on Vulkan.** An architecture upstream llama.cpp lacks, in our llama.cpp; `1bit serve --device vulkan` routes it there | `1bit-MONSTER/llama.cpp` `1bit/hrx2-archive` (`src/models/zaya.cpp`) and 1bit-MONSTER's `tools/convert_zaya_safetensors_to_gguf.py` | **landed** ([docs/vulkan.md](vulkan.md#zaya1-zyphra-from-our-llamacpp)): ported to the pinned fork with a converter; matches transformers in FP32 at 95/96 teacher-forced positions; Q4_K_M decodes at 93.5 tok/s (74.8 on the old ggml-hrx2 build) and passes `serve_e2e` |
| 6 | **ZINC (NVIDIA and more).** Upstream `zolotukhin/zinc`, a Zig GGUF engine with Vulkan, ROCm, CUDA and Metal backends; its CUDA backend reaches NVIDIA GPUs (Ada `sm_89`, Blackwell `sm_120`) | not in 1bit-MONSTER; pinned from upstream `main` | **pinned** ([docs/zinc.md](zinc.md)): `scripts/build-zinc.sh` builds it privately; the Vulkan build gives 12095 (" Paris") at 295 tok/s on Strix Halo; the CUDA build answers " Paris." at 167–173 tok/s on an RTX 5090 (Qwen3.5-9B); kept current by `bump-zinc.yml`. `1bit serve --device zinc` runs it (`-DONEBIT_ZINC=ON`, e2e passes) |

## How the pieces fit

```
Lemonade (the host users talk to)
  └─ runs, as its onebit backend ─> 1bit serve -m <model> --port <p>   one OpenAI-compatible API
                   ├─ NPU model directory ──> NPU fast lane (full ELFs), in process
                   ├─ .gguf --device vulkan|hrx ──> this build's llama.cpp (ggml-vulkan + ggml-hrx)
                   ├─ .gguf --device zinc ──> this build's ZINC (Vulkan, ROCm, CUDA)
                   ├─ HF id --device mlx ──> lemon-mlx-engine (Apple Silicon)
                   ├─ Laya ──> picks the device per request (step 4, opt-in)
                   └─ model registry ──> every HF architecture, refreshed by the daily census (step 5)
```

## Known facts carried over

- **Vulkan is not inside HRX.** One build compiles both ggml backends.
  - (Previous build, ggml-hrx2.) On zaya1-8b, `Vulkan0` decoded at 74.8 tok/s and `HRX20` at 18.4, because
    HRX2 fell back to the CPU for 2,088 ops.
  - Q4NX loaded only on `HRX20`, the ggml-hrx2 backend AMD dropped in June. Since the
    move to AMD's live ggml-hrx (docs/hrx.md) it is not served; the Q4NX kernels are
    kept on `1bit-MONSTER/llama.cpp` `1bit/hrx2-archive` for a port.
- **NPU concurrency:**
  - There is one device, and each engine instance uses 4 hardware contexts.
  - Throughput peaks at about 4 concurrent instances.
- **Where the NPU model containers live:** they are currently in `~/.config/flm/models/*-NPU2` on the dev box.
