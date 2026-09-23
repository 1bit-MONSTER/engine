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
| 1 | **Embedded Lemonade.** Lemonade's server core runs inside the `1bit` binary, with the `onebit` backend added | `third_party/lemonade` (v11.9.0 plus local deltas, see its `UPSTREAM.md`); `tools/unified_server.cpp` `run_embedded_lemonade` | **landed** ([docs/lemonade.md](lemonade.md)): `1bit lemonade` serves health, the catalog and system info; a smoke test runs in CI |
| 2 | **HRX with Vulkan.** The AMD-Ecosystem llama.cpp fork built with `GGML_HRX2=ON` and `GGML_VULKAN=ON` in one build | fork build recipe from `fix/zaya-lmhead-evidence` (67503a794); `src/backend_hrx.cpp` | **landed** ([docs/hrx.md](hrx.md)): built from pinned submodules; Lemonade serves the same checkpoint on `Vulkan0` (117 tok/s) and `HRX20` (34 tok/s) |
| 3 | **NPU engine.** Full ELFs only, and the open 16-tile layer kernel built from source | `engine/npu` (`npu_engine_universal.cpp`, `I8Ctx::init_elf`); ELF dispatch table on `backup/iso-build-elf-native-2026-09-22`; kernel on `bench/fastlane-16tile-corrections-2026-09-22` | **3a landed** ([docs/npu.md](npu.md)): full ELFs generated in C++, reproducing all 5,153 captured contexts. **3b/3c**: the lane runtime (logits bit-identical to the reference lane, 24/24 steps, 11.0 ms/token), tokenizer, `1bit unified`, NPU models registered with `onebit`. Done when Lemonade loads Qwen3-0.6B on the NPU through `onebit` |
| 4 | **Laya router.** A non-autoregressive scorer that picks where each request runs | `src/laya_scorer.cpp`, `include/laya_scorer.h` on `backup/laya-and-results-2026-09-22`; model at `~/models/laya` | matches its Python reference; routes requests |
| 5 | **Every HF model, kept current.** The architecture registry (569 tokens mapping 2,030 HF arch strings) and the daily HF census that finds new architectures and proposes mappings | `src/model_registry*.cpp`, `Testing/census_*.py` and `.json`, `.github/workflows/census-{watch,sweep,autopr}.yml` | the census runs daily in CI; docs report *mapped* and *run and checked* counts separately |

## How the pieces fit

```
1bit lemonade  ──  Lemonade server core (in-process)
                     ├─ llamacpp-hrx recipe ──> HRX build: llama.cpp + ggml-hrx + ggml-vulkan
                     │                            devices HRX20 and Vulkan0
                     ├─ onebit recipe ────────> 1bit unified -m <artifact>  ──> NPU engine (full ELFs)
                     ├─ Laya ─────────────────> scores each request: which model and device
                     └─ model registry ───────> every HF architecture, refreshed by the daily census
```

## Known facts carried over

- **Vulkan is not inside HRX.** One build compiles both ggml backends.
  - On zaya1-8b, `Vulkan0` decodes at 74.8 tok/s and `HRX20` at 18.4, because
    HRX2 falls back to the CPU for 2,088 ops.
  - Q4NX loads only on `HRX20`.
- **NPU concurrency:**
  - There is one device, and each engine instance uses 4 hardware contexts.
  - Throughput peaks at about 4 concurrent instances.
- **Where the NPU model containers live:** they are currently in `~/.config/flm/models/*-NPU2` on the dev box.
