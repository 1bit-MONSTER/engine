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
| 2 | **HRX with Vulkan.** llama.cpp with `GGML_HRX=ON` and `GGML_VULKAN=ON` in one build, on AMD's tested pair | `1bit-MONSTER/llama.cpp` `1bit/hrx-vulkan` (AMD's `hrx-graph-develop-v2`) + `ROCm/hrx-system`, pinned from `ROCm/ggml-staging-automation` | **landed** ([docs/hrx.md](hrx.md)): Lemonade serves the same checkpoint on `Vulkan0` and `HRX0`; kept current by `bump-hrx.yml` |
| 3 | **NPU engine.** Full ELFs only, and the open 16-tile layer kernel built from source | `engine/npu` (`npu_engine_universal.cpp`, `I8Ctx::init_elf`); ELF dispatch table on `backup/iso-build-elf-native-2026-09-22`; kernel on `bench/fastlane-16tile-corrections-2026-09-22` | **3a–3c landed** ([docs/npu.md](npu.md)). 3a (#8): full ELFs generated in C++, reproducing all 5,153 captured contexts. 3b/3c (#9): the lane runtime (logits bit-identical to the reference lane, 24/24 steps, 11.0 ms/token), tokenizer, `1bit unified`, and Lemonade serving Qwen3-0.6B on the NPU through `onebit`. The XDNA driver and XRT are pinned upstream and built privately (#12). **Open:** the layer kernel and lm-head artifacts are not yet built from source (npu.md, "Open") |
| + | **Linux kernel.** The kernel that provides `amdxdna` and `amdgpu`, pinned to upstream | `torvalds/linux` release tags; config from the Strix Halo kernel of 2026-09-23 | **pinned** ([docs/kernel.md](kernel.md)): v7.3-rc4 builds into Debian packages with `amdxdna` in-tree; kept current by `bump-linux.yml`. Installing it on Strix Halo is a separate, deliberate step |
| 4 | **Laya router.** A non-autoregressive scorer that picks where each request runs | `src/laya_scorer.cpp`, `include/laya_scorer.h` on `backup/laya-and-results-2026-09-22`; model at `~/models/laya` | matches its Python reference; routes requests |
| 5 | **Every HF model, kept current.** The architecture registry (569 tokens mapping 2,030 HF arch strings) and the daily HF census that finds new architectures and proposes mappings | `src/model_registry*.cpp`, `Testing/census_*.py` and `.json`, `.github/workflows/census-{watch,sweep,autopr}.yml` | the census runs daily in CI; docs report *mapped* and *run and checked* counts separately |
| 6 | **ZINC (NVIDIA and more).** Upstream `zolotukhin/zinc`, a Zig GGUF engine with Vulkan, ROCm, CUDA and Metal backends; its CUDA backend reaches NVIDIA GPUs (Ada `sm_89`, Blackwell `sm_120`) | not in 1bit-MONSTER; pinned from upstream `main` | **pinned** ([docs/zinc.md](zinc.md)): `scripts/build-zinc.sh` builds it privately; the Vulkan build gives 12095 (" Paris") at 295 tok/s on Strix Halo; the CUDA build answers " Paris." at 167–173 tok/s on an RTX 5090 (Qwen3.5-9B); kept current by `bump-zinc.yml`. `1bit lemonade` serves it as the `zinc` recipe (`-DONEBIT_ZINC=ON`, e2e passes) |

## How the pieces fit

```
1bit lemonade  ──  Lemonade server core (in-process)
                     ├─ llamacpp-hrx recipe ──> HRX build: llama.cpp + ggml-hrx + ggml-vulkan
                     │                            devices HRX0 and Vulkan0
                     ├─ onebit recipe ────────> 1bit unified -m <artifact>  ──> NPU engine (full ELFs)
                     ├─ Laya ─────────────────> scores each request: which model and device
                     └─ model registry ───────> every HF architecture, refreshed by the daily census
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
