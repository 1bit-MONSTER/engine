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
# HRX + Vulkan

One llama.cpp build with two GPU backends, ggml-hrx2 (AMD's HRX runtime) and
ggml-vulkan. Its single `llama-server` exposes both devices on Strix Halo:

```
Vulkan0: AMD Radeon 8060S Graphics (RADV STRIX_HALO)
HRX20:   AMD Radeon 8060S Graphics (Node 1) (gfx1151)
```

`1bit lemonade` uses this binary for two Lemonade recipes:

| Recipe | Device | Serves |
|---|---|---|
| `llamacpp-hrx` | `HRX20` | HRX-only formats (Q4NX) and the HRX catalog entries |
| `llamacpp` (Vulkan backend) | `Vulkan0` | standard GGUF quants, where Vulkan is the faster device |

Lemonade config keys that a user has set are never overridden. Only its
defaults (`builtin`, `auto`, unset) are replaced.

## Build

```bash
git submodule update --init --depth 1 third_party/hrx third_party/hrx-system third_party/llama.cpp
cmake -B build -G Ninja -DONEBIT_HRX=ON          # needs TheRock at /opt/rocm-therock (ONEBIT_HRX_TOOLCHAIN)
cmake --build build --target onebit
```

A fresh clone builds in about 4 minutes on Strix Halo: the submodules take 28 s,
and the whole build takes 217 s.

## Pinned sources

| Submodule | Commit | Used for |
|---|---|---|
| `third_party/hrx` ([ROCm/hrx](https://github.com/ROCm/hrx)) | `0bc22fb` | libhrx, libloomc and the Loom AMDGPU binding, plus `patches/hrx/0001`–`0003` |
| `third_party/hrx-system` ([ROCm/hrx-system](https://github.com/ROCm/hrx-system)) | `6743075f` | `loom-link`, plus `patches/hrx/0002` |
| `third_party/llama.cpp` (fork, branch `1bit-engine/hrx-vulkan`) | `3b33c8a9` | ggml-hrx2 kernels and catalog, and the llama.cpp + Vulkan build |

The patches:

- **`0001`, `0002`:** switch off install/export rules that fail to configure,
  and let the AMDGPU binding link against loomc. Build plumbing only.
- **`0003`:** a runtime fix. gfx1151 rejects the `PM4_EMULATION` agent probe,
  which is now treated as native AQL execution instead of an error. It is a
  candidate for upstreaming to ROCm/hrx.

**Why two HRX repositories:** the fork's kernel catalog is linked with
`loom-link --mode=selective`. ROCm/hrx-system removed that mode in `87e3715963`,
so `loom-link` is built from the commit just before it.

## Verified

- **Kernel artifacts are reproducible.** On the same source path, all 48 kernel
  artifacts linked by this build's `loom-link` are byte-identical to those of the
  hand-assembled build the project had been running.
- **The HRX prefix matches the old one.** It has the same headers and libraries as
  the hand-assembled `install-new` it replaces, which was previously undocumented.
- **Speed matches the hand-assembled build** (llama-bench, pp128/tg32, 2 runs):

  | Device, model | hand-assembled | this build |
  |---|---|---|
  | Vulkan0, Qwen3-1.7B Q4_K_M | tg 151.4 | tg 148.7 |
  | HRX20, zaya1-8b Q4NX | tg 17.8 | tg 18.3 |

  HRX prefill measured 190 vs 167 tok/s on a shared box; it has not been re-measured yet.
- **Lemonade runs both devices end to end.** `tests/hrx_lemonade_e2e.sh` (a
  `ctest` test in ONEBIT_HRX builds) pulls `unsloth/Qwen3-0.6B-GGUF:Q4_0` through
  both recipes and chats with greedy decoding. It checks the answer and which
  binary and device served it:

  | Recipe | Device | Answer | Decode |
  |---|---|---|---|
  | `llamacpp` | Vulkan0 | Paris | 117.3 tok/s |
  | `llamacpp-hrx` | HRX20 | Paris | 33.5 tok/s |

  The HRX catalog's `hrx_serve: NO (GET_ROWS)` note describes AMD's hrx-b66
  bundle. This build serves the entry.

## Not yet

- **zaya1-8b Q4NX through Lemonade.** It was measured with llama-bench only.
  Lemonade has no catalog entry for the local file, so it arrives with the
  engine's model registry (docs/PORTING.md step 5).
- **A relocatable package.** The HRX shared libraries are used from the build
  tree through their RPATHs.
- **CI.** CI has no AMD GPU and no TheRock, so `ONEBIT_HRX` is exercised on Strix Halo only.
