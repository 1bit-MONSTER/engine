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

One llama.cpp build with two GPU backends, ggml-hrx (AMD's HRX runtime) and
ggml-vulkan. Its single `llama-server` exposes both devices on Strix Halo:

```
HRX0:    AMD Radeon 8060S Graphics (Node 1) (gfx1151)
Vulkan0: AMD Radeon 8060S Graphics (RADV STRIX_HALO)
```

`1bit lemonade` uses this binary for two Lemonade recipes:

| Recipe | Device | Serves |
|---|---|---|
| `llamacpp-hrx` | `HRX0` | the HRX catalog entries |
| `llamacpp` (Vulkan backend) | `Vulkan0` | standard GGUF quants |

Only Lemonade's defaults (`builtin`, `auto`, unset) are replaced. Config keys a
user has set are never overridden.

## Pinned sources, kept current

| Submodule | Source | Pinned to |
|---|---|---|
| `third_party/llama.cpp` | [1bit-MONSTER/llama.cpp](https://github.com/1bit-MONSTER/llama.cpp), branch `1bit/hrx-vulkan` | AMD's `hrx-graph-develop-v2` (ggml-hrx on ggml-org llama.cpp) |
| `third_party/hrx-system` | [ROCm/hrx-system](https://github.com/ROCm/hrx-system) | libhrx, loomc and the Loom tools |

- **Where the pair comes from.** The two pins are the pair AMD's integration repo,
  [ROCm/ggml-staging-automation](https://github.com/ROCm/ggml-staging-automation),
  builds and tests together.
- **How it stays current.** `.github/workflows/bump-hrx.yml` runs daily. When AMD
  moves its pair, it:
  - syncs the fork (`master` to ggml-org, `1bit/hrx-vulkan` to AMD's pin);
  - opens a PR here moving both submodules.
- **The token it needs.** The secret `HRX_BUMP_TOKEN` is a fine-grained token with
  Contents read/write on `1bit-MONSTER/llama.cpp` and `1bit-MONSTER/engine`, and
  Pull requests read/write on `1bit-MONSTER/engine`.
- **Validation.** CI builds that PR without HRX, so run the checks below on Strix
  Halo before merging.

There are no local patches. `1bit-MONSTER/llama.cpp` also holds
`1bit/hrx2-archive`, the previous build (AMD's abandoned ggml-hrx2 plus our Q4NX
kernels and the zaya architecture). It is kept for a later port of that work to
ggml-hrx.

## Build

```bash
git submodule update --init --depth 1 third_party/hrx-system third_party/llama.cpp
cmake -B build -G Ninja -DONEBIT_HRX=ON          # needs TheRock at /opt/rocm-therock (ONEBIT_HRX_TOOLCHAIN)
cmake --build build --target onebit
```

- **One external project.** llama.cpp's ggml-hrx builds hrx-system itself
  (`HRX_SOURCE_DIR`) with TheRock's `amdclang`.
- **`IREE_ROCM_PATH` is not passed.** It would switch hrx-system to "package"
  mode, which needs TheRock's aqlprofile-sdk headers, and our `/opt/rocm-therock`
  does not ship them. Left unset, hrx-system fetches its pinned HSA/AQL headers.
- **The HSA runtime.** HRX dlopens it at run time, and the distro's
  `libhsa-runtime64` rejects the `HSA_AMD_AGENT_INFO_PM4_EMULATION` probe on
  gfx1151, so HRX then registers no device. CMake finds TheRock's
  `libhsa-runtime64.so.1` (`ONEBIT_HRX_LIBHSA`), and `1bit lemonade` passes it to
  llama-server as `IREE_HAL_AMDGPU_LIBHSA_PATH`, unless it is already set. To run
  llama-server or llama-bench by hand, export that variable yourself.

## Verified (2026-09-23, llama.cpp `f1a0aca`, hrx-system `fab1624`)

`tests/hrx_lemonade_e2e.sh` passes. `unsloth/Qwen3-0.6B-GGUF:Q4_0` answers
"Paris" through both recipes, served by this build's llama-server on `Vulkan0`
and on `HRX0`.

llama-bench, Qwen3-0.6B, pp512 / tg128 tok/s, against the previous build (April
llama.cpp with ggml-hrx2 on `HRX20`):

| Device, file | previous build | this build |
|---|---|---|
| Vulkan0, Q4_K_M | 10627 / 306 | **13905 / 338** |
| Vulkan0, UD-Q4_K_XL | 10668 / ~300 | 13505 / 330 |
| HRX, Q4_K_M | 1046 / 68 | **21702 / 303** |
| HRX, UD-Q4_K_XL | 837 / 83 | 18059 / 288 |

HRX prefill is now faster than Vulkan's.

## Not yet

- **Q4NX on HRX.** Our Q4NX kernels lived in ggml-hrx2 and are not in ggml-hrx.
  They are kept on `1bit/hrx2-archive` until they are ported.
- **First-request cost.** Lemonade's telemetry for the first short HRX chat shows
  18 tok/s against llama-bench's 303. It is probably first-use kernel
  compilation, not steady state, but this is not yet separated.
- **CI.** CI has no AMD GPU and no TheRock, so `ONEBIT_HRX` is exercised on Strix
  Halo only.
