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

`1bit serve` runs GGUF models on either device with it ([serve.md](serve.md)):

| `--device` | Device | Serves |
|---|---|---|
| `hrx` | `HRX0` | AMD's ggml-hrx |
| `vulkan` (and `auto` for GGUF) | `Vulkan0` | standard GGUF quants, the fastest measured device for them |

## Pinned sources, kept current

| Submodule | Source | Pinned to |
|---|---|---|
| `third_party/llama.cpp` | [1bit-MONSTER/llama.cpp](https://github.com/1bit-MONSTER/llama.cpp), branch `1bit/hrx-vulkan-patched` | AMD's `hrx-graph-develop-v2` (ggml-hrx on ggml-org llama.cpp) plus our commits ("Our patches") |
| `third_party/hrx-system` | [ROCm/hrx-system](https://github.com/ROCm/hrx-system) | libhrx, loomc and the Loom tools |

- **Where the pair comes from.** The two pins are the pair AMD's integration repo,
  [ROCm/ggml-staging-automation](https://github.com/ROCm/ggml-staging-automation),
  builds and tests together.
- **How it stays current.** `.github/workflows/bump-hrx.yml` runs daily. When AMD
  moves its pair, it:
  - syncs the fork (`master` to ggml-org, `1bit/hrx-vulkan` to AMD's pin);
  - rebases our commits onto AMD's pin and moves `1bit/hrx-vulkan-patched` there,
    after tagging the previous tip `patched-<sha>` so every commit the engine has
    pinned stays reachable (a rebase conflict fails the run: rebase by hand);
  - opens a PR here moving both submodules, listing the rebased commits.
- **The token it needs.** The secret `HRX_BUMP_TOKEN` is a fine-grained token with
  Contents read/write on `1bit-MONSTER/llama.cpp` and `1bit-MONSTER/engine`, and
  Pull requests read/write on `1bit-MONSTER/engine`.
- **Validation.** CI builds that PR without HRX, so run the checks below on Strix
  Halo before merging.

### Our patches

`1bit/hrx-vulkan` is AMD's commit unchanged; `1bit/hrx-vulkan-patched` adds:

- **IQ3_XXS matmul on HRX.** A matrix-vector kernel (`kernels/hrx/mul_mat_vec_iq3xxs_f32.loom`)
  and its matcher (`common/dispatch-mul-mat-iq3-xxs.cpp`), ported from the
  `feat/hrx-port-ae91949` work.
- **Honest op claims.** ggml-hrx used to claim every op of a declared type, so
  ggml's scheduler handed it nodes it could not dispatch and the graph failed
  (`unsupported HRX node`) with no CPU fallback. It now claims a node only when
  its dispatcher can execute it, except nodes already in HRX memory (KV cache
  views), which cannot move. Leaf (`NONE`) nodes count as covered. AMD's IQ4_NL
  and IQ4_XS matmul kernels, and GET_ROWS for IQ4_XS and batched IQ3_S, give wrong
  values, so those nodes are left to the CPU.

Measured on Strix Halo (Qwen3-0.6B, perplexity over 8 x 512 wikitext tokens):

| File | Vulkan0 | HRX0 on AMD's commit | HRX0 with our patches |
|---|---|---|---|
| Q4_K_M | 22.53 | 22.52 | 22.52, same speed (21691 / 324 tok/s) |
| UD-Q4_K_XL | 22.43 | fails | **22.41** |
| UD-Q2_K_XL | 36.26 | fails | **36.46** |
| UD-IQ2_M | 55.44 | fails | **55.58** |

`test-backend-ops -b HRX0`: 791 OK, 0 failed (on AMD's commit, about 1150 fail).
Correct is not fast: the UD files run their IQ4_XS and sub-4-bit layers on the CPU
(UD-Q4_K_XL 6303 / 246 tok/s, UD-Q2_K_XL 974 / 76), so they belong on `Vulkan0`.

`1bit-MONSTER/llama.cpp` also holds
`1bit/hrx2-archive`, the previous build (AMD's abandoned ggml-hrx2 plus our Q4NX
kernels and the zaya architecture). It is kept for a later port of that work to
ggml-hrx.

The Vulkan route has its own upstream llama.cpp pin (latest release), so new
architectures do not wait for AMD's pair: see [vulkan.md](vulkan.md). This build
still has a Vulkan backend, which `--device vulkan` uses when `ONEBIT_VULKAN` is
off.

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
  `libhsa-runtime64.so.1` (`ONEBIT_HRX_LIBHSA`), and `1bit serve` passes it to
  llama-server as `IREE_HAL_AMDGPU_LIBHSA_PATH`, unless it is already set. Without a
  build path it searches `/opt/rocm-therock`. To run
  llama-server or llama-bench by hand, export that variable yourself.

## Verified (2026-09-23, llama.cpp `f1a0aca`, hrx-system `51b1739`)

`ctest` passed, re-run after the first daily bump (#13). At the time the check
was `tests/hrx_lemonade_e2e.sh`, through the engine's former embedded Lemonade:
`unsloth/Qwen3-0.6B-GGUF:Q4_0` answered "Paris" on `Vulkan0` and on `HRX0`. The
same build now passes `tests/serve_e2e.sh` on both devices through `1bit serve`.

llama-bench, Qwen3-0.6B, pp512 / tg128 tok/s, against the previous build (April
llama.cpp with ggml-hrx2 on `HRX20`):

| Device, file | previous build | this build |
|---|---|---|
| Vulkan0, Q4_K_M | 10627 / 306 | **14097 / 349** |
| Vulkan0, UD-Q4_K_XL | 10668 / ~300 | 13943 / 324 |
| HRX, Q4_K_M | 1046 / 68 | **21990 / 314** |
| HRX, UD-Q4_K_XL | 837 / 83 | 17906 / 293 |
| Vulkan0, BF16 | n/a / 115 | 4504 / 144 |
| HRX, BF16 | n/a / 40 | 7646 / 101 |

HRX prefill is now faster than Vulkan's.

## Not yet

- **Fast sub-4-bit and IQ4 kernels on HRX.** With our patches the UD files are
  correct on `HRX0` but slow, since IQ4_XS, IQ2 and IQ3_S matmuls fall back to the
  CPU. AMD's IQ4_XS / IQ4_NL kernels need fixing upstream.
- **Q4NX on HRX.** Our Q4NX kernels lived in ggml-hrx2 and are not in ggml-hrx.
  They are kept on `1bit/hrx2-archive` until they are ported.
- **First-request cost.** Lemonade's telemetry for the first short HRX chat shows
  18 tok/s against llama-bench's 303. It is probably first-use kernel
  compilation, not steady state, but this is not yet separated.
- **CI.** CI has no AMD GPU and no TheRock, so `ONEBIT_HRX` is exercised on Strix
  Halo only.
