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
| `third_party/llama.cpp` | [AMD-Ecosystem/llama.cpp](https://github.com/AMD-Ecosystem/llama.cpp), branch `hrx-graph-develop-v2` | AMD's ggml-hrx on ggml-org llama.cpp, unchanged |
| `third_party/hrx-system` | [ROCm/hrx-system](https://github.com/ROCm/hrx-system) | libhrx, loomc and the Loom tools |

- **Where the pair comes from.** The two pins are the pair AMD's integration repo,
  [ROCm/ggml-staging-automation](https://github.com/ROCm/ggml-staging-automation),
  builds and tests together.
- **How it stays current.** `.github/workflows/bump-hrx.yml` runs daily. When AMD
  moves its pair, it opens a PR here moving both submodules.
- **The token it needs.** The secret `HRX_BUMP_TOKEN` is a fine-grained token with
  Contents and Pull requests read/write on `1bit-MONSTER/engine`.
- **Validation.** CI builds that PR without HRX, so run the checks below on Strix
  Halo before merging.

## Private GPU build

Our own GPU kernel work (fixes to ggml-hrx, the HRX prefill split below, and ROCmI4
on Vulkan for the lean option) is closed source. It lives in the private
`1bit-MONSTER/gpu-kernels` repository, not here. The engine keeps only the hook:

- `-DONEBIT_GPU_PRIVATE=<gpu-kernels checkout>` (branch `addons/gpu`, its submodules
  initialised) makes `-DONEBIT_HRX=ON` and `-DONEBIT_LEAN=ON` build that checkout's
  llama.cpp, hrx-system and ROCmFPX trees instead of the public pins above and in
  [lean.md](lean.md). The checkout's `addons/gpu/gpu.cmake` names the three trees.
- Without it, `1bit serve --prefill-device hrx` exits with "--prefill-device hrx is not
  part of this build; build with -DONEBIT_GPU_PRIVATE=<gpu-kernels checkout>", and the
  `serve_e2e_vulkan_prefill_hrx` test is not added.

What the public build (AMD's pair as is) and the private build do on `HRX0`:

| | public build | private build |
|---|---|---|
| Qwen3-0.6B Q4_K_M, perplexity (8 x 512 wikitext) | 22.52 (Vulkan: 22.53) | 22.52, same speed (about 21700 / 324 tok/s pp512 / tg128) |
| Prompts of about 400 to 1,700 tokens (Qwen3-0.6B) | garbled answers: decoding goes wrong once the KV cache holds more than 256 tokens | answers match Vulkan's at every length measured (404 to 2,844 tokens) |
| MoE models with more than 128 experts (Qwen3.6-35B-A3B Q8_0) | loads, but every request fails with a compute error (`serve_e2e`, 2026-09-25) | pp512 909, tg128 35.3 tok/s, KLD 0.0046 against Vulkan; `serve_e2e` passes |
| Qwen3-Coder-30B-A3B Q4_K_M | not measured | pp512 2040, tg128 89.4 tok/s |
| Unsloth Dynamic files (UD-Q4_K_XL, UD-Q2_K_XL, UD-IQ2_M) | fail | correct (perplexity within 0.2 of Vulkan's), with their IQ4_XS and sub-4-bit layers on the CPU, so slow |
| `test-backend-ops -b HRX0` | about 1150 fail | 790/790 |
| `--prefill-device hrx` | not available | available (below) |

Use `--device vulkan` (the default for GGUF) for anything outside the first row on a
public build.

### Known issues on `HRX0` (both builds)

- **Several sequences per batch fail.** `llama-perplexity` with `n_seq` > 1 stops on an
  unsupported 3-D MUL_MAT; use `-b 512`.
- **`-fa off` fails.** A SET_ROWS into the non-flash-attention V cache is rejected.

The prefill split below is not affected by either: HRX0 only prefills whole ubatches
there, and decoding is Vulkan's.

### Prefill on HRX, decode on Vulkan (private build)

HRX prefills faster than Vulkan and Vulkan decodes faster, on the same GPU.
`1bit serve --device vulkan --prefill-device hrx` uses both: the HRX build's
llama-server decodes on `Vulkan0`, and a second copy of the model on `HRX0` runs
the prompt prefix. The two contexts share one KV cache with no copy.

- **What runs on HRX:** the prompt prefix in whole ubatches, from
  `--prefill-min-tokens` tokens (default 1024; below that the split does not pay).
  The rest, and all decoding, run on Vulkan.
- **Needs:** flash attention on both sides (`serve` passes `-fa on`), no LoRA, no
  multimodal. The weights are loaded twice (once per device).

Measured on Strix Halo, llama-server on `Vulkan0`, prefill / whole request:

| Model | Prompt | Vulkan alone | HRX prefill + Vulkan |
|---|---|---|---|
| Qwen2.5-7B Q4_K_M | 2048 | 1541 / 4314 ms | **1078** / 3890 ms (-10%) |
| Qwen2.5-7B Q4_K_M | 8192 | 7347 / 10376 ms | **4646** / 7668 ms (**-26%**) |
| Qwen3-0.6B Q4_K_M | 8192 | 1289 / 2267 ms | **1089** / 2032 ms (-10%) |

Decode on the shared KV is within 3% of Vulkan with its own KV. Against Vulkan
alone, teacher-forced over 64 tokens, the split's mean KL is 0.0002-0.0006 nats
(max 0.007) and the top token agrees on 64-65 of 65 positions; Q4_K_M against
BF16 is 0.063.

The previous build (AMD's abandoned ggml-hrx2 with our Q4NX kernels and the zaya
architecture) is kept in the private archive for a later port to ggml-hrx.

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

The private build ("Private GPU build" above) needs no engine submodules for HRX:

```bash
git clone -b addons/gpu https://github.com/1bit-MONSTER/gpu-kernels.git   # private
git -C gpu-kernels submodule update --init --depth 1
cmake -B build -G Ninja -DONEBIT_HRX=ON -DONEBIT_GPU_PRIVATE=$PWD/gpu-kernels
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

## Verified (2026-09-25)

`tests/serve_e2e.sh` on Strix Halo, one build with `-DONEBIT_HRX=ON -DONEBIT_LEAN=ON` per row:

| Build | Qwen3-0.6B Q4_K_M `hrx` | `vulkan` | `vulkan --prefill-device hrx` | Qwen3.6-35B-A3B Q8_0 `hrx` |
|---|---|---|---|---|
| public (llama.cpp `f1a0aca`, hrx-system `a351789`) | PASS | PASS | refused with the message above | FAIL (compute error) |
| `-DONEBIT_GPU_PRIVATE` | PASS | PASS | PASS | PASS |

Earlier (2026-09-23, llama.cpp `f1a0aca`, hrx-system `51b1739`): `ctest` passed, re-run
after the first daily bump (#13). At the time the check was `tests/hrx_lemonade_e2e.sh`,
through the engine's former embedded Lemonade: `unsloth/Qwen3-0.6B-GGUF:Q4_0` answered
"Paris" on `Vulkan0` and on `HRX0`.

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

- **Fast sub-4-bit and IQ4 kernels on HRX.** On the private build the UD files are
  correct on `HRX0` but slow, since IQ4_XS, IQ2 and IQ3_S matmuls fall back to the
  CPU. AMD's IQ4_XS / IQ4_NL kernels need fixing upstream.
- **Q4NX on HRX.** Our Q4NX kernels lived in ggml-hrx2 and are not in ggml-hrx.
  They are kept in the private archive until they are ported.
- **First-request cost.** Lemonade's telemetry for the first short HRX chat shows
  18 tok/s against llama-bench's 303. It is probably first-use kernel
  compilation, not steady state, but this is not yet separated.
- **CI.** CI has no AMD GPU and no TheRock, so `ONEBIT_HRX` is exercised on Strix
  Halo only.
