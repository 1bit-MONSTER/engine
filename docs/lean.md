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
# The lean option: ROCmFP4 and ROCmI4

`1bit serve --lean` trades accuracy for speed. It runs models in the AMD-focused
formats of [ROCmFPX](https://github.com/charlie12345/ROCmFPX) (MIT), a llama.cpp fork
that upstream llama.cpp cannot read, so the lean route has its own tree:
`third_party/llama.cpp-rocmfpx`, pinned to a commit measured on Strix Halo.

| Command | Format | Device | Best at |
|---|---|---|---|
| `1bit serve -m model-ROCMFP4.gguf --lean` | ROCmFP4 (`Q4_0_ROCMFP4_STRIX_LEAN`) | `Vulkan0` | decode |
| `1bit serve -m model-ROCMI4.gguf --lean --device rocm` | ROCmI4 (`Q4_0_ROCMI4`) | `ROCm0`, W4A4 | prompt processing |

The default route stays Unsloth's UD-Q4_K_XL on upstream llama.cpp ([vulkan.md](vulkan.md)):
it stays far closer to the full model. Use lean when speed matters more than that.

## Measured (Strix Halo, Qwen3.8-27B, 2026-09-24)

Every file quantized from the same BF16; KL divergence and top-token agreement
against BF16 over wikitext-2 (40 x 512 tokens); llama-bench pp512 / tg128 in tok/s;
MTP chat is decode speed with the MTP head on three prompts (code / prose / short).

| File | Size | Mean KLD | Same top token | Vulkan | ROCm | ROCm W4A4 | MTP chat (Vulkan) |
|---|---|---|---|---|---|---|---|
| **UD-Q4_K_XL** (default) | 16.4 GiB | **0.008** | **95.3%** | 353 / 11.9 | 307 / 10.8 | | 22.6 / 20.5 / 19.4 |
| ROCmFP4 STRIX_LEAN | 13.8 GiB | 0.055 | 88.7% | 348 / **14.2** | 411 / 13.7 | | **27.8 / 23.3 / 24.1** |
| ROCmI4 | 13.9 GiB | 0.051 | 89.7% | 6 / 4.1 | 397 / 13.6 | **465** / 13.6 | |

- **ROCmFP4 decodes 19% faster** than UD-Q4_K_XL on Vulkan (14-24% with MTP), from
  a 16% smaller file, at about 7x the KL divergence.
- **ROCmI4 belongs on ROCm.** Vulkan has no kernel for it (6 / 4.1). On ROCm, the
  gfx1151 W4A4 path lifts prompt processing 17% (465 against 397) and leaves decode
  as it is.
- The ROCmFPX files were quantized without an importance matrix; UD-Q4_K_XL was
  made with one. An imatrix would narrow the accuracy gap somewhat.

## Build

```
git submodule update --init --depth 1 third_party/llama.cpp-rocmfpx
cmake -B build -G Ninja -DONEBIT_LEAN=ON                        # Vulkan: ROCmFP4
cmake -B build -G Ninja -DONEBIT_LEAN=ON -DONEBIT_LEAN_ROCM=ON  # also ROCm: ROCmI4
cmake --build build
```

The ROCm build uses TheRock's `amdclang++` from `ONEBIT_LEAN_ROCM_TOOLCHAIN`
(default `/opt/rocm-therock`). If it fails on `__ocml_*` in the distribution's HIP
headers, point that at a TheRock tree whose compiler builds HIP on its own.

## Verified (Strix Halo, ROCmFPX `fb08d7c`, 2026-09-24)

| Test | Result |
|---|---|
| `serve_e2e_lean`: Qwen3.8-27B ROCmFP4 on `Vulkan0` | PASS |
| `tests/serve_e2e.sh ... rocm --lean`: Qwen3.8-27B ROCmI4 on `ROCm0` | PASS; the server logs `ROCmI4 W4A4: enabled` |

The ROCm build there used `-DONEBIT_LEAN_ROCM_TOOLCHAIN=$HOME/therock100`; the
default `/opt/rocm-therock` fails on the `__ocml_*` headers described above.

## Making a lean file

The lean build carries ROCmFPX's `llama-quantize`. Start from the BF16 GGUF:

```
build/lean/llama/bin/llama-quantize model-BF16.gguf model-ROCMFP4.gguf Q4_0_ROCMFP4_STRIX_LEAN
build/lean/llama/bin/llama-quantize model-BF16.gguf model-ROCMI4.gguf  Q4_0_ROCMI4
```

`--imatrix` takes an importance matrix, as in upstream. ROCmFPX has more formats
(2, 3, 6 and 8 bits); only these two are measured here.

## Test

```
cmake -B build -DONEBIT_LEAN=ON -DONEBIT_SERVE_TEST_LEAN_GGUF=$HOME/models/model-ROCMFP4.gguf
ctest --test-dir build -R serve_e2e_lean
```

`bump-rocmfpx.yml` opens a PR moving the pin to ROCmFPX's latest `main` once a week;
rerun the table above on Strix Halo before merging one.
