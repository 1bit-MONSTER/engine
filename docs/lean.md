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
# The lean option: ROCmFP4, ROCmI4 and Hadamard Q4_0

`1bit serve --lean` trades a little accuracy for speed: with a Hadamard-rotated file it reads
Qwen3.8-27B prompts at 509 tok/s, against about 400 on the exact path. It runs models in the AMD-focused
formats of [ROCmFPX](https://github.com/charlie12345/ROCmFPX) (MIT), a llama.cpp fork
that upstream llama.cpp cannot read, so the lean route has its own tree:
`third_party/llama.cpp-rocmfpx`, pinned to a commit measured on Strix Halo.

| Command | Format | Device | Best at |
|---|---|---|---|
| `1bit serve -m model-ROCMFP4.gguf --lean` | ROCmFP4 (`Q4_0_ROCMFP4_STRIX_LEAN`) | `Vulkan0` | decode |
| `1bit serve -m model-ROCMI4.gguf --lean --device rocm` | ROCmI4 (`Q4_0_ROCMI4`) | `ROCm0`, W4A4 | prompt processing |
| `1bit serve -m model-Q4_0-H32.gguf` | Q4_0, Hadamard-rotated (`tools/hadamard_q4_0.py`) | `ROCm0`, W4A4 | prompt processing, closest to the model of the W4A4 routes |

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
- **ROCmI4 on ROCm:** the gfx1151 W4A4 path lifts prompt processing 17% (465 against
  397) and leaves decode as it is. The Vulkan numbers in the table (6 / 4.1) are from
  before Vulkan had a ROCmI4 kernel.
- **ROCmI4 on Vulkan (2026-09-25):** the engine's ROCmFPX pin, `1bit/vulkan-rocmi4` on our
  fork 1bit-MONSTER/ROCmFPX, adds Vulkan kernels for `Q4_0_ROCMI4` with the ROCm path's
  exact semantics. `test-backend-ops -b Vulkan0` passes (MUL_MAT 26/26, MUL_MAT_ID 73/73,
  GET_ROWS 4/4, CPY 17/17); Qwen3-0.6B ROCmI4 over wikitext-2 20 x 512 reads perplexity
  28.33 on Vulkan against 28.37 on ROCm, at pp512 13714 / tg128 335 tok/s on Vulkan. The
  27B row above has not been re-measured on Vulkan yet. This also makes ROCmI4 files
  usable in 1bit OS, which has Vulkan and no ROCm.
- The ROCmFPX files were quantized without an importance matrix; UD-Q4_K_XL was
  made with one. An imatrix would narrow the accuracy gap somewhat.

## Round 2: with Unsloth's imatrix, and what to use (2026-09-24)

Round 1's ROCmFPX files had no importance matrix. Rebuilt with Unsloth's own
(`imatrix_unsloth.gguf`, the one UD-Q4_K_XL was made with), next to Unsloth's smaller files:

| File | Size | Mean KLD | Same top token | Best decode, tok/s |
|---|---|---|---|---|
| UD-Q4_K_XL (default) | 16.4 GiB | **0.008** | **95.3%** | 12.2 (Vulkan); 35.0 with `--mtp` |
| UD-IQ4_XS (Unsloth) | 13.3 GiB | 0.019 | 93.3% | 15.0 (Vulkan) |
| **UD-Q3_K_XL (Unsloth)** | 12.2 GiB | 0.028 | 92.1% | **15.8 (Vulkan)** |
| ROCmI4 + imatrix | 13.9 GiB | 0.036 | 91.3% | 13.5 (ROCm W4A4; prompt 455) |
| ROCmFP4 + imatrix | 13.8 GiB | 0.045 | 89.6% | 14.1 (Vulkan) |
| ROCmFP2 + imatrix | 8.6 GiB | 0.341 | 75.5% | 21.2 (Vulkan) |

- **For a smaller, faster file, use Unsloth's UD-Q3_K_XL on the default route** (no
  `--lean` needed): it is smaller, faster and closer to the model than either
  4-bit ROCmFPX format, even with the imatrix.
- **`--lean` keeps two jobs:** ROCmI4 for the fastest prompt processing (455 tok/s
  with W4A4 on ROCm), and ROCmFP2 when memory is the limit (8.6 GiB, 21 tok/s, at a
  large accuracy cost: 75% top-token agreement).
- **MTP beats every format change:** `--mtp` on the default file gives 2.4-2.9x
  (docs/serve.md), more than any quant here.

## ROCm accuracy: MMQ only

hipBLAS returns wrong GEMMs on gfx1151 (ROCm/rocm-libraries#11530). Qwen3.8-27B UD-Q4_K_XL
against BF16, 2026-09-24:

| Build | Mean KLD | Same top token | pp512 / tg128 |
|---|---|---|---|
| Vulkan | 0.0081 | 95.3% | 353 / 11.9 |
| ROCm, default | 0.0094 | 95.1% | 343 / 11.5 |
| ROCm, `GGML_CUDA_FORCE_MMQ` | **0.0064** | **96.5%** | 354 / 11.5 |

The default ROCm build is correct for 4-bit GGUF (it mostly runs MMQ already); forcing
MMQ is more accurate at the same speed, so the lean ROCm build does. F16/BF16 models go
through hipBLAS and are not safe on ROCm here.

## Two backends at once

`Vulkan0`, `ROCm0` and `HRX0` are one GPU. Decode on both at the same time
(UD-Q4_K_XL, tg256): Vulkan alone 12.2 tok/s, ROCm alone 11.9; **Vulkan + ROCm together
7.5 + 6.8 = 14.3 (+18% total)**; two Vulkan processes 6.3 + 6.3 = 12.7 (+4%). A single
stream uses about 200 of the bus's roughly 256 GB/s, and two different drivers fill the
gap better than two copies of one. Each stream slows down, so it pays for serving
several requests at once (one per backend), not for one chat.

## Hadamard-rotated Q4_0: W4A4 prompt processing

The fastest prompt processing the engine has on Strix Halo. The W4A4 kernel multiplies int4
weights by int4 activations on the Radeon's matrix units, twice the int8 rate. It reads plain
Q4_0 weights bit for bit (a Q4_0 code `q` means `q - 8`, and `q ^ 8` is exactly that as a signed
4-bit value), so the weights stay exact and only the activations are rounded to 4 bits. Rounding
activations to 4 bits loses accuracy wherever a few large values in a block of 32 force a coarse
scale on the rest. A 32-point Walsh-Hadamard rotation spreads those values before rounding. The
same rotation is applied to the weights once, per 32-element block along K, so
`x . w = (Hx) . (Hw)` and the product is unchanged in exact arithmetic.

**Make the file** from a high-precision source (Q8_0, BF16, F16), with the imatrix you would
quantize it with anyway:

```sh
tools/hadamard_q4_0.py Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-Q4_0-H32.gguf --imatrix imatrix_unsloth.gguf
```

It rotates the attention, FFN and delta-net alpha/beta projections in a copy of the source,
gives the imatrix the matching change, quantizes those tensors to Q4_0 and every other matmul
weight to another type, stamps the file `onebit.hadamard_q4_0 = 32`, and checks that every Q4_0
tensor in it is a rotated one (it deletes a file that breaks that). Qwen3.8-27B: about 20 minutes
on Strix Halo, 15,182 MiB (4.66 bits per weight), 456 rotated tensors. That file is published as
[1bit-MONSTER/Qwen3.8-27B-Q4_0-H32-GGUF](https://huggingface.co/1bit-MONSTER/Qwen3.8-27B-Q4_0-H32-GGUF).

**Serve it:** `1bit serve -m Qwen3.8-27B-Q4_0-H32.gguf` in a build with `-DONEBIT_LEAN=ON
-DONEBIT_LEAN_ROCM=ON`. `serve` reads the stamp, runs the file on the lean ROCm build
(`--device auto` picks it; any other device is refused, because only this build rotates the
activations to match), and sets `GGML_Q4_0_HADAMARD=1` and `GGML_W4A4_TENSORS=all` for it. Set
`GGML_W4A4_TENSORS=` (empty) to run the rotated file on the exact int8 path instead.
`tests/hadamard_route.sh` (ctest `hadamard_route`) checks the routing and the environment
without a GPU.

**Measured** (Strix Halo, Qwen3.8-27B, KLD against BF16 over wikitext-2 40 x 512, pp512 from
llama-bench with ub 512):

| Qwen3.8-27B | pp512 | PPL | Mean KLD | Same top token |
|---|---|---|---|---|
| Q4_0, exact int8 | ~400 | 6.021 | 0.029 | 91.9% |
| Q4_0, W4A4 without rotation | 461-488 | 6.241 | 0.084 | 87.4% |
| **Q4_0-H32 (rotated), W4A4** | **503-512** (509 on the engine's lean ROCm build) | **6.096** | **0.055** | **89.3%** |
| Q4_0-H32 on the exact int8 path | ~400 | 6.016 | 0.031 | 91.6% |

The rotation removes a third of the 4-bit error. Against the exact path the rotated file costs
+1.2% perplexity for +28% prompt speed. The last row shows the rotation itself is exact. Through
`1bit serve -m Qwen3.8-27B-Q4_0-H32.gguf` (no device flag: the stamp picks the route), a
1,838-token prompt runs at 440-470 tok/s, against about 330 on the Vulkan route.

**Where else it helps.** The same ROCmFPX pin carries two prompt-processing kernels every model
with delta-net layers (Qwen3.8, Qwen3.6) uses on ROCm: a delta-net prefill kernel that keeps
each state column's rows in registers and stages the per-token inputs in shared memory, and a
tiled transpose for the concatenation that feeds the delta-net convolution. Both pass
`test-backend-ops` against the CPU (GATED_DELTA_NET 42/42, CONCAT 129/129).

**Decode.** This route prefills; it does not decode fast. The ROCm build decodes Qwen3.8-27B at
about 11 tok/s without drafting, and its MTP drafting slowed prompt processing badly in our
runs. For decode-heavy work use `--device vulkan --dflash` ([serve.md](serve.md)): 45.7 tok/s on
code.

## Build

```
git submodule update --init --depth 1 third_party/llama.cpp-rocmfpx
cmake -B build -G Ninja -DONEBIT_LEAN=ON                        # Vulkan: ROCmFP4
cmake -B build -G Ninja -DONEBIT_LEAN=ON -DONEBIT_LEAN_ROCM=ON  # also ROCm: ROCmI4, Hadamard Q4_0
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
