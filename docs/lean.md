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

It rotates the attention, FFN (MoE experts and shared experts included) and delta-net
alpha/beta projections in a copy of the source,
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

### MoE: W4A4 on the experts

W4A4 runs the expert matmuls (`MUL_MAT_ID`) too
([ROCmFPX #3](https://github.com/1bit-MONSTER/ROCmFPX/pull/3)). Before that, an MoE file took
W4A4 on its attention only, and its experts, where nearly all of the prompt's work is, stayed on
the exact int8 path. `tools/hadamard_q4_0.py` rotates the experts along with the rest.

```sh
tools/hadamard_q4_0.py Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf Qwen3-Coder-30B-A3B-Q4_0-H32.gguf
1bit serve -m Qwen3-Coder-30B-A3B-Q4_0-H32.gguf
```

**Measured** (Strix Halo, Qwen3-Coder-30B-A3B from unsloth's Q8_0, rev `b17cb02d`; KLD against
that Q8_0 over wikitext-2 40 x 512, Q8_0 PPL 8.489; llama-bench, 3 runs; 337 rotated tensors,
16,497 MiB):

| Qwen3-Coder-30B-A3B, Q4_0 | pp512 | pp2048 | tg128 | PPL | Mean KLD | Same top token |
|---|---|---|---|---|---|---|
| rotated, exact int8 | 1,218 | 1,186 | 79.0 | 8.671 | 0.048 | 91.5% |
| rotated, W4A4 on attention only (before ROCmFPX #3) | 1,236 | 1,208 | 79.1 | 8.720 | 0.061 | 90.3% |
| **rotated, W4A4 on attention and experts** | **2,156** | **2,048** | 79.2 | 8.862 | **0.081** | **88.7%** |
| not rotated, W4A4 everywhere | 2,155 | 2,058 | 79.0 | 8.847 | 0.137 | 85.3% |

Prompt processing goes up about 75% for an MoE, against about 25% for the dense 27B above,
because an MoE spends almost all of its prompt compute in the experts. The accuracy cost is
about the dense model's: KLD 0.048 to 0.081 (27B: 0.029 to 0.055). Without the rotation, W4A4
costs almost twice as much (0.137). Decode does not use W4A4 and does not change.

The gain depends on how much of the model's prompt time is in the experts. Same method, from
each model's Q8_0:

| Q4_0, rotated | Route | pp512 | pp2048 | tg128 | PPL | Mean KLD | Same top token |
|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (Q8_0 PPL 5.825; 401 tensors, 18,832 MiB) | exact int8 | 1,454 | 1,408 | 63.3 | 5.981 | 0.046 | 90.6% |
| | W4A4 on attention only | 1,514 | 1,478 | 64.0 | 6.032 | 0.058 | 89.6% |
| | **W4A4 on attention and experts** | **1,616** | **1,579** | 63.9 | 6.117 | **0.077** | **88.0%** |
| GLM-4.7-Flash (Q8_0 PPL 10.41; 562 tensors, 16,171 MiB) | exact int8 | 904 | 853 | 58.6 | 7.973 | 0.528 | 83.1% |
| | W4A4 on attention only | 1,029 | 861 | 61.3 | 8.130 | 0.550 | 81.3% |
| | **W4A4 on attention and experts** | **1,599** | **1,239** | 61.9 | 8.455 | **0.587** | **79.3%** |

Qwen3.6-35B-A3B gains only about 11%: most of its layers are delta-net layers, whose prompt
work does not run through a Q4_0 matmul. GLM-4.7-Flash (DeepSeek-2 layout, MLA attention; the
tool rotates the MLA projections `attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, `attn_kv_b`,
`attn_k_b`, `attn_v_b`) gains 77% at pp512, and W4A4 adds KLD 0.06, as on the Qwen models.

GLM-4.7-Flash does not take plain Q4_0 well, W4A4 or not. A plain, unrotated Q4_0 on the exact
path already scores KLD 0.493 against the Q8_0; any single group of tensors at Q4_0 (the
experts, the attention, or the shared expert, dense layer and embedding) costs about 0.14 on its
own. Its wikitext perplexity drops under Q4_0 (10.41 to 7.97-8.73) because the output
distribution flattens, so perplexity alone reads this model the wrong way round. The Q8_0
reference is sound: it gives PPL 10.44 on upstream HIP and 11.01 on Vulkan, and KLD 0.015
across backends. An importance matrix repairs the weights but not the activations. An unrotated Q4_0 made with
an imatrix (wikitext-2 train, 150 x 512) scores KLD 0.197 on the exact path, PPL 10.81; the same
file with W4A4 on attention scores 0.322 and with W4A4 everywhere 0.571. On the rotated file an
imatrix does nothing: Q4_0 has one scale per 32 values, the rotation spreads the outlier columns
across the block, and even an imatrix collected on the rotated activations moves KLD from 0.528
to 0.522. For GLM-4.7-Flash, use the exact path with an imatrix Q4_0, or a higher-precision
file; its activations do not take 4 bits.

### End to end: time to first token and effective speed

A user waits for the whole answer, not for the decode phase. A request with P prompt tokens and
N output tokens takes TTFT + (N - 1) / decode speed, and TTFT is mostly P / prefill speed. The
effective speed, N over that whole time, is what the user gets (the idea behind Artificial
Analysis's end-to-end response time). `tools/e2e_bench.py` measures it against any running
server (`1bit serve` or llama-server): streamed `/v1/completions` with prompts cut to exactly P
tokens, exactly N output tokens, no prompt cache, median of 3.

```sh
tools/e2e_bench.py --url http://127.0.0.1:8080 --prompt-tokens 512,2048,8192,16384 --max-tokens 256
```

**Measured** (Strix Halo, Qwen3-Coder-30B-A3B, N = 256, `-c 20480 -fa on`): the engine's
Vulkan build with a Q4_K_M, against the rotated
Q4_0 above on the ROCm route, exact int8 and W4A4:

| Prompt tokens | Route | TTFT | Decode | End to end | **Effective** |
|---|---|---|---|---|---|
| 512 | Vulkan, Q4_K_M | 0.38 s | 78.7 t/s | 3.62 s | **70.7 t/s** |
| | ROCm, exact int8 | 0.43 s | 73.9 t/s | 3.89 s | 65.9 t/s |
| | ROCm, W4A4 | 0.25 s | 71.4 t/s | 3.85 s | 66.6 t/s |
| 2,048 | Vulkan, Q4_K_M | 1.56 s | 76.0 t/s | 4.92 s | 52.1 t/s |
| | ROCm, exact int8 | 1.72 s | 69.6 t/s | 5.38 s | 47.6 t/s |
| | ROCm, W4A4 | 1.00 s | 69.8 t/s | 4.66 s | **55.0 t/s** |
| 8,192 | Vulkan, Q4_K_M | 8.48 s | 61.6 t/s | 12.62 s | 20.3 t/s |
| | ROCm, exact int8 | 7.92 s | 58.2 t/s | 12.56 s | 20.4 t/s |
| | ROCm, W4A4 | 5.03 s | 57.5 t/s | 9.46 s | **27.0 t/s** |
| 16,384 | Vulkan, Q4_K_M | 24.87 s | 50.2 t/s | 29.92 s | 8.6 t/s |
| | ROCm, exact int8 | 19.94 s | 47.8 t/s | 25.26 s | 10.1 t/s |
| | ROCm, W4A4 | 13.76 s | 47.1 t/s | 19.17 s | **13.4 t/s** |

Vulkan decodes about 10% faster, and for a short prompt that wins. From about 2,000 prompt
tokens on, W4A4's first token comes early enough to more than pay for it: at 8K the answer is
done 3.2 s sooner (+33% effective speed), at 16K the first token comes 11 s sooner and the
effective speed is +56%. Agents and coding tools send long prompts (files, tool output,
history), so that is where this route is for. The two files are not the same quantization
(Q4_K_M against rotated Q4_0), so the table compares routes as they ship, not kernels.

`1bit serve --long-model` takes both at once: short conversations on Vulkan, long ones on
this route ([serve.md](serve.md#short-and-long-prompts---long-model)).

The rotation removes a third of the 4-bit error. Against the exact path the rotated file costs
+1.2% perplexity for +28% prompt speed. The last row shows the rotation itself is exact. Through
`1bit serve -m Qwen3.8-27B-Q4_0-H32.gguf` (no device flag: the stamp picks the route), a
1,838-token prompt runs at 440-470 tok/s, against about 330 on the Vulkan route.

**Where else it helps.** The same ROCmFPX pin carries two prompt-processing kernels every model
with delta-net layers (Qwen3.8, Qwen3.6) uses on ROCm: a delta-net prefill kernel that keeps
each state column's rows in registers and stages the per-token inputs in shared memory, and a
tiled transpose for the concatenation that feeds the delta-net convolution. Both pass
`test-backend-ops` against the CPU (GATED_DELTA_NET 54/54, CONCAT 129/129).

A KernelForge campaign later made the delta-net prefill kernel 5-7x faster, at 3.02 -> 0.41 ms per
512-token layer (ROCmFPX#5, [tools/kernelforge](../tools/kernelforge/README.md)). It uses:

- DPP reductions instead of shuffles;
- two state columns per lane, which halves the shared-memory reads;
- rows interleaved across banks;
- 16-byte staging.

The rotated 27B's prompt gains about 11% with or without a drafter (the table below).

**One server: W4A4 prompts and DFlash2 decode.** Without a drafter this route decodes
Qwen3.8-27B at about 13 tok/s. With the DFlash2 drafter ([serve.md](serve.md#dflash-draft-models---dflash))
the same server decodes three times faster and keeps 95% of the prompt speed:

```sh
1bit serve -m Qwen3.8-27B-Q4_0-H32.gguf --dflash Qwen3.8-27B-DFlash2-q8_0.gguf
```

The ROCmFPX pin carries upstream's DFlash2 support (ggml-org/llama.cpp#27816, ported in
ROCmFPX#2), a HIP top-k that keeps the drafter's 248k-vocabulary candidate pick on the GPU (it
fell back to the CPU before: 11 tok/s instead of 28), and bounded recurrent-state rollback for
DFlash, so a partly accepted block rewinds the delta-net state instead of restoring a checkpoint
and replaying. `serve` passes `--spec-draft-p-min 0` with `--dflash`: this tree's default of 0.75
cut DFlash2 blocks from 6.7 to 5.4 tokens a step.

Three things kept the drafter from costing the prompt (ROCmFPX#4 and `serve`):

- **Rollback snapshots in the prefill kernel.** Rollback keeps the delta-net state after each of
  the last K tokens, and the chunked prefill kernel did not write those, so with a drafter every
  prompt ran the per-token kernel: 1,322 tokens took 2,966 ms instead of 2,588. The prefill
  kernel now writes them (2,762 ms).
- **K is the drafter's block.** `serve` reads `dflash.block_size` from the drafter and drafts
  `block - 1` tokens, which is also K; asking for 16 against DFlash2's block of 8 wrote twice the
  snapshots.
- **512-token micro-batches for a dense file.** The 1024-token micro-batches `serve` gives
  rotated MoE files are slower on the dense 27B: 471-473 tok/s against 491-495 without a drafter.

The same ROCmFPX change also fixed the drafter's input on long prompts: the model's hidden states
for a batch of several micro-batches were all written at the first one's place, so the DFlash
(and EAGLE3) encoder read stale features for all but the last.

Measured with [`tools/bench.py`](bench.md) through `1bit serve` on Strix Halo, 2026-09-29, 2
interleaved rounds under the box lock. The prompt is its default (the first 8,000 characters of
`docs/*.md`, about 1,800 tokens; best / median of 10, as `serve` reports it, context checkpoints
included). Decode is 256 greedy tokens on the code / prose / short prompts (best / median of 6):

| Qwen3.8-27B-Q4_0-H32, lean ROCm route | Prompt t/s | Decode tok/s |
|---|---|---|
| no drafter | 558 / 550 | 13.2 / 13.2 / 13.6 |
| `--dflash` (DFlash2 Q8_0) | **521** / **516** | **42.1** / **24.9** / 13.6 |

Before the prefill kernel of ROCmFPX#5, the same run measured 502 / 494 and 470 / 467.

Mean accepted block: 6.54 tokens on code, 4.23 on prose, in line with upstream on this drafter
(6.71 / 4.25). Greedy output matches the no-drafter run on the code and short prompts; on prose
one near-tie phrase differs after 332 characters (batched verification rounds differently). What
the drafter still costs the prompt, about 0.2 s here, is its encoder reading the model's hidden
states for every prompt token (0.1 s) and the rollback snapshots.

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
