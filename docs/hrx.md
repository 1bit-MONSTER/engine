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
# HRX on the Radeon iGPU

On Strix Halo, HRX (AMD's ggml-hrx with our Loom kernels) reads Qwen3.8-27B prompts at 335 tok/s and
decodes ZAYA1-8B at about 90 tok/s. It is the engine's GPU route: a 14,435-token prompt runs at 265 tok/s,
and Qwen3.8-27B decodes at 97% of the previous Vulkan figure.

It is one llama.cpp build with one GPU backend, ggml-hrx, and the CPU. The engine builds no Vulkan or
ROCm llama.cpp (RFC #213 stage 3); what HRX does not run, Lemonade serves with its own
llamacpp backends ([lemonade.md](lemonade.md)). Its `llama-server` exposes on Strix Halo:

```
HRX0:    AMD Radeon 8060S Graphics (Node 1) (gfx1151)
```

`1bit serve` runs GGUF models with it ([serve.md](serve.md)):

| `--device` | Device | Serves |
|---|---|---|
| `hrx` (and `auto` for GGUF) | `HRX0` | AMD's ggml-hrx: the engine's GPU route |
| `cpu` | the CPU | the same llama-server with no GPU layers |

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

### Fixed: `--device hrx` answered wrongly on mid-length prompts

Measured 2026-09-24, Qwen3-0.6B Q4_K_M, greedy chat through llama-server on `HRX0`:
before the fix, prompts of 404 to 1,733 tokens came back as garbled text, while 146
tokens and 2,167 tokens or more read correctly. Prompt processing was right at every
length; decoding one token at a time went wrong once the KV cache held more than 256
tokens.

**Cause:** above 256 KV tokens the single-query flash attention kernel
(`flash_attention_decode_split_f32_f16_wmma.loom`) combines its per-block partial
results with a cooperative reducer. That reducer runs four subgroups over eight query
rows per KV head, and it checked only that a row's query head exists, not that the row
belongs to this KV head. With fewer than eight query heads per KV head (Qwen3-0.6B has
2), the spare subgroups reduced stale rows and wrote them over the next KV head's output.
The reducer used at 256 tokens and below already stays inside its own rows. The fix
(fork commit `79788e9`, on `1bit/hrx-vulkan-patched`) adds that bound to both copies of
the kernel.

After the fix, fed the same tokens as Vulkan (8 decode steps each), the relative logit
difference is 0.01 to 0.03 at 248, 250, 256, 300, 400, 1,200 and 2,040 tokens, the same
as below 256 before; and the chat answers match Vulkan's at every length from 404 to
2,844 tokens. `test-backend-ops -o FLASH_ATTN_EXT` cannot check it on this pin: 12 of
5,141 cases pass with and without the fix, because HRX0 refuses the test's empty `NONE`
graph node.

### Fixed: MoE models with more than 128 experts (Qwen3.6-35B-A3B)

Before this pin, Qwen3.6-35B-A3B (256 experts) failed on `HRX0` in two ways:
- **Prompt batches faulted.** Every batch of 2 or more tokens faulted the GPU
  (`HSA_STATUS_ERROR_MEMORY_FAULT`).
- **Decode was wrong.** Single-token decode ran at about 41 tok/s, but its output was
  wrong: KLD 15.3 against Vulkan, top-1 0%.

There were two causes:

1. **The router assumed 128 experts.** `dispatch-moe-router.cpp` wrote the expert
   partition table in the 128-expert layout (7-bit expert ids) for every expert count.
   The `mul_mat_id` kernels that read the table decode a 9-bit layout. So experts
   128-255 got no partitions, and an expert with 2 or more tokens read past its row.
   Fork commit `96049a2` picks the layout by expert count
   (`dispatch_registration/common/dispatch-moe-routing-layout.h`). Nothing changes up
   to 128 experts.
2. **Fused-only ops went to the CPU.** Our earlier claim rule accepted only nodes that
   run standalone. So the router chain, L2_NORM, SOFTPLUS, GATED_DELTA_NET and the
   per-head RMS_NORM/ROPE, which HRX runs only inside fused patterns, went to the CPU.
   The 35B then split per layer (KLD 2.63), and Qwen3-Coder-30B-A3B could not decode.
   Fork commit `9a7aad6` (`fused-context-claim.h`) claims those nodes when their fused
   pattern's producers are present and their shapes fit.

Measured on Strix Halo (llama-bench, fa on, `-r 3`; KLD against Vulkan):

| model | metric | before | this pin |
|---|---|---|---|
| Qwen3.6-35B-A3B Q8_0 | pp512 | 141 | 909 |
| Qwen3.6-35B-A3B Q8_0 | pp2048 | 156 | 1028 |
| Qwen3.6-35B-A3B Q8_0 | tg128 | 14.1 | 35.3 |
| Qwen3.6-35B-A3B Q8_0 | KLD vs Vulkan | 2.63 | 0.0046 |
| Qwen3-Coder-30B-A3B Q4_K_M | pp512 | 1114 | 2040 |
| Qwen3-Coder-30B-A3B Q4_K_M | tg128 | fails | 89.4 |
| Qwen3-0.6B Q4_K_M | pp512 / tg128 | 21938 / 322.5 | 21789 / 323.8 |

`test-backend-ops -b HRX0`: 790/790.

### Fixed: decode-split multipass gap above 2048 ([engine#124](https://github.com/1bit-MONSTER/engine/issues/124))

The multipass decode-split reducer (`reduce_completed.multipass`, used above
`key_value_token_capacity` 2048) finished with a scalar output pass: each workitem owned
one output channel and walked every KV block serially, recomputing
`expf(partial_max - maximum)` once per (block, element). Only half the workgroup is live
at `value_head_size = 128`, so decode above 2048 stayed about 16% below the <=2048
corridor. Fork PR
[1bit-MONSTER/llama.cpp#30](https://github.com/1bit-MONSTER/llama.cpp/pull/30) computes
the per-block scale once into a per-row workgroup (LDS) stage and runs one vectorised,
all-rows output pass (4 channels per workitem). The per-channel block order is
unchanged, so the reduce is bit-identical.

Measured on Strix Halo (Qwen3-Coder-30B-A3B-Instruct Q4_K_M, `-dev HRX0`,
`llama-bench -p 0 -n 8 -r 5`, median of 6 interleaved runs):

| depth | blocks | before | this pin | delta |
|---|---:|---:|---:|---:|
| 2000 (<=2048 control) | 32 | 66.06 | 68.82 | +4.2% |
| 2100 | 33 | 58.19 | 67.15 | **+15.4%** |
| 3000 | 47 | 50.84 | 60.89 | **+19.8%** |
| 4800 | 76 | 41.83 | 51.02 | **+22.0%** |

`d2100/d2000` moves 0.881 -> 0.976, so the boundary cliff is gone. 1.24 GB of
decode-path logits (`llama-perplexity -c 2049 -b 1 --save-all-logits`) are byte-identical
to the previous pin, the buried code word is exact at 4700 tokens, and no GPU faults
were observed.

### MoE decode and the kernel cache (llama.cpp #32, #33)

- **MoE decode.** With one to four tokens, a `MUL_MAT_ID` (a MoE expert matmul) runs as a GEMV over
  the routed experts' rows (`common.mul_mat_id.decode_f32_wave64`), not the batched WMMA kernel,
  which read a single token's expert at ~45 GB/s. HRX0 decode: ZAYA1-8B 46.3 -> 62.0 tok/s,
  GLM-4.7-Flash 22.1 -> 25.3; ZAYA's teacher-forced check is unchanged (69/96) and batch-1
  perplexity stays within 0.1-0.35% of the CPU backend. `GGML_HRX_DISABLE_DISPATCH=mul_mat_id.decode`
  turns it off for comparison. Found with HRX's dispatch profiler: `HRX_PROFILE_MODE=dispatch
  HRX_PROFILE_FILE=x.prof`, read with `iree-profile executable x.prof` (built from hrx-system with
  Bazel).
- **Kernel cache.** HRX compiles each kernel specialization on first use, and the specialization
  includes the batch's token count, so each new prompt-length remainder compiles a new set. The
  compiled kernels are kept in `~/.cache/1bit/hrx-jit` (`GGML_HRX_JIT_CACHE_DIR`), keyed on every
  compile input and the compiler library, so later processes skip the compiles: ZAYA1-8B's first
  5.4K-token prompt in a new process 4.46 -> 3.43 s, first short prompt 0.47 -> 0.22 s, with
  identical output. `GGML_HRX_JIT_CACHE=0` turns it off; it is off while a Loom sanitizer is set.
  Deleting the directory is always safe.
- **Fused residual scale (llama.cpp #35).** ZAYA scales both residual branches twice per layer,
  `(x + bx) * sx + (r + br) * sr`, which ran as up to nine ADD / MUL dispatches per layer. Each side
  now runs as one dispatch (`common.res_scale_pair.*`, `ops/res_scale_pair_f32.loom`): 3,366 -> 3,165
  dispatches per token, bitwise the same output, ZAYA1-8B HRX0 decode 67.0 -> ~69.6 tok/s and
  pp512 2,109 -> 2,183. `GGML_HRX_DISABLE_DISPATCH=res_scale_pair` turns it off. At decode most of
  a ZAYA token is small dispatches and the ~1.8 us gaps between them, not matmuls; the profiler
  (`HRX_PROFILE_MODE=dispatch`) shows which ones. Profile through `llama-server`: `llama-bench`
  under the profiler fails.
- **ZAYA decode fusions (llama.cpp #36-#40).** ZAYA1-8B HRX0 decode 67 -> ~90 tok/s (Vulkan 93),
  each step measured in interleaved A/B runs:
  - PR #36 skips CONTs of fresh contiguous results in `src/models/zaya.cpp` (views keep theirs: on HRX
    their consumers take slower strided paths), bitwise the same output: +2.4%.
  - PR #39 runs the CCA convolution (Q/K concat, conv-state update, depthwise conv, grouped 2-tap conv
    and biases, ~35 graph nodes) as one dispatch (`zaya.cca_conv.decode_f32`): 72 -> 81 tok/s.
  - PR #40 runs the CCA query/key mixing and per-head norms (~30 nodes) as one dispatch
    (`zaya.cca_qk_norm.decode_f32`): 81 -> 90 tok/s.
  PRs #39 and #40 accumulate in their own order, so output is not bitwise the same; teacher-forced top-1
  against the FP32 model stays 69/96 and batch-1 perplexity moves 33.31 -> 33.49 -> 33.45 (CPU backend
  34.30). Both match decode only (one token, one sequence). #37 and #38 remove runtime integer
  division from #35's and #32's kernels, which newer Loom rejects. Each new `.loom` file carries a
  reference kernel and `check.case` differentials that pass under `iree-test-loom --sanitizer=access`.
  `GGML_HRX_DISABLE_DISPATCH=zaya.cca` turns the two ZAYA kernels off for comparison.

### Ternary Bonsai (PrismML's Hadamard-folded GGUFs)

PrismML's [Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) is Qwen3.8-27B
trained to weights of -1, 0 and +1, with one fp16 scale per 128 weights. It runs on `HRX0`,
straight from PrismML's file:

```sh
1bit serve -m Ternary-Bonsai-2-27B-PTQ1_0.gguf        # --device auto picks hrx for this file
```

- **The weights** ([llama.cpp #62](https://github.com/1bit-MONSTER/llama.cpp/pull/62)). PrismML's
  PQ2_0 (ggml type 142) and PTQ1_0 (143) are native types in our HRX build, with the layouts of
  PrismML's llama.cpp fork. Our CPU decode of every tensor of both 27B files matches PrismML's own
  build bit for bit. On `HRX0` the K-quant decode kernels read them directly, so one copy of the
  weights is resident:

  | Ternary-Bonsai-2-27B | weights | GPU peak over idle | tg128 (tok/s) | pp512 (tok/s) |
  |---|---|---|---|---|
  | Q4_0 + packed ternary (below) | 14.13 GiB | 21.8 GiB | 15.3 | 13.1 |
  | PQ2_0 | 6.70 GiB | 10.1 GiB | 15.8 | 14.2 |
  | PTQ1_0 | 5.53 GiB | 8.9 GiB | 14.3 | 14.2 |

  Balanced power mode, llama-bench `-fa 1`, medians of 3. Against the CPU logits of the
  bit-identical Q4_0 copy (3 x 512) all three give the same KLD, 0.000129 at decode (`-ub 8`) and
  0.002051 at `-ub 512`. Files in these types run on `--device hrx` only, with or without
  `prism.hadamard` keys (Ternary-Bonsai-1.7B has none). For another route,
  `tools/ternary_to_q4_0.py` writes the same weights as Q4_0 (q = trit + 8 with the group's own fp16
  scale in each of its four blocks); every group is decoded back and compared before the file is
  kept, and `tests/ternary_to_q4_0_test.py` checks it against PrismML's own encoder.
- **Packed ternary decode** ([llama.cpp #54](https://github.com/1bit-MONSTER/llama.cpp/pull/54)).
  For a file stamped `onebit.ternary_q4_0`, `1bit serve` sets `GGML_HRX_TERNARY_Q4_0=1`. HRX0's
  K-quant decode kernels then read those Q4_0 weights from a 2-bit copy made at load (68 bytes per
  256 values: the two group scales and the trits), 2.125 bits per weight instead of 4.5. The load
  checks that every block is ternary and refuses the model otherwise. Prompt batches still use the
  Q4_0 weights, so both copies stay resident: about +4 GiB of GPU memory for Bonsai-2-27B (21.3 GiB
  peak over idle instead of 17.3), roughly 30% of the file. Bonsai-2-27B decode goes from 9.3 to
  15.4 tok/s (balanced power mode); prompt speed is unchanged. Against CPU logits the packed path is
  closer than the Q4_0 one (KLD 0.00013 against 0.0019). `GGML_HRX_TERNARY_Q4_0=0` turns it off.
- **The rotation.** Bonsai stores almost every matmul weight rotated by a 1024-point
  Walsh-Hadamard transform with fixed signs (`prism.hadamard.*` keys), and the activations have to
  be rotated to match. Our llama.cpp does that (`src/llama-hadamard.{h,cpp}`, llama.cpp #34,
  following PrismML's implementation): a plain F32 matmul against the rotation matrix before each
  folded weight, the inverse after the token-embedding lookup, and a check on the first graph that
  no folded weight is used without it. Upstream llama.cpp would load the file and answer garbage,
  so `1bit serve` sends a `prism.hadamard` file, or one in PrismML's own types, to `hrx` only
  (`tests/prism_route.sh`).
- **The rotation on the GPU** ([llama.cpp #64](https://github.com/1bit-MONSTER/llama.cpp/pull/64)).
  The rotation is a MUL_MAT that ggml hints as `GGML_HINT_SRC0_IS_HADAMARD`. The dense HRX matmul
  takes at most 2048 rows, so at a 512-token prompt the 17408-wide FFN input (8704 rows of 1024)
  ran on the CPU in every layer. `hadamard_f32.loom` now computes the normalized transform per row
  (log2(n) butterfly stages, the matrix never read), as the CPU, Vulkan, CUDA and Metal backends do
  for this hint. Ternary-Bonsai-2-27B (balanced mode): pp512 13.5 -> 90.9 tok/s, and decode
  15.8 -> 19.0 tok/s, since the decode rotations no longer go through a dense WMMA kernel.
  `tests/test-hrx-hadamard` checks six shapes against the CPU and an exact product, four times each.
- **Checked.** Against the F16 model on the CPU in PrismML's own fork, wikitext-2 running
  perplexity on `HRX0` is 7.962 / 10.408 / 10.171 at chunks 5 / 10 / 15 (CPU: 7.958 / 10.382 /
  10.142), the difference Q4_0's 8-bit activations make. Files without the keys build the same graph
  as before.

### Prompt matmuls on the q8_1 x4 kernel (llama.cpp #55)

On Qwen3.8-27B UD-Q4_K_XL, 94% of HRX prompt time went to the generic F32 WMMA matmuls,
which work in 32-token tiles. An iree-profile dispatch trace of a 1.8K-token prompt split it
as `mul_mat_swiglu` 51%, `mul_mat_add` 25% and `mul_mat` 18%. HRX already has a q8_1 x4
prefill kernel, but two routing rules kept this model's projections, mostly Q5_K and IQ4_XS,
away from it:

1. Its matcher took Q5_K and IQ4_XS weights only when q8_1 activations already existed, and
   never for a matmul feeding a GLU. Q4_K was allowed both. Q5_K and IQ4_XS now get the Q4_K
   policy.
2. The generic fused `mul_mat_swiglu` (priority 290) outranks the q8_1 x4 matcher (285) and
   claimed every FFN gate/up projection. For prompt chunks of 256-2048 tokens in multiples of
   256, whose gate and up are Q4_K, Q5_K or IQ4_XS, it now declines. The two projections
   then take the q8_1 x4 kernel and the GLU runs as its own op.

`GGML_HRX_Q8_PREFILL_RELAX=0` restores the old routing.

Measured on Strix Halo, llama.cpp `6e42b51`, Qwen3.8-27B UD-Q4_K_XL, 2026-10-01:

| metric | old routing | this pin |
|---|---|---|
| pp512 (llama-bench, `-b 512 -ub 512`) | 98.5 | **334.6** |
| pp2048 | 98.1 | **309.5** |
| `1bit serve --device hrx`, 14,435-token prompt | 90.7 tok/s | **264.7 tok/s** (same text) |
| KLD vs BF16 logits (wikitext-2, 20 x 512) | 0.00717 | 0.00712 |
| same top token | 96.13% | 96.27% |

`test-backend-ops -o MUL_MAT -b HRX0`: 287/287. Decode does not change, since both rules apply
only to chunks of at least 256 tokens. Since
[llama.cpp #61](https://github.com/1bit-MONSTER/llama.cpp/pull/61), a Q5_K or IQ4_XS chunk that is not a
multiple of 256 sends its 256-aligned head to the q8_1 x4 kernel too, and only the tail goes to the
generic kernel. Q4_K remainders still go to the generic kernel: its packed Row64 weight layout cannot
also be resident raw. Measured against `6e42b51` (3 interleaved runs, `-b 512 -ub 512`):

| metric | `6e42b51` | `6e42b51` + #61 (`a36d547`) |
|---|---|---|
| pp400 | 67.4 | **103.4** |
| pp1862 | 172.1 | **234.6** |
| KLD vs `6e42b51` (20 chunks of 400 = 256 + 144 tokens) | | 0.00215, same top token 97.19% |

`test-backend-ops -o MUL_MAT -b HRX0`: 287/287. `GGML_HRX_Q8_PREFILL_RELAX=0` turns this off too.

### Hadamard-rotated Q4_0 files ("H32", dropped)

The engine dropped its own Hadamard-rotated Q4_0 format (owner, 2026-10-03): on HRX it was slower
and less accurate than the UD-Q4_K_XL file of the same model (Qwen3.8-27B pp512 63.5 against 335
tok/s; KLD 0.029 against 0.007 vs BF16), and the lean ROCm W4A4 route it was made for left with the
ROCm build. `tools/hadamard_q4_0.py` is removed, and `1bit serve` refuses a file stamped
`onebit.hadamard_q4_0` by name. The loader support in our llama.cpp (`llama-hadamard`, llama.cpp
#58) stays in the fork; it is the same code that rotates PrismML's files
([Ternary Bonsai](#ternary-bonsai-prismmls-hadamard-folded-ggufs)), which keep running.

### Server memory: the graph program cache cap (llama.cpp #63)

HRX builds a graph program for every new graph shape (each new prompt or batch length) and kept every
one: a long-running `llama-server` with prompts of varying length grew its GPU memory without bound
(ZAYA1-8B: about 1 GiB per request, 24.6 GiB after 28 requests). Since
[llama.cpp #63](https://github.com/1bit-MONSTER/llama.cpp/pull/63) the cache holds at most
`GGML_HRX_GRAPH_PROGRAM_CACHE` programs (default 64, `0` = unbounded): when a new program takes it over
the limit, HRX waits for the stream and drops every other program. On ZAYA1-8B, 40 requests of varying
length peak at 8.7 GiB instead of growing; greedy answers are identical, and llama-bench pp512/tg128
is unchanged. A repeated prompt length right after a flush costs 1.75 s instead of 1.12 s, since its
program is rebuilt.

### Known issues on `HRX0`

- **Several sequences per batch fail.** `llama-perplexity` with `n_seq` > 1 stops on an
  unsupported 3-D MUL_MAT; use `-b 512`. On Qwen3.5 / Qwen3.8 (Bonsai too) the gated delta net runs
  one sequence per batch, so `serve` gives these models one slot on HRX and refuses `--parallel`
  (`--parallel N` under "Not yet" below).
- **`-fa off` fails.** A SET_ROWS into the non-flash-attention V cache is rejected.
- **Decode-split flash attention is on.** `flash_attention_decode_split_next_q8` used to give
  nondeterministic attention on HRX0 ([#140](https://github.com/1bit-MONSTER/engine/issues/140)) and
  to fault Qwen3-Coder-30B-A3B, so `1bit serve --device hrx` turned it off (#148). Since llama.cpp
  `00adc2b` (see the #123 section below) it is correct, and it is as fast or faster:

  | model (Q4_K_M), tg tok/s | ctx 0 | ctx 512 | ctx 2100 |
  |---|---|---|---|
  | Qwen3-0.6B, decode-split on | 286 | 254 | 140–158 |
  | Qwen3-0.6B, off | 296–306 | 220–223 | 118–122 |
  | Qwen3-Coder-30B-A3B, on | 78–85 | 81 | 42–62 |
  | Qwen3-Coder-30B-A3B, off | 78–81 | 65–69 | 47–49 |
  | ZAYA1-8B, on | 39–41 | 40–43 | 38–42 |
  | ZAYA1-8B, off | 39–42 | 39–40 | 34–36 |

  (llama-bench `-p 0 -n 32/64`, runs interleaved on a shared strixhalo, 2026-09-27.)
  `ONEBIT_HRX_DECODE_SPLIT=0` turns it off; a `GGML_HRX_DISABLE_DISPATCH` you set yourself wins.

### Prefill on HRX, decode on Vulkan (removed)

`--prefill-device hrx` left with the Vulkan build (RFC #213 stage 3): `1bit serve` refuses it.
The fork still carries the zero-copy KV sharing code (below); what it did and how it measured is
in [hrx.md at 0baf286](https://github.com/1bit-MONSTER/engine/blob/0baf286/docs/hrx.md).

### Our patches

`1bit/hrx-vulkan` is AMD's commit unchanged; `1bit/hrx-vulkan-patched` adds:

- **Zero-copy KV sharing** (unused by the engine since it builds no ggml-vulkan): dma-buf export in ggml-hrx
  (`ggml-hrx-dmabuf.cpp`), dma-buf import in ggml-vulkan (`ggml-vulkan-dmabuf.inc`),
  `llama_kv_share_next` / `llama_kv_share_from` / `llama_kv_cells_copy` in
  llama (`llama-kv-share.cpp`), and `ONEBIT_PREFILL_DEVICE` in llama-server
  (`server-prefill.cpp`).
- **IQ3_XXS matmul on HRX.** A matrix-vector kernel (`kernels/hrx/mul_mat_vec_iq3xxs_f32.loom`)
  and its matcher (`common/dispatch-mul-mat-iq3-xxs.cpp`), ported from the
  `feat/hrx-port-ae91949` work.
- **Honest op claims.** ggml-hrx used to claim every op of a declared type, so
  ggml's scheduler handed it nodes it could not dispatch and the graph failed
  (`unsupported HRX node`) with no CPU fallback. It now claims a node only when
  its dispatcher can execute it, except nodes already in HRX memory (KV cache
  views), which cannot move. Leaf (`NONE`) nodes count as covered. GET_ROWS for batched IQ3_S
  gives wrong values, so those nodes are left to the CPU. IQ4_NL and IQ4_XS were left to the CPU too
  until [llama.cpp #41](https://github.com/1bit-MONSTER/llama.cpp/pull/41) (below).
- **Kernels for ops HRX sent to the CPU** ([llama.cpp #24](https://github.com/1bit-MONSTER/llama.cpp/pull/24)):
  a batched F16 matmul (`ggml_grouped_mul_mat_f16_f32`, e.g. ZAYA's grouped convolution, whose
  weights the loader can now place on HRX), and short-row kernels in `small_rows_f32.loom`:
  SOFT_MAX without a mask, SUM_ROWS, ARGSORT, GET_ROWS for narrow rows (strided ids, as top-k views
  are), CONT of strided views and broadcast-only REPEAT. Each matcher claims only what the existing
  kernels do not. `test-backend-ops -b HRX0` passes every case of these ops. ZAYA1-8B's decode graph
  goes from 641 graph splits to 1: 25.5 to 47.9 tok/s (tg128).
- **Prompt matmuls on the q8_1 x4 kernel** ([llama.cpp #55](https://github.com/1bit-MONSTER/llama.cpp/pull/55)):
  Q5_K and IQ4_XS prefill matmuls get Q4_K's q8_1 x4 policy, and the generic fused SwiGLU leaves those
  prompt chunks to it. On Qwen3.8-27B UD-Q4_K_XL, pp512 goes 98.5 to 334.6 tok/s
  ([section above](#prompt-matmuls-on-the-q8_1-x4-kernel-llamacpp-55)).
- **Remainder chunks on the q8_1 x4 kernel** ([llama.cpp #61](https://github.com/1bit-MONSTER/llama.cpp/pull/61)):
  Q5_K / IQ4_XS chunks not a multiple of 256 send their aligned head to it; pp400 67.4 to 103.4 tok/s
  (same section).
- **Prompts past 32K context** ([llama.cpp #59](https://github.com/1bit-MONSTER/llama.cpp/pull/59)):
  V stays row-major past `copy_transpose_f16`'s 32768-row range. pp512 at depth 32768 failed before
  (JIT `row_count` 33280 violates `range`) and runs at 227.4 tok/s now (depth 16384 unchanged at 269);
  one 34,816-token wikitext chunk gives PPL 6.2150.
- **Mixed-format SwiGLU pairs** ([llama.cpp #60](https://github.com/1bit-MONSTER/llama.cpp/pull/60)):
  the K-quant SwiGLU decode kernels stage a separate codebook for gate and for up, so Unsloth's sub-4-bit
  UD files no longer fall back to the generic path for pairs that mix formats. Qwen3.8-27B UD-IQ2_S
  tg128 3.12 to 8.32 tok/s; UD-Q4_K_XL unchanged (11.88 / 11.81); KLD 0.00013 vs the previous build,
  `test-backend-ops -b HRX0` 996/996.
- **PrismML's PQ2_0 / PTQ1_0 types** ([llama.cpp #62](https://github.com/1bit-MONSTER/llama.cpp/pull/62)):
  native ggml types 142 / 143 with a CPU reference and HRX decode on the K-quant kernels; Q1_0 decode
  moves onto them too ([Ternary Bonsai](#ternary-bonsai-prismmls-hadamard-folded-ggufs)).
- **The Hadamard rotation on HRX** ([llama.cpp #64](https://github.com/1bit-MONSTER/llama.cpp/pull/64)):
  a kernel for the `GGML_HINT_SRC0_IS_HADAMARD` MUL_MAT ([Ternary Bonsai](#ternary-bonsai-prismmls-hadamard-folded-ggufs)).
- **MXFP4 weights** ([llama.cpp #65](https://github.com/1bit-MONSTER/llama.cpp/pull/65)): MXFP4 in the
  shared dequantizer (17 bytes per 32 values, the E8M0 scale built exactly as ggml does).
  `tests/test-hrx-mxfp4` checks GET_ROWS on HRX0 bit for bit against ggml's own decoder;
  `test-backend-ops -b HRX0` 1051/1051. MUL_MAT_ID now takes only input sizes that are a multiple of
  256, which its kernels declare (until #66 below).
- **gpt-oss on HRX** ([llama.cpp #66](https://github.com/1bit-MONSTER/llama.cpp/pull/66),
  [#67](https://github.com/1bit-MONSTER/llama.cpp/pull/67), [#73](https://github.com/1bit-MONSTER/llama.cpp/pull/73)):
  MUL_MAT_ID takes input sizes that are a multiple of 32 (gpt-oss-20b's experts are 2880 wide), with a
  decode-loader stride fix that this exposed (rows were read 3072 apart); ADD_ID and the clamped SWIGLU_OAI
  run on HRX in our own kernels; and a placement guard (#73) keeps those two ops with their MUL_MAT_ID when
  the experts run on the CPU, where the mix gave wrong values (root cause open, engine #286).
  gpt-oss-20b MXFP4, balanced mode: pp512 25.8 -> about 1000 tok/s, tg128 12.6 -> about 35, text correct,
  KLD vs CPU 0.029.
- **Attention sinks** ([llama.cpp #68](https://github.com/1bit-MONSTER/llama.cpp/pull/68)): gpt-oss's
  per-head sink logits run on HRX as an exact post-correction of AMD's unchanged flash-attention output,
  O x sigmoid(LSE - sink), in our own dispatch; an output that overlaps an input's storage is refused.
  gpt-oss-20b greedy text is identical to the CPU-sink path; pp512 990 +/- 22, tg128 35.0 +/- 5.5 tok/s.
- **PrismML tile bytes** ([llama.cpp #74](https://github.com/1bit-MONSTER/llama.cpp/pull/74)): the low-token
  SwiGLU sized PQ2_0 / PTQ1_0 weight rows from a tile size of 0, so every row read row 0. Ternary-Bonsai-1.7B
  PQ2_0 decode perplexity 1,225,624 -> 20.65 (CPU 20.70); the 27B files took another kernel and were unaffected.
- **TQ1_0 / TQ2_0** ([llama.cpp #69](https://github.com/1bit-MONSTER/llama.cpp/pull/69)): ggml's ternary types
  in the shared dequantizer and on the K-quant decode kernels. Ternary-Bonsai-1.7B, HRX0 vs CPU: KLD 0.000523,
  same top token 98.67%; pp512 / tg128 TQ1_0 3542 / 113, TQ2_0 4100 / 156 tok/s.
- **MLA attention past 512 tokens** ([llama.cpp #70](https://github.com/1bit-MONSTER/llama.cpp/pull/70)): the
  V transpose assumed contiguous rows, but MLA (deepseek2, GLM) stores V with the QK head stride, so prompts of
  512+ tokens read misaligned rows. GLM-4.7-Flash Q4_K_M: PPL 315,664 -> 5.916 (CPU 5.959).
- **Hadamard files: qwen3next** ([llama.cpp #71](https://github.com/1bit-MONSTER/llama.cpp/pull/71)): `ssm_ba`
  is folded too, and a stamped file with any other unfoldable Q4_0 weight is refused by name.
- **Masked keys in flash attention** ([llama.cpp #72](https://github.com/1bit-MONSTER/llama.cpp/pull/72)):
  KV rows past a sequence's end hold whatever an earlier request or rejected draft wrote, and the F16 WMMA made
  P = 0 x V depend on that V's sign, so identical requests gave different logits (up to 0.13 on Qwen3-0.6B,
  13.5 on GLM-4.7-Flash). Masked rows now reach P x V as exactly +0; Qwen3-0.6B and Qwen3.8-27B repeat
  bit-identically. Cost: pp512 -4.7% on Qwen3-0.6B, the rest within noise.
- **A cap on the graph program cache** ([llama.cpp #63](https://github.com/1bit-MONSTER/llama.cpp/pull/63)):
  `GGML_HRX_GRAPH_PROGRAM_CACHE`, default 64 ([section above](#server-memory-the-graph-program-cache-cap-llamacpp-63)).
- **Hadamard-rotated Q4_0 files** ([llama.cpp #58](https://github.com/1bit-MONSTER/llama.cpp/pull/58)):
  llama loaded the engine's `onebit.hadamard_q4_0` ("H32") files through `llama-hadamard`; Qwen3.8-27B-Q4_0-H32
  on HRX vs BF16 (wikitext 40 x 512): PPL 6.021, KLD 0.0292, same top token 92.5%. The engine has since
  dropped the format ([above](#hadamard-rotated-q4_0-files-h32-dropped)).

Measured on Strix Halo (Qwen3-0.6B, perplexity over 8 x 512 wikitext tokens):

| File | Vulkan0 | HRX0 on AMD's commit | HRX0 with our patches |
|---|---|---|---|
| Q4_K_M | 22.53 | 22.52 | 22.52, same speed (21691 / 324 tok/s) |
| UD-Q4_K_XL | 22.43 | fails | **22.41** |
| UD-Q2_K_XL | 36.26 | fails | **36.46** |
| UD-IQ2_M | 55.44 | fails | **55.58** |

`test-backend-ops -b HRX0`: 791 OK, 0 failed (on AMD's commit, about 1150 fail).
Correct is not fast: the UD files run their IQ4_XS and sub-4-bit layers on the CPU
(UD-Q4_K_XL 6303 / 246 tok/s, UD-Q2_K_XL 974 / 76). (That was on AMD's commit; our later patches
run IQ4_XS on HRX0, "Not yet" below.)

`1bit-MONSTER/llama.cpp` also holds
`1bit/hrx2-archive`, the previous build (AMD's abandoned ggml-hrx2 plus our Q4NX
kernels). It is kept for a later port of that work to ggml-hrx. Its zaya
architecture has been ported to this branch (ZAYA1).

The engine's separate upstream llama.cpp pin for Vulkan (`third_party/llama.cpp-vulkan`) is gone
(RFC #213 stage 3). Architectures only that pin had (Zyphra Zamba, Zamba2 and BlackMamba, and a
few upstream-only ones) are refused by `1bit serve` until they are ported here; Lemonade's own
llamacpp backend serves them meanwhile. Qwen3.8-Flash-Next (`qwen4exp`, with its MTP draft head)
was ported at pin `e57beb97`. This build has no Vulkan
backend (`GGML_VULKAN=OFF`).

### The GPU's matrix units (WMMA)

The Radeon 8060S (gfx1151, RDNA 3.5) has RDNA3's six matrix instructions, all on 16×16×16 tiles.
The rates come from AMD's
[matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) (commit
`2ef9189`), for example
`uv run --with tabulate python3 matrix_calculator.py -a rdna3 -i v_wmma_i32_16x16x16_iu4 -d -w 32`:

| Instruction | Inputs | Per WGP per clock | Cycles |
|---|---|---|---|
| `v_wmma_f32_16x16x16_f16` / `_bf16` | f16 / bf16, f32 accumulate | 1024 FLOPs | 32 |
| `v_wmma_f16_16x16x16_f16`, `v_wmma_bf16_16x16x16_bf16` | same, 16-bit accumulate | 1024 FLOPs | 32 |
| `v_wmma_i32_16x16x16_iu8` | int8 × int8 | 1024 ops | 32 |
| `v_wmma_i32_16x16x16_iu4` | int4 × int4 | 2048 ops | 16 |

- int8 runs at the f16 rate. Converting activations to int8 saves memory traffic, not matrix time.
  Only int4 × int4 doubles the rate.
- WMMA does not co-execute with other vector instructions.
- In wave32, A and B are held twice, in lanes 0–15 and again in 16–31. `-A -R` (or `-B`, `-C`,
  `-D`) prints the register and lane of every element.
- This box reports 40 compute units (20 WGPs) at up to 2,900 MHz (`rocminfo`). That gives a
  theoretical peak of about 59 TFLOPS for f16 and 119 TOPS for int4. These are computed from the
  table, not measured. They bound prefill; decode is limited by memory bandwidth.

HRX's WMMA kernels (e.g. `flash_attention_decode_split_f32_f16_wmma`) load and store fragments
through the kernel compiler, which places them in these layouts. The fragment sizes match the
table: 16 f16 inputs and 4 f32 results per lane in wave64.

## Build

```bash
git submodule update --init --depth 1 third_party/hrx-system third_party/llama.cpp
cmake -B build -G Ninja -DONEBIT_HRX=ON          # needs TheRock at /opt/rocm-therock (ONEBIT_HRX_TOOLCHAIN)
cmake --build build --target onebit
```

- **One external project.** llama.cpp's ggml-hrx builds hrx-system itself
  (`HRX_SOURCE_DIR`) with TheRock's `amdclang`.
- **HRX0 and the CPU only.** `GGML_VULKAN`, `GGML_HIP` and `GGML_CUDA` are off. The targets are
  `llama-server` and `llama-bench`. With
  `-DONEBIT_SERVE_TEST_GGUF=<gguf>`, ctest runs `serve_e2e_hrx` and `serve_e2e_cpu`.
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

### Private HIP kernels

Our own HIP kernels for HRX are closed source, like the NPU kernels: they live in the private
`1bit-MONSTER/gpu-kernels` repository (`addons/hip-hrx`). The llama.cpp fork keeps only the
public HIP-through-HRX plumbing (`ggml/src/ggml-hrx/hip/`). `-DONEBIT_GPU_PRIVATE=<gpu-kernels checkout>`
(with `-DONEBIT_HRX=ON`) passes `GGML_HRX_HIP_ADDON_DIR=<checkout>/addons/hip-hrx` to the HRX
llama.cpp build, which compiles those kernels and matchers in; it needs a llama.cpp pin with
that hook. Without the option the build is unchanged. To drop the add-on from an existing
build directory, delete `build/hrx/llama` (its CMake cache keeps the setting).

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

## MoE router expert-id fault fix ([engine#123](https://github.com/1bit-MONSTER/engine/issues/123))

`llama_decode` could die with `HSA_STATUS_ERROR_MEMORY_FAULT` (`ret = -3`) on MoE
models (`Qwen3-Coder-30B-A3B`). It was reported against the decode-split
multipass path, but it is not in flash attention: the fault is a wild weight
read in `qwen3_moe_routed_gate_up_swiglu_q4k_q8`.

`router_top8_f32.loom` seeded each lane's argmax with `best_id = 0x7FFFFFFF` and
only replaced it on an ordered `ogt` or the equal-value tie-break. A lane whose
candidate logits are unordered (NaN) or below `-FLT_MAX` therefore published the
sentinel verbatim into `route_ids`; consumers treat the id as bounded via
`index.assume` (a hint, not a check) and compute `expert * expert_stride` in 32
bits, so `0x7FFFFFFF * 884736 mod 2^32 = 0xFFF28000` (~4.29 GB) walked off the
end of the weights buffer. The fix seeds the argmax with the lane's own first
expert id. Upstream PR: [1bit-MONSTER/llama.cpp#25](https://github.com/1bit-MONSTER/llama.cpp/pull/25),
merged as `895d63f`, which the engine pins.

Measured on `HRX0` (`Qwen3-Coder-30B-A3B-Instruct-Q4_K_M`, `llama-bench -p 0 -n 8`,
quiet box) with the fault made observable by aligning the three decode-split
*partial* transients to 256 B instead of 4096 B: pre-fix `-d 2100` 5/5 faults,
post-fix `-d 2100` 0/5, `-d 3000`/`-d 4800` 0/3, `<=2048` path 0/3, and greedy
output identical to the CPU backend. The alignment change is a repro stress knob
and is **not** part of the fix.

What the fix does not cover: the fault needs router logits that are NaN, and
the fix routes such a lane to a valid expert instead of faulting. Since
[llama.cpp #26](https://github.com/1bit-MONSTER/llama.cpp/pull/26) that case is loud: a router row
with no ordered logit publishes NaN, and llama-server refuses to sample NaN logits (it logs
"HRX returned NaN logits" and aborts) instead of decoding a wrong but plausible token. On the
author's repro 6 of 13 samples used to decode `' Paris???…'` silently; they now abort. Healthy
runs are unchanged (Qwen3-Coder-30B and ZAYA1-8B perplexity on HRX0 identical to before).
`GGML_HRX_FA_PARTIAL_ALIGN` (default 4096) sets the decode-split partials' alignment, a test knob
that reproduces the fault's layout; the investigation's artifacts are archived privately.
**Root cause of the NaNs ([engine#123](https://github.com/1bit-MONSTER/engine/issues/123),
[#140](https://github.com/1bit-MONSTER/engine/issues/140)), fixed in llama.cpp `00adc2b`**
([#28](https://github.com/1bit-MONSTER/llama.cpp/pull/28),
[#29](https://github.com/1bit-MONSTER/llama.cpp/pull/29)). In every decode-split `reduce_fused`
variant the last workgroup writes the attention output to global memory, and `pack_completed_q8`
then reads it back with other waves to build `next_q8`. The barrier between the two was
`kernel.barrier<workgroup>`, which fences LDS only (no store wait, no `buffer_gl0_inv`), so a pack
wave could quantize the output slot's previous contents. The f32 output was right and `next_q8`
was wrong, and NaN when the stale bytes decoded as NaN. The window opens when a dispatch is
preempted, and any process's KFD queue eviction (page compaction, KSM, memory pressure) does that,
which is why the rate followed those settings. It was never the cause. The fix fences both memory
spaces (`barrier<global>` then `barrier<workgroup>`). With a separate process forcing queue
evictions: Qwen3-Coder-30B (2113-token prompt, partial alignment 256) went from 1 of 8 correct to
8 of 8 with no NaN, and greedy Qwen3-0.6B from 13-34 divergent per ~177 requests to none.

## Not yet

- **IQ4_NL / IQ4_XS on HRX ([llama.cpp #41](https://github.com/1bit-MONSTER/llama.cpp/pull/41)).**
  AMD's IQ4 nibble packers remapped every table index (XOR 12) for a lowering that does not apply on
  gfx1151, in `motifs/dequant.loom` and again in the Q5_K/IQ4_XS prefill kernel, so IQ4 was declined
  and ran on the CPU. #41 fixes both copies and the q8-plane SwiGLU path they exposed. Qwen3.8-27B
  UD-Q4_K_XL: 91 graph splits -> 1, wikitext PPL 5.8294 (CPU 5.8288), decode 7.25 -> 9.0 tok/s
  (Vulkan 12.5); `test-backend-ops -b HRX0` passes the IQ4 MUL_MAT and GET_ROWS cases.
- **K-quant decode kernels ([llama.cpp #42](https://github.com/1bit-MONSTER/llama.cpp/pull/42)).**
  One-token projections on Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS and Q8_0 now read the GGUF blocks
  directly with f32 activations: the FFN gate/up pair fused with SwiGLU, and plain projections
  with the residual add folded in. Mixed-quant files (Unsloth UD) used to take generic kernels at
  ~105-176 GB/s; these run at ~200-224. Decode, tg128 against the Vulkan build on the same box:

  | Model | HRX | Vulkan | |
  |---|---|---|---|
  | Qwen3.8-27B UD-Q4_K_XL | 12.15 | 12.48 | 97.3% (was 9.0) |
  | ZAYA1-8B Q4_K_M | 94.0 | 93 | ~100% (was ~90) |
  | Qwen3-0.6B Q4_K_M | 326.6 | 357 | 91.4% (was 320.8) |
  | Qwen3-Coder-30B-A3B Q4_K_M | 90.6 | 93.7 | 96.6% (unchanged) |

  [llama.cpp #43](https://github.com/1bit-MONSTER/llama.cpp/pull/43) added Q3_K and IQ3_S (IQ3_S keeps
  its 512-entry grid in workgroup memory), which covers every weight type of the UD-Q4_K_XL 27B.
  Decode-shaped wikitext PPL (`-ub 1`) on the 27B: 7.1595, CPU 7.1475.
- **HRX decode no longer spins a CPU core
  ([llama.cpp #44](https://github.com/1bit-MONSTER/llama.cpp/pull/44)).** The per-token stream wait
  sat in ROCr's signal wait, which returned from the KFD event ioctl on every completed command
  (~900 per 27B token) without sleeping: a full core for the whole token, ~10 W of package power,
  and a 90-93 C APU during 27B decode. Long expected waits (over 20 ms) now sleep through 80% of
  the wait; shorter ones are unchanged. 27B decode: 71-76 C and ~125 W instead of 83-93 C and
  133-137 W, 0.4% slower; Qwen3-0.6B and ZAYA1-8B unchanged. `ONEBIT_HRX_BLOCKING_WAIT=1` turns it
  off. Model load and the first request still burst to ~88 C for a couple of seconds (host weight
  conversions and first-use GPU work share the package power limit).
- **Lossless SwiGLU for every FFN layer and MTP verify batches
  ([llama.cpp #45](https://github.com/1bit-MONSTER/llama.cpp/pull/45),
  [#46](https://github.com/1bit-MONSTER/llama.cpp/pull/46)).** The K-quant SwiGLU kernel now wins
  over AMD's int4 lowrow path (weights repacked to int4 at first use, activations quantized to
  int4), which also ran slowly under llama-server: 27B `llama-server` decode 10.7 -> 12.0 tok/s,
  KLD against CPU logits 0.00167 (AMD kernels 0.00318). #46 adds 2-8 token variants for speculative
  verify batches: Qwen3.8-27B with its MTP head (`1bit serve --mtp`, draft 3, p-min 0.75) decodes
  22.5 / 17.7 / 19.0 tok/s on code / prose / translation prompts (was 20.1 / 16.4 / 17.6; Vulkan
  34.6 / 24.3 / 24.5) with the same draft acceptance. A 4-token step still costs ~1.7x a 1-token
  step (f32 multiply-adds per token); int8 activations, as Vulkan does, are the next step.
- **NaN after an MTP rollback on Qwen3.5/3.8
  ([llama.cpp #52](https://github.com/1bit-MONSTER/llama.cpp/pull/52)).** With `--mtp`, the gated
  delta-net kernel publishes one recurrent-state snapshot per verify token so a rejected draft can
  roll back. It formed each snapshot's decay ratio c_t/c_s by dividing two exponentials. When a
  strongly forgetting head drove both to 0, that gave 0/0 = NaN, and the next verify resuming from
  that snapshot returned NaN logits ("NaN logits at vocab index 0"). It depends on the prompt, and
  showed up on Qwen3.8-27B Q8_0, UD-Q5_K_XL and UD-Q6_K files. The same bug caused an AMDGPU memory
  fault on a repeated request without MTP. The ratios are now formed in log space, as the main
  recurrence already did. Q8_0 `--mtp` now runs clean on prose and code (15.7 / 20.2 tok/s, balanced power
  mode, 85 W); UD-Q4_K_XL `--mtp` speed and output are unchanged (also measured in balanced mode).
- **`1bit serve --device hrx --parallel N`.** With one KV stream per slot, the K/V tensors are
  4-D and HRX's flash attention falls back to the CPU in every layer (Qwen3-4B, 4 slots: 39 tok/s
  aggregate, below one stream). serve now passes `-kvu` (one shared KV cache) and turns off AMD's
  Qwen attention path (`qwen.attention`), which derives its mask from positions and let the slots'
  answers bleed into each other under `-kvu`. Qwen3-4B 4 slots: 153 tok/s aggregate (single stream
  76; Vulkan 216); Qwen3-Coder-30B-A3B 34 -> 74; ZAYA1-8B 47 -> 71. Hybrid gated delta-net models
  (qwen35, qwen35moe, qwen3next: Qwen3.5/3.8) are refused with `--parallel` on HRX: there is no
  multi-sequence GATED_DELTA_NET kernel yet ([llama.cpp #48](https://github.com/1bit-MONSTER/llama.cpp/pull/48)
  fixed the two failures before it).
- **Fast sub-4-bit kernels on HRX.** IQ3_S has a decode kernel (#43) but not a fast prefill one. Q2_K
  ([llama.cpp #49](https://github.com/1bit-MONSTER/llama.cpp/pull/49)), IQ2_XXS/IQ2_XS
  ([llama.cpp #50](https://github.com/1bit-MONSTER/llama.cpp/pull/50)) and IQ3_XXS/IQ2_S
  ([llama.cpp #51](https://github.com/1bit-MONSTER/llama.cpp/pull/51)) and IQ1_S/IQ1_M
  ([llama.cpp #53](https://github.com/1bit-MONSTER/llama.cpp/pull/53)) run on HRX0: in the shared
  dequantizer (prefill WMMA, generic decode, GET_ROWS; IQ2_S was already there) and the K-quant
  decode kernels. Before them, llama.cpp's load-time buffer check (a 512-token matmul) failed, and
  every such weight went to the CPU. On Qwen3-4B, with KLD against the Q8_0 model equal to the CPU's:
  - Q2_K: 709 MiB CPU_REPACK and 217 graph splits -> all on HRX0 and 73 splits; pp512
    686 -> 1300 tok/s, tg128 47 -> 75.
  - IQ2_XXS: pp512 59 -> 715 tok/s, tg128 23.0 -> 39.5.
  - IQ2_XS: pp512 87 -> 268 tok/s, tg128 28.0 -> 30.2.
  - IQ3_XXS: pp512 2.8 -> 250 tok/s (mixed quant; 510-560 with every tensor IQ3_XXS), tg 1.5 -> 8.9
    (mixed) and 33.4 (pure), balanced power mode.
  - IQ2_S: tg 1.5 -> 11.3 tok/s (pure); its prefill is unchanged.
  - IQ1_S: pp512 70.8 -> 81.7 tok/s, tg128 14.1 -> 18.6 (balanced; the first decode run in a process
    includes about 18 s of kernel JIT for the IQ1 grid). IQ1_M: pp512 70.8 -> 80.4, tg128 14.1 -> 18.1.

  Some batched IQ2_XXS matmul shapes are still declined and run on the CPU. Mixed files decode
  slower than pure ones: a SwiGLU gate/up pair whose two formats need different grids shares one
  grid buffer, so it is declined and takes the generic path.
- **Q4NX on HRX.** Our Q4NX kernels lived in ggml-hrx2 and are not in ggml-hrx.
  They are kept on `1bit/hrx2-archive` until they are ported.
- **First-request cost.** Lemonade's telemetry for the first short HRX chat shows
  18 tok/s against llama-bench's 303. It is probably first-use kernel
  compilation, not steady state, but this is not yet separated.
- **CI.** CI has no AMD GPU and no TheRock, so `ONEBIT_HRX` is exercised on Strix
  Halo only.
