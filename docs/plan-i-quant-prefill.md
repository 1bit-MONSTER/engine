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
# Plan: prefill for the i-quant formats (engine #2832)

## What is measured, and what it costs

Same binary, same box (strixhalo, gfx1151, 122 GB UMA), Qwen3-0.6B, `-p 512 -n 128 -t 16`:

| pack | bpw | prefill tok/s | weight bytes | implied weight traffic |
|---|---|---|---|---|
| Q4_K_M | 5.09 | 16 615 | 484 MB | 16.1 TB/s |
| Q6_K | 6.56 | 16 410 | 623 MB | 20.4 TB/s |
| IQ3_XXS | 3.62 | 1 929 | 346 MB | 1.3 TB/s |
| IQ1_S | 1.56 | 2 047 | 259 MB | 1.1 TB/s |

UMA GTT bandwidth on this part is ~200 GB/s. Implied traffic above 200 GB/s is only possible if the
weight is read **once for the whole 512-token batch** and then reused out of shared memory / registers
across the row dimension. The packed formats (Q4_K, Q6_K) do exactly that; the i-quants read the weight
roughly once per row group and land at ~1.1 TB/s effective, which is the observed 8x gap.

Roofline for the same case, measured on this box:

- register-resident wmma peak, 40 CU, wave32: **51.7 TFLOPS f16**, **52.6 TFLOPS i32-from-iu8**
  (`mma_sync` on `fragment<16,16,16>`, 320 workgroups, one wave each; the two are the same rate because
  RDNA3.5's int8 wmma is also 16x16x16, it just packs 4 values per VGPR instead of 2)
- 0.6B × 2 × 512 tok = 0.615 TFLOP per prefill pass ⇒ 12 ms at peak, i.e. ~43 000 tok/s ceiling
- Q4_K_M's measured 16 615 tok/s = 20 TFLOP/s = **38 % of peak**, which is a normal number for a
  tensor-core GEMM with dequant in the inner loop

So the target is not exotic: it is Q4_K_M-class efficiency (≥ 12 000 tok/s, 3-4x of today) or better.

## Why int8, not f16, for the matrix core

The i-quants store a codebook index, not a value: IQ1_S packs two 4-bit indices per byte into a 2048-entry
grid with a per-256-block scale; IQ3_XXS similarly. Two consequences:

1. Dequantizing to f16 costs 2 bytes per value and two VGPRs per wmma operand slot; dequantizing to int8
   costs 1 byte and one VGPR slot per four values, and the accumulator stays i32 with the per-block scale
   applied once after the tile. That halves the register and shared-memory traffic of the inner loop, which
   is what the inner loop is actually made of once the weight is resident.
2. The decode lane already produces **Q8_1 activations** (`dispatch-kquant-decode.cpp:164`,
   `has_q8_alternate`), and `common.mul_mat.q4_k_q8_1_x4_prefill` is precedent for an int8 prefill row.
   Reusing Q8_1 keeps the activation path identical between decode and prefill.

This is the same shape as llama.cpp's MMQ (`ggml_mul_mat_a8`), which is what gives llama.cpp its
i-quant prefill on CUDA: int8 tensor cores, weight dequantized to int8 per block, per-block scale applied
after the tile.

## Staged implementation

### Stage A — tiled dequant-to-f16 GEMM, one weight read per batch (target: 3-4x)

Wedge into the existing loom corpus as a sibling of the packed prefill kernels:

- kernel: `common.mul_mat.iq1_s_prefill_wave32` (+ `iq3_xxs`), modelled on
  `build_q6_k_prefill_wave32_dispatch` (`dispatch-mul-mat.cpp:737`) and matching the layout of
  `match_q4_k_q8_1_x4_prefill_dispatch` (`:1048`)
- weight reads: `ggml_kquant_iq1s_lane_parts(%grid,%weight,%row_base,%block,%lane16)` and
  `ggml_kquant_iq1_slot_values(%code,%delta)` (`kquant_decode_f32.loom:7281`, `:7255`), the same readers the
  decode kernel uses, so there is exactly one implementation of the codebook
- tiling: weight tile 64 rows × 256 K per workgroup into shared memory once, then loop the token dimension
  over it (this is the step that turns 1.1 TB/s into 16 TB/s of "apparent" traffic)
- activations stay f32 for this stage; no accuracy question, no Q8_1 fusion needed
- gate: `test-hrx-` row comparing against the generic `f32_f32_wmma` route at n = 9, 64, 512 and
  k = 256, 2560, bit-exactness not required (same dequant, different order) but relative error < 1e-3,
  plus PPL parity on Qwen3-0.6B at chunk 1

If Stage A lands at ≥ 12 000 tok/s on IQ1_S, stop: the i-quants are at parity with the packed formats and
the int8 variant is an optimisation, not a fix.

### Stage B — int8 MMA inner loop (target: parity or better with Q4_K_M)

- dequant weight block → int8 + per-32 scale in registers, feed `fragment<matrix_a,16,16,16,int8_t>`
- activations: Q8_1 (already produced for decode; for prefill the graph needs the quantize node, which the
  Q4_K prefill path already arranges)
- scale application after the tile: acc(i32) × scale_weight × scale_act
- gate: same `test-hrx-` row plus PPL; Q8_1 activations change numerics, so PPL must be within 0.1 % of
  Stage A's

### Stage C — only if asked: IQ2/IQ3 nets, MoE rows, and the 27B

The 27B's `pp512 56.58` should follow the same multiplier; re-measure at the end rather than mid-plan.

### What reading the motif changed in Stage A (2026-11-18)

The Stage A bullet list above assumed the tile is a select over `%weight_format` inside the existing publish
helper. It is not, and the difference is measurable before writing any kernel:

- **the grid can be allocated where it is used.** `buffer.alloca<workgroup>` inside a `func.def inline` is
  established here (19 occurrences in the corpus, one at `motifs/mul_mat_quantized_f16_prefill.loom:609`
  inside this very motif), so `@ggml_kquant_iq1s_grid_fill` can be staged from the publish helper. It is a
  constant 4 KB codebook: fill it once at the top of `@ggml_mul_mat_quantized_f16_prefill_wave32_body`
  (beside the weight/activation stages, `:609-610`), not per epoch, and take the existing barrier.
- **the producer geometry differs, and a select would mis-stage.** Q6_K publishes two 16-value windows
  `%weight_k1 = %weight_k0 + 32` apart, stepping `%weight_k0 = %weight_group*32 + %load_packet*4` in 4s.
  `@ggml_iq1s_f16_pair` returns lanes `2p`, `2p+1`, i.e. `[32p, 32p+16)` and `[32p+16, 32p+32)` — 16 apart,
  with `p` a whole 32-value multiple. So IQ1_S needs `k1 = k0 + 16`, `k0 = %weight_group * 32`
  (`%load_packet` forced to 0), one producer per 32-value packet, and the 16-wide store arm rather than the
  `%short_weight_packets` 8-value slices. `%groups_per_epoch * %epochs_per_block = 8` already covers all
  eight packets of a 256-value block, so coverage is complete; the second producer per row rewrites the same
  values, which is redundant but not wrong — drop `%packed_threads_per_row` to 1 for the IQ1 arm once it
  measures, not before.
- **the row stride is the only place the block size enters.** `%native_block_bytes` feeds
  `%native_row_bytes = %quant_block_count * %native_block_bytes`, so 210 → 50 is safe on its own; every risk
  sits in which 32-value window each producer fills.

Order of work therefore becomes: producer arm in `@ggml_f16_prefill_publish_weights` (+ grid fill in `_body`)
→ op `common.mul_mat.iq1_s_prefill_wave32` cloned from `ops/mul_mat_q6_k_packed_f16_wmma.loom` (its
predicates `multiple_of 256/64` already fit) → engine matcher beside `match_q6_k_prefill_wave32_dispatch`
(`dispatch-mul-mat.cpp:775`) → `test-hrx-` row at n = 9/64/512 → device bench under `gpu-hold` against the
56.5 tok/s local baseline, not the shipped tree (#2858).

The corpus side of this is generator-owned where it matters: `motifs/kquant_iq1_s.loom` and the IQ1 region of
`ops/kquant_decode_f32.loom` come from `tools/generate_iq1_loom.py`, so the reader the tile calls is already
shared and pinned (engine pins hrx at the fork commit that makes a regeneration a no-op).

## Order of work, and what each step costs

1. Stage A kernel + `test-hrx` row + bench (one worktree, one PR). This is the bulk: expect the loom kernel
   to be ~200 lines plus the dispatch registration.
2. Re-run the format table (`/tmp/fmt-speed.sh` on strixhalo) and post it to #2832.
3. Stage B only if Stage A is under target.

## What this explicitly does not do

- does not touch the shipped bench tree (#2858 is a provenance problem, not a kernel problem)
- does not raise `kquant_decode.token_count`: measured, 56.44 vs 56.58 on the 27B and bit-identical PPL,
  so the decode kernel is not a prefill engine and chunking cannot make it one
- does not re-quantize or re-pack any model; the win has to come from tiling, not from format changes
