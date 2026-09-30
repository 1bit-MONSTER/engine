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

# Task: Q4_0 x Q8_1 matmul for 1-8 columns on gfx1151

Optimize the HIP kernel in `gemv_q4_0_kernel.py`. It is ggml's `mul_mat_vec_q` for Q4_0 weights
and Q8_1 activations. It is the matmul of speculative verification for Qwen3.8-27B: each step
reads all 15 GB of weights once to check up to 8 drafted tokens. `driver.py` is the oracle and
benchmark; do not edit it.

## The hardware is not an Instinct GPU

gfx1151 is the Radeon 8060S in AMD Strix Halo (Ryzen AI MAX+ 395):

- RDNA 3.5, 40 CUs, **wave32**;
- no MFMA; the matrix instructions are WMMA (16x16x16; int8 at 1024 ops/WGP/clk, int4 at 2048);
- `v_dot4_i32_iu8` is dp4a (`__builtin_amdgcn_sudot4`);
- 64 KB of LDS per workgroup;
- unified LPDDR5X memory at about 220-250 GB/s, shared with the CPU.

The MI300/MI355 advice in the knowledge base does not apply. Profile with `rocprofv3`; there are
no MFMA counters here.

## What is known

- **The floor is bandwidth.** A 5120 x 17408 Q4_0 matrix is 50 MB, about 0.2 ms at 250 GB/s.
  With 1 column the current kernel is close to that. With 8 columns it is not: in the full model,
  an 8-token batch takes about 112 ms against 80 ms for 1 token, where it should cost little more.
- **The likely cause.** The multi-column loop calls `vec_dot_q4_0_q8_1` once per column, so every
  weight int is re-loaded, re-masked and re-shifted 8 times. The engine's fork already fixed the
  same problem for another 2-bit format by unpacking the weights once and reusing them against
  every activation column.
- **Tuning that was tried.** Two warps per block was measured about 2% slower for 1 column on
  this GPU. Measure any launch-shape change on all four cases.
- **Scoring.** The 1-column and 4-column cases are in the score so that plain decode and short
  drafts do not regress.
- **First campaign (2026-09-29).** It found the row-tiled body now in the kernel file: 4 rows
  per warp, activations loaded once per k-block, weights unpacked once for all columns, and a
  reduce-scatter epilogue. 8 columns went from 0.298 to 0.226 ms, near the 0.217 ms it measured
  for a bare 50 MB read. In the full model it helps only at exactly 8 columns; 5-7 columns are
  level or slower (ROCmFPX#6). The Infinity Cache trick it rejected is noted in the kernel's
  comments. Open: 5-7 columns, which DFlash's confidence cutoff makes common.

## What must not change

- **Semantics.** Keep them exactly as the kernel file's docstring states: the block formats, the
  -8 offset applied through the Q8_1 block sum, and float32 output `[ncols][nrows]`.
- **Accuracy.** The driver checks against a float64 oracle on ordinary and adversarial inputs:
  scales over 7 orders of magnitude, all-edge nibbles, saturated activations, a zero column. The
  gate is 90 dB. Keep int8 x int4 products exact (dp4a or WMMA iu8/iu4). Do not convert
  activations or weights to lower precision.
- **Interface.** Keep the public entry point `gemv_q4_0(x, y, ncols_x, nrows, ncols)`, and support
  every ncols from 1 to 8.
