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

# Task: gated delta-net prefill on gfx1151

Optimize the HIP kernel in `gdn_prefill_kernel.py`, the prompt-processing (prefill) step of
Qwen3.8's delta-net layers. `driver.py` is the oracle and benchmark; do not edit it.

## The hardware is not an Instinct GPU

gfx1151 is the Radeon 8060S in AMD Strix Halo (Ryzen AI MAX+ 395):

- RDNA 3.5, 40 CUs (20 WGPs), **wave32**;
- no MFMA; the matrix instructions are WMMA (16x16x16, f16/bf16 at 1024 FLOP/WGP/clk, iu8 the
  same, iu4 twice that);
- 64 KB of LDS per workgroup;
- unified LPDDR5X memory at about 250 GB/s, shared with the CPU.

The MI300/MI355 advice in the knowledge base (MFMA, 64-wide waves, XCDs, HBM) does not apply.
Profile with `rocprofv3`; the MFMA counters do not exist here.

## What must not change

- **Semantics.** Keep the delta rule exactly as documented in the kernel file's docstring:
  - float32 inputs, outputs and state;
  - the transposed state layout `[H][col][row]`;
  - q/k heads shared as `h % Hk`.
- **Snapshots.** With K > 1, write the state after each of the last K tokens into slot
  `n_tokens - 1 - t` (slot 0 = final state). The engine's speculative rollback reads these slots.
- **Interface.** Keep the public entry point `gdn_prefill(q, k, v, g, beta, state, K)` and its
  return value.

## What is known

- The recurrence is sequential in tokens; the parallelism is heads × state columns (48 × 128).
- The current layout was the fastest of those measured by hand: 4 lanes per column, 32 columns
  a block, 32-token chunks in LDS. The others kept more state per lane and spilled registers, or
  reduced across more lanes.
- The K=8 case adds the snapshot writes (8 × 48 × 64 KB per call) on top of the K=1 case.

## Ideas worth trying

- **A chunked (WY / UT) form of the delta rule** that turns a chunk of tokens into small
  matrix products, which could use WMMA. The gate is 100 dB against float64, which is float32
  accuracy (the current kernel scores about 139 dB). Plain f16/bf16 WMMA inputs will not pass;
  a split-precision product (a value as the sum of two f16 parts) might.
- **A better LDS layout** for the per-chunk k/q staging (bank conflicts).
- **Overlapping** the next chunk's loads with the current chunk's compute.
