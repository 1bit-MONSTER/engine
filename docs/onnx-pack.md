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

# The ONNX pack: weights to Q4NX, matmuls to our kernels, no VitisAI

Design note for deliverable 4. It answers one question: if a model arrives as a plain ONNX graph,
what does it take to run it on this NPU without executing a single AMD custom operator?

## What a "Ryzen AI pack" is, and why we do not execute one

The AMD Ryzen AI toolchain ships prebuilt ONNX packs: a graph whose transformer blocks are replaced
by nodes in a custom domain (`com.amd.*`), each one executed by a compiled xclbin through the VitisAI
execution provider, with weights pre-quantised into that provider's own layout. It is fast and it is
opaque: the pack decides the quantisation, the kernel, the tiling and the schedule, and the operator
that runs on the device is a black box the vendor compiles.

We do not execute those operators. Three concrete reasons, in the order they bind:

1. **Licence and footprint.** The VitisAI EP arrives in the AMD wheel, which the contract forbids us
   from linking or shipping. A pack that only runs through `com.amd.*` nodes therefore cannot run in
   our binary at all.
2. **We already own the kernels.** `npu/` generates and launches the int8 × int8 GEMM that the
   forward route uses (`ONEBIT_NPU` lane), with our own quantisation, our own tiling and a layout we
   can debug with `xrt::elf` tooling. A vendor pack would replace a measured component with an
   unmeasured one.
3. **The graph must stay a graph.** Our plugin EP's value is that ONNX Runtime schedules the graph
   and *our* provider takes the matmuls it has a kernel for; everything else stays on the CPU EP.
   A pack with fused custom nodes removes exactly that composability.

So the pack we design is the same idea, built from parts we control: **ONNX graph stays plain, weights
become Q4NX on disk, each matmul shape is bound to a kernel we generated, and the EP is the thing that
does the binding at run time.**

## What the graph gives us

Census of `qwen3-0.6b.onnx` (from `onnx/export_qwen3.py --check`, opset 21, ai.onnx domain only,
float32, seq dynamic):

| MatMul (K, N) | count | projection | kernel today |
|---|---|---|---|
| 1024 × 151936 | 1 | lm_head | none — CPU EP |
| 1024 × 3072 | 56 | gate, up | none — CPU EP |
| 3072 × 1024 | 28 | down | none — CPU EP |
| 1024 × 2048 | 28 | q | `full_i8_QKV_K1024_N2048_m16.elf` |
| 2048 × 1024 | 28 | o | none — CPU EP |
| 1024 × 1024 | 56 | k, v | `full_i8_O_K1024_N1024_m16.elf` |

254 MatMul nodes, 6 distinct shapes, zero custom domains. The binding is therefore a table lookup on
`(K, N)`, which is exactly what `ONEBIT_EP_NPU_ELFS="16x1024x2048=<elf>,16x1024x1024=<elf>"` already
expresses; a shape with no entry returns `ORT_NOT_IMPLEMENTED`, the EP falls back to the CPU for it,
and the session still runs. That is why deliverable 4 can be honest about owning 84 of 254 matmuls
rather than claiming the model.

## The three layers of the pack

**1. Weights: Q4NX on disk, int8 + scales in the kernel's layout.** The on-disk format is already
Q4NX-family (4-bit blocks with per-block scales, the format `npu/q4nx.cpp` decodes). At session start
we bind a weight tensor by pointer identity and call the runtime's own `packB`, which quantises each
*column* to int8 with `scale = amax/127` and stores the column scales alongside the packed buffer —
the same function the forward route calls, so the numerics of a packed weight cannot drift from the
weights the forward route would have used. Packing happens once per (tensor, K, N); the EP caches one
slot per bound kernel and repacks when the pointer changes.

**2. Activations: per-row int8 with `amax/127`.** Every matmul input is quantised per row, on the
device's own scale convention, and the dequant multiplies back `C = Cm * ascale[row] * wscale[col]`.
This is the forward route's exact arithmetic; the measured device error on a 1024 × 1024 matmul is
`max_abs 9.893e-02` of `max|ref| 20.414` (rel `4.846e-03`) against a 2 % gate, so the int8 path is
inside the tolerance the proof asserts.

**3. Shapes: one ELF per `(M, K, N)`, generated not fetched.** An ELF is compiled for a fixed tiling
(`_m16` here, 16 rows per launch). The pack step is therefore a *coverage* problem, not a
compatibility problem: for every `(K, N)` in the census we either have a generated ELF or we do not,
and the plan is to generate the missing ones with the existing generator rather than to reshape the
model.

| shape | owner today | next step |
|---|---|---|
| q `1024×2048`, k/v `1024×1024` | our EP | done, proven on the device |
| o `2048×1024`, down `3072×1024` | CPU EP | generate `m16` ELFs, same generator |
| gate/up `1024×3072` | CPU EP | same |
| lm_head `1024×151936` | CPU EP | needs a vocab-blocked kernel (N far past one ELF); leave on the CPU until the FFN set is proven |

## What the pack does not do, deliberately

- **KV cache: opt-in, not the default.** The default export re-runs the whole sequence per step, so its
  tokens/s is that graph's rate. `--cached` exports the same model with the cache as an input and an
  output (one stacked `[56, 1, 8, past, 128]` tensor), which decodes a 256-token prompt at 63.7 ms/token
  against 291.2 — 4.6x, and the gap widens with context. Output matches transformers running the same
  greedy loop to a mean KL of 1.2e-07 with top-1 identical at every step. What it is still not is a
  *server*: no batching, no paged cache, one sequence at a time, so its number is never put next to a
  serving stack's.
- **No fp16, no int4 activations.** The kernel is int8 × int8 → int32; the 4-bit-ness of the
  checkpoint lives on disk and is a memory-bandwidth story, not a compute one. An int4 tensor-core
  kernel would need a different ELF, and nothing here blocks that later.
- **No vendor operators, ever.** If a shape has no kernel it goes to the CPU EP. That keeps the
  failure mode boring: a slower session, never a session that refuses to load because a custom domain
  is missing.

## Where the numbers live

The measurements live in the tree, with the commands that produced them:
[onnx/BENCHMARK-PARITY.md](../onnx/BENCHMARK-PARITY.md) holds the CPU rows (ryzen and Strix Halo, five
prompt lengths, default and `--cached` exports) and the NPU row taken with the AMD `model_benchmark`
protocol through `onnx/bench_npu.sh`; the execution-provider proof (MatMul on the NPU within 0.5 % of
the CPU provider, the A8W8 QDQ pattern byte-identical) is on [ONNX Runtime](onnx.md). This note is the
design; those two pages are the evidence.

## A8W8 QDQ, the shape Quark emits

AMD's Quark ONNX-to-ONNX flow (the 2025 Ryzen AI NPU quantisation article; Linux-only) writes a
*QDQ* graph rather than a custom-op graph: every tensor that used to be float becomes

```
x_q  --DequantizeLinear(s_x, zp_x)-->  x_dq  \
                                              MatMul --> y  --QuantizeLinear(s_y, zp_y)--> y_q
w_q  --DequantizeLinear(s_w, zp_w)-->  w_dq  /
```

with int8 activations and int8 weights (A8W8), float32 scales, and the same pattern at A16W8, Xint8
(power-of-two scales), BF16/BFP16, MX4/6/9, MXFP4/MXINT8 and Int4 weight-only, using GPTQ, AdaRound,
AdaQuant, SmoothQuant or QuaRot to pick the scales. That is the good news: **A8W8 is exactly what our
device kernel already computes** — int8 × int8 into int32, dequantised by one scale per activation
row and one per weight column.

### Why it folds without a new kernel

Our GEMM is `C_s32[m][n] = Σ_k a_q[m][k]·w_q[k][n]`, dequantised as `C = C_s32 · s_a[m] · s_w[n]`.
The QDQ graph dequantises the same product as `(s_x·s_w) · C_s32` (per-tensor), or `s_x[m]·s_w[n] ·
C_s32` (per-axis, which is what Quark emits for weights and what SmoothQuant needs for activations).
Those are the same expression: our kernel's `go_rows` already applies `Cm·as[m]·gs[n]`, so a
Quark-quantised model maps onto it by *feeding* the graph's scales instead of deriving them from
`amax/127`. Nothing about the kernel changes; what changes is where the two numbers come from.

Folding the output `QuantizeLinear` is the same trick one step later: give the readback the scale
`1/s_y` and it emits int8 directly (`y_q = clamp(round(C/s_y) + zp_y)`) instead of float. The
intermediate float tensor then never exists.

### What works today, and what does not

- **Works: the group is ONE node.** `onebit_ep_qdq_proof` builds exactly that pattern with the model
  editor API (int8 in / int8 out, `DequantizeLinear` → `MatMul` → `QuantizeLinear`) and our EP claims
  all four nodes as a single fused node, so neither `DequantizeLinear` nor `QuantizeLinear` runs and
  the dequantised float tensors never exist. The int8 activations and the int8 weights go into the
  kernel as they are; the readback folds both scales and the output scale into one multiply per
  element (`y_q = clamp(round(C_s32 · s_x·s_w/s_y))`, ONNX's round-half-to-even), so int8 out comes
  straight from the integer accumulator. Measured gate, `ctest -R onebit_ep_qdq_proof`:
  int8 outputs identical to the CPU EP's, `worst |d| = 0` LSB on the 8×16×6 proof graph and also on
  a 256×1024×1024 case (262144 of 262144 values equal, so the int32-first arithmetic is in fact
  *more* exact than the CPU EP's float accumulation, which reaches the same function through two
  float tensors). Peak RSS growth of the run, measured around the run with the peak reset and
  `malloc_trim` between the warm-up and the measurement: **6404 kB for the CPU EP vs 260 kB fused**
  on 256×1024×1024 (three runs: delta 6144, 6144, 6140 kB) — 6144 kB, which is exactly `x_dq`
  (1 MB), `w_dq` (4 MB) and `y` (1 MB) of float32 that no longer get allocated. `ONEBIT_EP_QDQ=0`
  is the control: the same binary then claims the
  `MatMul` alone, logs a float `matmul` line, and the fused `qdq` line never appears.
- **Not done: per-axis scales and asymmetric zero points.** The fused path takes one scale per
  tensor: `s_x` and `s_w` are read as scalars, and a graph whose scales are per-axis vectors is left
  on the per-node path (correct, slower). Zero points are claimed only when `zp = 0`, which is what
  the proof's graph has; a non-zero `zp` is not free — with `zp_x` or `zp_w` non-zero the product
  gains the correction terms `-zp_x·Σ_k w_q[k][n]`, `-zp_w·Σ_k a_q[m][k]` and `+K·zp_x·zp_w`, which
  belong in the epilogue (column sums precomputed at weight load, row sums with the activation
  quantisation), with a proof whose graph has `zp != 0`.
- **Not done: the device leg of the fused path.** The fused node's int8 GEMM runs on the host today.
  Feeding int8 activations straight into the device kernel's input BO needs a runtime entry that
  writes `Am` from int8 and packs int8 weights without the `amax/127` quantisation step (`packB` is
  float-in), and it cannot be validated without a window on the array. Until then the fused path is a
  correctness-and-memory win, not a speed one on a real model — the shapes matter, not the pattern.
- **Different kernels, not a configuration**: A16W8 needs int16 activations, MX/BFP16 need block
  scales, and Int4 weight-only is the format our *own* forward route already stores (Q4NX), so the
  natural split afterwards is our Q4NX pack for weight-only and the Quark QDQ path for A8W8.

The proof is deliberately opset-21, domain-only, symmetric and per-tensor: it pins down the claim
(our EP takes the whole group as one node and the arithmetic is exact) for the form Quark emits by
default, and leaves the two documented gaps above for their own proofs.
