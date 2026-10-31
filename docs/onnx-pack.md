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
  `malloc_trim` between the warm-up and the measurement: the fused side is stable at **248-268 kB**,
  the CPU EP side is **6.4-7.3 MB** on 256×1024×1024. Both are ranges on purpose — the CPU EP figure
  is that process's arena growth, not a tensor size, so it moves run to run (eight runs on this tree,
  deltas over each run's own starting point: 6156, 6432, 6144, 6136, 6144, 6368, 7004, 6524 kB). What
  does not move is the gap's stable part: 6.1 MB, exactly the 1 MB `x_dq` + 4 MB `w_dq` + 1 MB `y` of
  float32 that no longer get allocated; the rest is the CPU EP's own arena churn.
  `ONEBIT_EP_QDQ=0` is the control: the same binary then claims the
  `MatMul` alone, logs a float `matmul` line, and the fused `qdq` line never appears.
- **Works: per-axis scales, in the two forms the epilogue can fold.** `s_w[n]` (the weight's `axis 1`,
  one scale per output column — what Quark emits for weights and what SmoothQuant needs) and `s_x[m]`
  (the activation's row axis, one scale per token) are read as vectors instead of scalars and land in
  the same epilogue: `y_q[m][n] = clamp(round(C_s32 · s_x[m]·s_w[n]/s_y))`. Measured, same gate as
  per-tensor: `ctest -R onebit_ep_qdq_proof` reports `worst |d| = 0` LSB for the per-axis case
  (8 rows × 6 columns of scales, trace `sx=8 sw=6`) exactly as for the per-tensor one. Both proofs
  include a refusal: a per-axis scale along **K** cannot fold into an int8 × int8 epilogue at all
  (it weights the individual products inside the sum, which is SmoothQuant's job at export time), so
  the case gates on the EP *not* claiming the group — `claimed=1, qdq_groups=0`, the per-node path,
  and the same int8 output.
- **Works: asymmetric zero points, folded into the same epilogue.** Writing the shifted product out,
  `Σ_k (a-zx)(w-zw) = Σ_k a·w − zx·Σ_k w − zw·Σ_k a + K·zx·zw`, the GEMM already computes the first
  term, the second is a **column sum of the weight** (computed once on the first run that needs it and
  kept for the session — never per step), and the third is a **row sum of the activation** (one extra
  add per k while the row is already being read). The output zero point is added after the
  round-half-to-even, in that order, because rounding after an integer shift would move exact ties to
  the wrong neighbour. All three are per-tensor or per-axis along the same axes as the scales, and the
  corrections are exact int64. Measured, `ctest -R onebit_ep_qdq_proof`: `worst |d| = 0` LSB against
  the CPU EP for `zp_x = −40, zp_w = −25, zp_y = −7` with per-tensor scales, **and** for the fully
  per-axis case (8 row scales and 8 row zero points, 6 column scales and 6 column zero points, plus
  `zp_y`); the EP trace for those runs reads `zp_x=8 zp_w=6 zp_y=-7` with a one-off
  `qdq_colsum k=… n=…`. A symmetric graph never computes the column sums (its trace reads `zp_x=1
  zp_w=1 zp_y=0` and no `qdq_colsum`), which is the point of doing it lazily.
- **Not done: a zero point along K.** Like a scale along K it would weight the products inside the sum
  — it needs `Σ_k zp_x[k]·w_q[k][n]`, a column-weighted vector of the weight that no kernel here
  precomputes — so a graph with one keeps the per-node path.
- **Not done: the device leg of the fused path.** The fused node's int8 GEMM runs on the host today.
  Feeding int8 activations straight into the device kernel's input BO needs a runtime entry that
  writes `Am` from int8 and packs int8 weights without the `amax/127` quantisation step (`packB` is
  float-in), and it cannot be validated without a window on the array. Until then the fused path is a
  correctness-and-memory win, not a speed one on a real model — the shapes matter, not the pattern.
- **Different kernels, not a configuration**: A16W8 needs int16 activations, MX/BFP16 need block
  scales, and Int4 weight-only is the format our *own* forward route already stores (Q4NX), so the
  natural split afterwards is our Q4NX pack for weight-only and the Quark QDQ path for A8W8.

### A real Quark export, through the EP

`onnx/quantize_quark_a8w8.py` runs AMD's Quark ONNX flow (source tree, `release/0.13`) on
`qwen3-0.6b.onnx`, calibrating on 32 x 512 tokens of local text (no dataset download). Quark writes the
`QuantizeLinear`/`DequantizeLinear` pair in the **com.microsoft** domain, which the EP accepts as the
same operator; it also quantises only what it is told to, so `--matmul-only` keeps attention's
QK^T/AV (which have no constant weight) in float.

On a 0.6B A8W8 export with per-channel weights, the graph ORT hands the EP (4169 nodes, ORT_ENABLE_BASIC
so the DQ/Q pairs survive) partitions like this:

| | count |
|---|---|
| `DequantizeLinear -> MatMul -> QuantizeLinear` groups fused into one node each | **197** (all the projections: 28x7 + lm_head) |
| ONNX nodes those 197 groups replace | 788 |
| `MatMul` left on the CPU EP | 57 (28 QK^T + 28 AV, whose activation is rank 4, + 1 rotary matmul whose weight is a `Cast`) |
| everything else on the CPU EP | the remaining 3381 nodes: activation Q/DQ, LayerNorm, softmax, rope, elementwise |

Measured, single-threaded, `1bit ort-run --basic` on the same 5-token prompt (`--dump-logits`), 16
tokens generated:

| run | tokens/s | ms/token | KL vs float (prefill logits) | top-1 |
|---|---|---|---|---|
| float32 graph, CPU EP | 7.39 | 135 | — | 12095 (" Paris") |
| A8W8 per-channel, CPU EP | 1.43 | 699 | 2.19 nats | 9625 |
| A8W8 per-channel, our EP | **2.09** | 479 | 2.81 nats | 279 |
| A8W8 per-tensor, CPU EP | 3.09 | 323 | 6.49 nats | 279 |
| A8W8 per-tensor, our EP | 2.06 | 486 | 6.49 nats | 279 |

Three things that table says, none of them flattering and all of them worth knowing:

1. **The fused path is 1.46x the CPU EP on the per-channel export and 0.67x on the per-tensor one.**
   Per-channel `DequantizeLinear` is expensive on the CPU EP (it broadcasts a vector), so removing the
   whole DQ -> float MatMul -> Q sequence pays off there; per-tensor DQ is cheap and MLAS's float
   MatMul is faster than this host int8 kernel. This is a host-kernel measurement: the device leg is
   where the 1.5x-per-projection argument in this document actually lands.
2. **The quantisation, not the EP, is what costs accuracy.** KL 2.2-2.8 nats is my calibration's
   fault (32 documents, MinMax, no AdaRound/QuaRot/SmoothQuant), and the per-tensor export (6.5 nats)
   is worse still: one int8 scale per weight matrix is not enough for this model. AMD's own recipe is
   the one that produces a usable model; this export exists to exercise the EP.
3. **The Python `probe_dump` comparison is the honest correctness gate, not bit-identity.** On the
   per-tensor export the two paths are bit-identical (KL 0.000 between them, 17 of 17 decode steps
   the same). On the per-channel export the fused int8 kernel and the CPU EP's DQ -> float MatMul -> Q
   agree exactly on the first 38 projection tensors and then differ on **1 value in 5120 (0.02 %)**:
   the accumulator is -3090, the exact value is -3.499998 and rounds to -3, while the CPU EP's float32
   accumulation lands on exactly -3.500000 and rounds half-to-even to -4. Both are right for their own
   arithmetic; the difference then cascades, because every layer requantises to int8 and there is no KV
   cache to damp it. That is why the fused path's KL is 2.81 and the CPU EP's 2.19 on the same graph,
   and why the proof's gate is bit-exactness on graphs where the values are not on a boundary.

The EP's debug knob for exactly this work is `ONEBIT_EP_QDQ_DEBUG=1`: it logs, per graph, why each
`MatMul` was *not* fused (`qdq_refused why=... count=...`). Without it "0 groups" is not an answer.

The proof is deliberately opset-21 and domain-only. It pins down the claim (our EP takes the whole
group as one node and the arithmetic is exact) for the per-tensor, per-axis and asymmetric forms
Quark emits by default, and leaves the one documented gap above — the device leg — for its own proof.
