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
# HeteroMosaic's split prefill, checked on our stack

HeteroMosaic (UIUC + AMD, MICRO 2026, [arXiv 2607.12839](https://arxiv.org/abs/2607.12839),
[UIUC-ChenLab/heteroMosaic](https://github.com/UIUC-ChenLab/heteroMosaic), Apache-2.0) is the closest
published work to a direction we already have: the XDNA 2 NPU and the Radeon iGPU working on **one
prompt at the same time**. On our box (Ryzen AI MAX+ 395) they report prefill **1.17x over iGPU-only**,
decode neutral. This page is what of it transfers here, by measurement, and what we will not do.

Nothing of their tree is built or shipped. Their prebuilt NPU binaries (`hw_bins/`, `npu2`) are not
used, their ROCm 7.1.1 / kernel 6.11 / AOCL stack is not installed on this box, and no code was copied:
the harness below drives **our** kernels — ggml-hrx `MUL_MAT` for the iGPU and our own
`gemm_q4_prefill` (IRON/mlir-aie) for the NPU — with their design as the hypothesis.

## What their design is (read from the source, not the abstract)

| idea | where | what it means for us |
|---|---|---|
| **M-split prefill GEMM** | `hetero_compute.cpp:111` (`..._gemm_packed_M`), split at `:191`, disjoint row ranges into the same output, no join pass | exploitable only if the two engines' throughputs are comparable |
| **K-split** for the `down_proj` shape | `hetero_compute.cpp:282`, two pre-split weight buffers, join = `output_2d.add_()` at `:444` | same throughput condition, plus a padded reduction |
| **Ratio is a calibration table, not a formula** | `config_map` (`npuSetup.cpp:85`), keyed `(M,K,N,layer,chunk)`; seeded then swept by `py/unified_llm_w4a16/_hetero_tune.py:97-118`, `:2208-2225` | their tuned NPU share on Strix Halo is **M/8** (strixP 1/2, krackanP 7/8) — i.e. they expect NPU:GPU ≈ 1:7 |
| **One packed W4A16 layout for both engines** | 4352-byte tile = 4096 B nibble-packed 128×64 weights + 128 B bf16 scales + 128 B zeros (`hipkernels/w4a16_gemm_packed.cu:35-39`), packer `py/.../qwen25_14b_w4a16_model.py:662-793` | the single best idea in the paper for us: one weight copy in DRAM, consumed by HIP and by the NPU |
| **DRM shareable buffers** | HIP allocates and exports (`hsa_amd_portable_export_dmabuf`), amdxdna imports (`PRIME_FD_TO_HANDLE`) — `npuSetup.cpp:1913-2010`, handles in `bo_args[4]` at `:1836` | we can do this too, and we already have the export hook: `ggml_backend_buffer_export_dmabuf` (`ggml-hrx.cpp:1125`) |
| **Chunked prefill + host dispatcher** | per-chunk HIP streams (`unified_llm_w4a16.cpp:2179-2260`), NPU work in `std::async` threads gated by a 16-event pool (`hetero_compute.cpp:36-41`) spinning on `hipEventQuery` (`npuSetup.cpp:1852-1872`) | the "scheduler" is a mutex (`npu_mutex`, `npuSetup.cpp:1827`) plus hand-tuned `stage_bubbles` delays in the config; there is no coalescing thread in the code |

Two things the paper's summary gets wrong about the code, worth having straight because they change
what we would port: requests are **not** coalesced at run time (one global mutex serialises the array;
only PDI/instruction contexts are shared), and cross-engine ordering is **host-side polling of a HIP
event**, because amdxdna has no wait-on-HIP-fence. The NPU submission also rebuilds two command BOs
per GEMM call (`npu_matmul_func.cpp:325-430`).

## What we measured

Harness `~/mosaic-bench/mosaic_bench` (ours): one process, the engine's ggml-hrx `MUL_MAT` on `HRX0`
for the iGPU and our `gemm_q4_prefill` NPU kernel through XRT (T=256 rows/launch), each alone and then
both concurrently with rows split. Shapes are Qwen3-0.6B and Qwen3-4B prefill GEMMs, `M` = 256 / 2048 /
16384 prompt tokens. Every run under `npu-hold` outside `gpu-hold`, one ticket at a time. The NPU side
was validated against the kernel's own fp64 fixture first (`rel_fro=2.22e-3`, their gate 5e-3), so the
timings are of a correct kernel.

**iGPU alone** (HRX, Q4_K weights, ms / TFLOP-s):

| shape | model | M=256 | M=2048 | M=16384 |
|---|---|---|---|---|
| up_gate 2560×9728 | 4B | 0.383 / 29.1 | 2.80 / 36.4 | 20.03 / 40.7 |
| down 9728×2560 | 4B | 0.489 / 26.1 | 3.38 / 29.5 | 21.21 / 38.5 |
| up_gate 1024×3072 | 0.6B | 0.455 / 3.5 | 0.60 / 21.2 | 8.58 / 11.9 |
| down 3072×1024 | 0.6B | 0.274 / 5.9 | 0.67 / 19.3 | 4.03 / 25.6 |

Independent anchor, not my microbench: `llama-bench -p 2048 -n 0 --device HRX0` on Qwen3-4B-Q4_K_M =
**1372.8 tok/s**, i.e. 11.0 TFLOP/s for the whole prefill with attention and dequant included. The
single-op numbers above are upper bounds; the anchor is what a split would actually compete with.

**NPU alone** (our `gemm_q4_prefill`, T=256): per-tile time is flat in M — 8.61-8.66 ms for the 4B
shapes, 1.15-1.30 ms for the 0.6B shapes — so **1.47-1.58 TFLOP/s** (4B) and 1.24-1.40 TFLOP/s (0.6B),
and total NPU time is `ceil(M/256) × tile`. It is weight-stream bound: the design refetches the whole
weight column for every T-tile, which is why the per-tile time tracks K·N and not T.

**Together (M-split, wall clock, best ratio per shape):**

| shape | M | iGPU alone | best split | speed-up |
|---|---|---|---|---|
| 4B up_gate | 16384 | 20.03 ms | m=15872, NPU 2 tiles | **1.013x** |
| 4B down | 16384 | 21.21 ms | m=15872, NPU 2 tiles | 1.001x |
| 0.6B up_gate | 16384 | 8.58 ms | m=15872, NPU 2 tiles | 1.009x |
| 4B up_gate | 2048 | 2.85 ms | m=1792, NPU 1 tile | 0.32x |
| 4B down | 2048 | 3.38 ms | m=1792, NPU 1 tile | 0.41x |
| 0.6B down | 2048 | 0.67 ms | m=1792, NPU 1 tile | 0.53x |
| any measured shape | 256 | 0.16-0.49 ms | — | ≤ 0.37x |

**The result: on this box the M-split never wins.** It is +1.3 % at best, in the one regime where the
NPU's share fits entirely under the iGPU's (2 tiles = 17.2 ms under 19.4 ms of GPU work), and any
larger NPU share loses 2-12x. It is not interference — with the NPU busy the iGPU's own time does not
move (a 16384-row GEMM costs 20.03 ms alone, and the whole split took 20.06 ms while the NPU streamed
8.6 ms alongside; the NPU's 1.8 GB/s of weights is 1 % of DRAM bandwidth, and this matches the
GPU-busy NPU measurement in [docs/hybrid.md](hybrid.md), which keeps 87 % of NPU bandwidth in the
other direction). It is plain throughput asymmetry: **our iGPU runs these GEMMs at 29-41 TFLOP/s and
our NPU at 1.5 TFLOP/s, a 20-25x ratio**, so the ceiling on any split is ~3-5 % and only at M=16384.

Their 1.17x is therefore not contradicted, it is *out of reach for a different reason*: their own
tuned NPU share (M/8 on this SoC) implies an NPU:GPU ratio near 1:7 in their stack, and their iGPU
path is a rocWMMA W4A16 kernel, not hipBLASLt-class throughput. Our GPU side is the fast half; there
is nothing for the NPU to take that the GPU is slow at.

### Re-measured after the 2026-11-01 11:13 GPU reset: the sentence above is conditional on the iGPU's state

The table and the "never wins" verdict were measured on one boot (06:00-06:40 AST, before the reset).
On the boot that followed the reset, the same harness, the same NPU kernel sets and the same shapes give
a different iGPU:

| shape | M | iGPU, pre-reset boot | iGPU, post-reset boot |
|---|---|---|---|
| 4B up_gate 2560×9728 | 2048 | 2.80 ms / 36.4 TFLOP-s | 2.88 ms / 35.4 TFLOP-s |
| | 16384 | 20.03 ms / 40.7 TFLOP-s | **65.68 ms / 12.4 TFLOP-s** |
| 4B down 9728×2560 | 16384 | 21.21 ms / 38.5 | **67.51 ms / 12.1** |
| 0.6B up_gate 1024×3072 | 16384 | 8.58 ms / 11.9 | 12.39 ms / 8.3 |
| 0.6B down 3072×1024 | 16384 | 4.03 ms / 25.6 | 9.71 ms / 10.6 |

M ≤ 2048 reproduces within 3 %, and a fresh-process M-sweep puts a hard knee between M=2048
(36.9 TFLOP-s) and M=4096 (12.5 TFLOP-s), linear above it.

The knee is a dispatch rule, not an accident: the Q4_K prefill path
(`common.mul_mat.q4_k_q8_1_x4_prefill`) is capped at 256-2048 tokens, and Q4_K cannot use the
split-tail relaxation because its head binds the weight in the packed Row64 layout, so every M above
2048 rides `hip.dense_prefill.mm` instead. That alone is not a problem — on the `rc-2026-10-31c`
binaries the same GEMM at M=16384 runs at **18.6 ms / 43.95 TFLOP-s** on this very boot, i.e. slightly
better than the pre-reset numbers. The 12.4 TFLOP-s in the table above is what *current main* does
with that kernel.

That gap is a build regression, isolated by crossing the two libraries: today's `libggml-hrx` is slow
even when driving the older `libhrx`/`libloomc`, and the older `libggml-hrx` is fast even when driving
today's, with the same dispatch and the same kernel corpus (335 code objects) in all four arms. Both
sides are on llama.cpp pin `4a58da9b0`, so it is build configuration rather than source. Filed as
[engine#782](https://github.com/1bit-MONSTER/engine/issues/782); clocks (2895 MHz, 72-78 C) and the
box's GTT occupancy were ruled out, several builds each.

The consequence for the split is in the next table, and the honest reading of it is that a split's
ceiling follows whatever the iGPU is currently doing — which, on main today, is 3.5x below what the
same hardware does on the release candidate.

At that 8:1 iGPU:NPU ratio the split *does* win, and at the balance point rather than at M-256:

| shape | M | iGPU alone | best split | speed-up |
|---|---|---|---|---|
| 4B up_gate | 16384 | 65.71 ms | m=14848, NPU 6 tiles | **1.098x** |
| 4B down | 16384 | 67.66 ms | m=14592, NPU 7 tiles | **1.111x** |
| 0.6B up_gate | 16384 | 12.73 ms | m=14592, NPU 7 tiles | **1.128x** |

One formula covers both boots: balance the iGPU's per-row time against the NPU's per-256-row-tile time.
At 40.7 TFLOP-s that puts the optimum at m≈15800 and a 1.036x ceiling (what the earlier sweep saw as
1.013x at m=15872); at 12.1 TFLOP-s it puts it at m≈14600 and 1.10-1.13x (measured 1.098-1.128x).
Overlap is effectively free either way: at the optimum the wall clock is within 0.5 % of the iGPU's own
share of the rows, and `amdxdna` logged no errors across the runs.

Read together, the numbers say the split is a *ratio* question, not an NPU-quality question, and that
the ratio is not a property of this SoC alone. They also say the split is not a mitigation: 59.8 ms of
split on the degraded boot is still 3x slower than 20.0 ms of iGPU-only on the healthy one. A 10 % win
taken from a 3.3x loss is not a reason to build a split dispatcher; it is a reason to keep the iGPU
numbers of this document dated, and to re-run the ratio before anyone quotes a ceiling from it.

### Third measurement, 2026-11-06 15:09 — corrected twice: the device-tick unit (10x), and then the library it ran on (pre-#782, so the split does **not** pay after all)

> **Correction, 2026-11-06 20:5x, from the review of #1270 at `d0122cec`.** The first version of this section read
> the capture's `start_tick`/`end_tick` as nanoseconds. They are `device_tick` units: the bundle's own
> `timestamp_frequency_hz` is **99 810 000 Hz**, and the `clock_correlation` records agree to 100.0 MHz. Every
> millisecond figure in this section was therefore **10.02x too small** - which is what produced the "30x
> asymmetry", the "~1.03x ceiling" and the "~4.5x bar". Everything below is the corrected reading, reproduced from
> the same bundles with `iree-profile export --format=ireeperf-jsonl` and
> `~/lb/prefill-split/review-1270-recompute.py`. The tok/s rows and the kernel *shares* are ratio-only and never
> depended on the unit; the ms columns, the gaps line and every derived ceiling did.
>
> **Second correction, 2026-11-07 07:0x.** The unit fix stands, and so does every corrected number *for the binary it was
> taken on* - but that binary was running a **pre-#782 library**, so the conclusion drawn from it was wrong. The harness
> (`~/lb/exp50-split-ratio.sh`) sets `LD_LIBRARY_PATH=~/wt/c6-control`, which contains **only `libhrx`**; `libggml-hrx`
> therefore comes from the binary's RUNPATH, `~/wt/release-rc/work/build/hrx/llama/bin/libggml-hrx.so.0.18.0`, and that file is
> dated **2026-11-02 07:03** - four days before the #782 fix. Two device-free checks settle it: that library references
> `m128_w8_f2_sb` **24 times**, while the post-fix build at `~/.cache/hw-gate/build-13a7fda57a/bin` (2026-11-06 16:12,
> gpu-kernels `fd69590`) references it **0 times**, and `origin/main`'s `config/gpu-kernels.config` pins
> `fd69590c6c471507f6d804751f38473261436350` - i.e. **the pin already carries the fix; the installed runtime the capture
> used was stale**. The pp2048 anchor could not have caught this: pp2048 is below the M=2048 knee, which is why it is fast
> on both builds.
>
> So the 1.441 ms and 1.055 ms per 256-row group below are the **regressed** kernel selection, the "3.4-4.6x slower than the
> microbenchmark" is #782 seen through a stale install rather than a dispatch-side gap, and on a post-fix library the iGPU
> side of the ratio is the microbenchmark's own 0.29-0.31 ms per group (27-30x from the NPU) unless a post-fix **capture**
> says otherwise - which is unrun. What that restores is the page's original verdict: **on a fixed build the split does not
> pay** (ceiling ~1.03x), and #754's bar stands. The genuinely new finding of the whole exercise is the deployment one: a
> capture harness can silently measure a four-day-old runtime, and the wall clock (653 tok/s) looked plausible on both.

That is exactly what the paragraph above asked for. The 12.4 TFLOP-s boot was **engine#782**, attributed
(2026-11-06 08:26Z) to the compiler that built the embedded device code — clang-23 objects in
`rc-2026-10-31c` versus clang-24 `_sb` tiles that spill — and fixed by flipping the `m128_w8_f2_sb`
dense-prefill tiles off by default in the pinned kernels. Same box, same model as the anchor above
(Qwen3-4B-Q4_K_M), `llama-bench -dev HRX0 -ngl 99`, no-SVM trio, one ticket, die 49-66 C. **The capture is
`llama-bench`'s warmup plus one timed rep**: 64 submissions at pp16384 are 2 x 32 ubatches of 512 rows, and 8
at pp2048 are 2 x 4, so the span below covers **two** prefills:

| pp | reference (3 reps) | profiled capture (1 rep) | device span, both prefills | per prefill |
|---|---:|---:|---:|---:|
| 2048 | **1563.9 ± 87.9 tok/s** | 1 360.3 | 3 898.4 ms (59.7 % busy) | 1 949.2 ms |
| 16384 | **728.0 ± 35.1 tok/s** | 653.4 | **50 697.7 ms** (84.9 % busy) | **25 348.9 ms** |

The pp16384 span cross-checks against `llama-bench`'s own `avg_ns` in `cap-pp16384.csv` (**25 076 221 122 ns**
= 25.08 s per prefill, 653.4 tok/s), so the corrected scale is not a judgement call.

The pp2048 anchor is the comparable one (the document's own is **1372.8 tok/s** on the rc31c binaries),
so today's binary is **13.9 % faster** than the anchor the "healthy boot" table was taken on: the
regression is gone.

**The per-shape number the ceiling actually needs**, read off the pp16384 capture rather than a
microbenchmark (`HRX_PROFILE_MODE=dispatch`, 36 752 dispatches in 64 submissions = two prefills):

| kernel | device time | share of busy | dispatches | per dispatch |
|---|---:|---:|---:|---:|
| `fa_prompt_f16_d128` (attention) | 20 870.6 ms | **48.5 %** | 2 304 | 9.058 ms |
| `ggml_mul_mat_q4_k_f16_wmma_prefill_wave32` | 12 504.7 ms | 29.0 % | 9 216 | 1.357 ms |
| `ggml_mul_mat_swiglu_q4_k_f16_wmma_prefill_wave32` | 6 457.3 ms | 14.9 % | 2 240 | 2.883 ms |
| `ggml_mul_mat_q6_k_f16_wmma_prefill_wave32` | 2 854.7 ms | 6.6 % | 2 240 | 1.274 ms |

`swiglu_q4_k` is the `up_gate` shape, and the counts are per **two** prefills: 2 240 dispatches is 1 120 per
prefill, i.e. **one per layer per ubatch** (32 ubatches x 35 layers), not two. One dispatch covers 512 rows of
**one** half of the fused pair, so the comparable unit is **2.883 x 256/512 = 1.441 ms per 256-row group**.

A second reading from the same capture agrees with that within 30 %: the three weight-matmuls total
**21 816.7 ms for two prefills** = 10 908.4 ms per prefill, and a 4.02 B-parameter model's weight matmuls at
16 384 tokens are 2 x 4.02e9 x 16 384 = **131.8 TFLOP**, i.e. **12.1 TFLOP-s** in the engine - which is
**1.055 ms** per 12.75-GFLOP 256-row `up_gate` group.

Either reading puts **that build's** iGPU at **1.06-1.44 ms per 256-row group** against the NPU's **8.61 ms**
per tile: an asymmetry of **6-8x** - and it is **3.4-4.6x slower than the microbenchmark this page's first two
measurements used** (0.313 ms per 256-row group, 40.7 TFLOP-s). Reading that as a "loop-to-engine gap" was my
first conclusion and it was wrong: the capture ran on the pre-#782 library (second correction above), and the
post-fix library does not contain those tiles at all. **A post-fix capture is unrun**, so what is known is the
microbenchmark's rate on the post-fix build and nothing about the engine's own prefill on it - which is the one
measurement this page still owes.

**What that does to the bar**, stated for both builds so neither can be quoted alone. On the **pre-#782 library
the capture ran**: iGPU 1.441 ms per 256 rows, NPU 8.61 ms per tile, 64 row-groups at M=16384, iGPU-only =
92.2 ms -> 9 row-groups on the NPU, both sides at 79.3 ms, **1.164x** (1.120x on the 1.055 ms reading), and
**1.650x** at #754's 2.15 ms/tile bar. On the **post-fix pin** (`fd69590`, the microbenchmark's 0.313 ms per
group, iGPU-only 20.03 ms): 2 row-groups -> **1.032x**, and **~1.12x** at the same 2.15 ms bar. The second row
is the one the shipped engine is entitled to, so the page's bar stands: a per-GEMM 15 % needs the NPU near
1.9 ms per tile (~4.5x), and less end to end because attention is 48.5 % of the cell. M<=2048 stays unresolved:
1.00x / 1.143x on the pre-fix build, and no measurement on the post-fix one.

**And the model where our prefill really is slow is not an NPU case either.** Measured the same day
across the models the Supreme board compares (`~/lb/prefill-split/RANKED-FOR-SUPREME.md`): the worst
prefill on the box is `bonsai2-27b-ternary` at **144-152 tok/s**, and its device time is **93-96 %** in
three **f32** wmma kernels — a 1-bit model doing f32 prefill math. That is ~7.8 TFLOP-s of effective
work, from wall clock rather than the capture, so the unit correction does not touch it.

### The scheduling half, which the ratio alone does not cover

The measurements above are per-GEMM ratios. A split has a second requirement the paper's design hides:
the iGPU side must be able to run *while* the NPU side runs, and our iGPU cannot fill its own gaps.
From the same capture and the lane's prefill work of 2026-11-06:

- **Submissions never overlap on the iGPU**: 0.00 % of busy time has two dispatches in flight from
  different submissions, in every one of 24 prefill captures across six models — consistent with
  [docs/hybrid.md](hybrid.md) and with the queue-wide AQL barrier a wait edge inserts. So a split cannot
  be "one engine taking the other's idle": on the iGPU side, the idle is *host* time between
  submissions (**94.6 %** of this arm's 15.1 % idle sits in gaps > 200 us at pp16384, **96.0 %** at
  pp2048; the first version of this bullet said 57.3 %, which was `exp34-gaps.py` comparing ticks against
  a 200 000-**tick** threshold and calling it 200 us - 200 000 ticks is ~2 ms).
- **The critical path is the slower side, per row-group.** At the corrected 6-8x per-row asymmetry, the
  NPU's first tile costs 8.61 ms to save 1.06-1.44 ms of iGPU time, so the split only pays once ~6-9
  groups are on the NPU at once - which is why the optimum is 7-9 of 64 groups and not "a tile or two",
  and why the 1.12-1.16x ceiling is the whole of what a perfect overlap could buy.
- **The join is free by construction, and that part does transfer**: disjoint row ranges into one output
  tensor means no reduction pass (their `hetero_compute.cpp:111`, our equivalent is the same), which is
  why the ratio — not the join — is the whole question.

The verdict of this page is therefore **back where it started, and now for two reasons rather than one**: on a
build that carries the pinned kernel fix, the split's ceiling is **~1.03x** (2 of 64 row-groups) and the bar in
[#754](https://github.com/1bit-MONSTER/engine/issues/754) stands. The intermediate reading - "the split pays
1.12-1.16x, the bar is met" - was a **stale-library artefact**, not a dispatch gap: the capture ran on the Nov-2
`libggml-hrx`, which still contained the `m128_w8_f2_sb` tiles the pin had already turned off. Two things do
survive as findings: the **unit correction** (device ticks are 100 MHz, so that page's ms column was 10x out),
and the **deployment trap** - a harness can measure a four-day-old runtime and produce a plausible wall clock
(653 tok/s) while doing it. Numbers above: one hold, `~/lb/prefill-split/qwen3-4b/` (bundles, CSVs, both
stderr logs), harness `~/lb/exp50-split-ratio.sh`, recompute `~/lb/prefill-split/review-1270-recompute.py`,
and the library evidence is `strings <lib> | grep -c m128_w8_f2_sb` (24 on the capture's, 0 on the pin's).

## What transfers, and what we will not do

**Transfers (worth doing regardless of the split):**

- **One packed weight layout both engines read.** This is the idea with real value here, and it is
  also the answer to the hybrid route's biggest waste — the model resident twice (`docs/hybrid.md`,
  `~/briefs/TZEROCOPY-STATE.md` row 12: a 15 GB GGUF plus a 16 GB `model.q4nx`). Their tile is a flat
  `uint8` buffer with the scales and zeros inline, so a HIP kernel and an NPU instruction stream can
  both consume the same bytes with no repack. Ours are two packings today (ggml Q4_K blocks for HRX,
  Q4NX tiles for the NPU kernels).
- **The DRM buffer bridge.** Measured, works, and we already own half of it (next section).
- **The dispatcher shape** — one HIP stream per in-flight chunk, NPU work on its own thread, ordering
  by a recorded event — is what our hybrid route would need anyway, and it does not need their code.

**Will not do:**

- **Port their NPU binaries or their stack.** Prebuilt `.pdi`/instruction blobs are theirs to ship, not
  ours to run; our NPU route is our own full-ELF kernel sets (docs/npu.md). Their ROCm 7.1.1 / kernel
  6.11 / AOCL pins are not this box's.
- **Quote a ceiling from a runtime you have not identified.** That is the real lesson of the two corrections
  above: the first was a unit error, the second a library four days older than the pin, and either one alone
  changes the answer by 3-10x. Print the library's path, mtime and kernel-generation marker next to the
  number (`ldd` plus `strings … | grep -c m128_w8_f2_sb` is enough), the way the capture already prints the
  model, the pin and the die temperature.
- **Copy the weights to give the NPU its own layout per layer.** Measured cost: 50 MiB of copies per
  layer for the 4B up_gate (~3.6 ms at 14 GB/s) against a 1.3 % best-case gain.

## The DRM buffer, measured

`~/mosaic-bench/dmabuf_test.cpp`, 64 MiB, under `npu-hold` + `gpu-hold`:

| step | result |
|---|---|
| HRX allocation → `ggml_backend_buffer_export_dmabuf` → dma-buf fd | fd in **0.044 ms** |
| amdxdna `PRIME_FD_TO_HANDLE` + `GET_BO_INFO` + `mmap` | **0.206 ms** (0.090 ms for a plain `hipMalloc`) |
| iGPU writes → the amdxdna mapping reads, and the reverse | **MATCH both ways** over 64 MiB, one buffer |
| bandwidth through the shared mapping | **29.3 GB/s** CPU read, vs **14.0 GB/s** for `hipMemset` + `hipMemcpy` D2H |
| NPU-allocated BO → dma-buf (`xrt::bo::export_buffer()` + `PRIME_HANDLE_TO_FD`) | **fails** (XRT's own handle, ENOENT) — only iGPU-allocated buffers are shareable |

So the sharing half of their design is available to us today with our own export hook, and it is 2x
the copy path. What is missing is **execution on the imported buffer**: our NPU kernels go through
XRT, which allocates its own BOs and cannot bind a foreign dma-buf handle as a kernel argument.
HeteroMosaic executes on imported handles only because it submits to amdxdna by raw ioctl
(`npuSetup.cpp:1913-2010`), with no XRT in the path at all.

## Follow-ups filed

- [#753](https://github.com/1bit-MONSTER/engine/issues/753) — dma-buf import for the NPU submission
  path: XRT cannot bind an imported BO, so today only iGPU→NPU sharing is possible, and only for
  buffers the NPU cannot yet execute on.
- [#754](https://github.com/1bit-MONSTER/engine/issues/754) — split-prefill in the hybrid route, **blocked
  on** an NPU prefill GEMM with T>256 weight reuse; on a build that carries the pinned kernel set the
  ceiling is ~1.03x, and the 4x bar buys ~1.12x per GEMM. What the thread's last round did add is the
  measurement discipline: the pre-fix-library reading (1.12-1.16x) is reproducible and wrong for this box,
  and `tools/split_ceiling.py --gpu-tile-ms 0.313 --npu-tile-ms 8.61 --rows 16384` is the post-fix number.
- `docs/hybrid.md`'s "Running beside the GPU" section already covers the other direction (an NPU stream
  under a busy GPU keeps 87 % of its bandwidth); this page adds the GPU-side view: the iGPU loses
  < 0.2 % when the NPU runs beside it.

## Reproducing

```
~/mosaic-bench/mosaic-run env ITERS=3 ./sweep.sh        # alone: HRX, NPU, first split sweep
~/mosaic-bench/mosaic-run env ITERS=3 ./sweep2.sh       # split at the ratios that matter
~/mosaic-bench/mosaic-run ./dmabuf_test --mode hip      # DRM bridge, HIP allocation
~/mosaic-bench/mosaic-run ./dmabuf_test --mode hrx      # DRM bridge, HRX allocation (engine hook)
```

`mosaic-run` sets what HRX needs on this box and takes the holds (`HOLDS=gpu|npu|both`). On the
rebuilt box the 2026-10-01 `~/.local/share/1bit-canonical` install cannot enumerate `HRX0` at all
(`hrx_gpu_initialize` → NOT_FOUND); the rc31c libraries plus the TheRock `libhsa-runtime64.so.1` path
can, and that is what the numbers above used.

The third measurement's iGPU numbers come from the engine binary instead of the microbenchmark, which
is why they cannot be skewed by the harness: `~/lb/exp50-split-ratio.sh` (reference row plus a
`HRX_PROFILE_MODE=dispatch` capture per prompt length; `~/lb/exp34-gaps.py` and `~/lb/exp37-offline.py`
read the captures). One caveat, because it is easy to misread: a single-rep capture includes the
per-shape first pass, so its idle share at M=2048 (40.3 %) is host time in the prepared-program
cache-miss path, not steady state.

**Unrun, declared:** Qwen3-4B `q_proj` (4096×2560) and 0.6B `k`/`v` (1024×1024) — our prebuilt kernel
set has no such shapes and the mlir-aie venvs that would build them are gone on this box
(`~/ironenv142` empty, `~/iron` → a dangling link). k/v/o/x-projection shapes we did measure are the
two biggest per model (up/gate and down = 75 % of Qwen3-4B projection FLOPs). Also unrun: K-split (the
`down_proj` K-split path), decode (the paper reports it neutral; our NPU serves decode through a
different kernel), and anything end-to-end through `serve`.

Source of the analysis: `~/amd-oss/heteroMosaic` at `61714a3`, read-only, no build.
