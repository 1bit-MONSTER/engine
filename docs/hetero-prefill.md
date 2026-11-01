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
- **Build a split-prefill dispatcher before the NPU GEMM is faster.** A dispatcher around a 1.5 TFLOP/s
  NPU buys ≤ 1.3 %, measured. The precondition is an NPU prefill GEMM ~4x faster: our per-tile time is
  weight-fetch bound, so a T>256 tile with weight reuse across T is the work (roughly 4x at T=1024),
  and only then does a ~15 % split at M=16384 become possible — still 0 at M≤2048.
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
- [#754](https://github.com/1bit-MONSTER/engine/issues/754) — split-prefill in the hybrid route,
  **blocked on** an NPU prefill GEMM with T>256 weight reuse; the numbers above are the acceptance bar
  it has to clear.
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

**Unrun, declared:** Qwen3-4B `q_proj` (4096×2560) and 0.6B `k`/`v` (1024×1024) — our prebuilt kernel
set has no such shapes and the mlir-aie venvs that would build them are gone on this box
(`~/ironenv142` empty, `~/iron` → a dangling link). k/v/o/x-projection shapes we did measure are the
two biggest per model (up/gate and down = 75 % of Qwen3-4B projection FLOPs). Also unrun: K-split (the
`down_proj` K-split path), decode (the paper reports it neutral; our NPU serves decode through a
different kernel), and anything end-to-end through `serve`.

Source of the analysis: `~/amd-oss/heteroMosaic` at `61714a3`, read-only, no build.
