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
# Zero-copy on the serving paths: what moves, what it costs, what stays

Strix Halo has one DRAM. The CPU, the iGPU (HRX over KFD/GTT) and the XDNA 2 NPU (its buffer objects are
host memory mapped through the IOMMU) all read and write it, so a `memcpy` between "host", "GPU" and "NPU"
buffers on this box is pure overhead — the owner's framing is *"we need to be using it, the whole point of
having it"* (engine#698). This page is the map and the audit: which copies are left on the serving paths,
what each one costs per token, and why it is still there.

Every number below was measured on this box, and each names what measured it. Nothing here is an estimate
unless it says so. The audit is the deliverable of engine#698's target 4; targets 1-3 are the code work and
their status is at the top.

## Status

| # | target | state |
|---|---|---|
| 1 | hybrid KV handoff without the file | **shipped**, opt-in: `ONEBIT_KV_SHARE=1` ([docs/hybrid.md](hybrid.md), "The shared route"). 0 bytes copied, 0.05-0.26 ms, flat in prompt length |
| 2 | NPU runtime: activations live in BOs | **not done**. Measured price is ~4 % of a token on the 27B NPU path, and the two things that decide it are the M=128 pad and the XDNA cache model — both NPU-runtime work, not a transport fix |
| 3 | ONNX EP IO binding onto BO memory | **not done**. Priced, not measured per-call here; ORT IO binding exists and is unused |
| 4 | this audit | **this page** |

## Target 1 — the hybrid handoff moves no bytes

The file route writes the provider's sequence state to disk and the decoder reads it back through the page
cache, for a handoff that never leaves the box. `ONEBIT_KV_SHARE=1` replaces it with dma-buf fds: the decoder
exports its KV region, serve passes the descriptors to the provider as inherited fds, the provider prefills
into that region, and the decoder adopts the cells with no data movement (`llama_kv_cells_adopt`). The
provider contract, the flag semantics and the fallbacks are in [docs/hybrid.md](hybrid.md); what this page
adds is the price.

| route, same box, same model | handoff | what it scales with |
|---|---|---|
| state file, 27B hybrid (Qwen3.5): 307 / 4 096 / 16 384 prompt tokens | **28.82 / 61.82 / 198.53 ms** | prompt KV (65 536 B/token) **plus** ~151 MB of GDN recurrent state on every handoff |
| state file, 30B-A3B (full attention): 307 / 4 096 / 16 384 | **8.22 / 68.29 / 242.86 ms** | prompt KV only (98 304 B/token), no constant term |
| shared region, same shapes | **0.048-0.258 ms, 0 bytes** | nothing — flat in prompt length |

The shared figure is 256 cells in 0.048 ms, 768 in 0.107 ms, 2 048 in 0.25 ms: cells bookkeeping, not bytes.
Measured with a CPU prefill provider standing in for the NPU one, greedy text identical to `--device hrx` on
every arm — the gate for this target is text identity, not a tolerance. Two properties that make it usable
rather than a demo: a decoder that refuses to export now falls back to the state-file route for that request
(engine#1203) instead of losing the handoff, and a decoder that *cannot* export (any backend but `ggml-hrx`:
the CPU, for instance) starts normally and warns once instead of dying at load (fork#212). `ONEBIT_KV_SHARE`
is read as "set and non-empty" on both sides: `=0` still enables the route, an unset variable is the opt-out.

What is **not** proven: the NPU is not yet the provider that writes that region (the provider in the gate
above prefills with the CPU). That needs an NPU-side K/V writer — kernel work in the private tree, tracked in
engine#753, which is parked until that writer exists.

## Target 2 — the NPU runtime's copies, measured

Per launch on the shipped int8 path the runtime memsets and quantises one row into the activation BO, flushes
it, waits, invalidates the result BO and dequantises row 0 back to host floats. On the 27B (Qwen3.5 hybrid,
64 layers, 48 of them GDN, H=5120, intermediate 17 408, vocab 248 320, M=128 int8 kernels) that is:

| | measured | inside |
|---|---|---|
| A stage: 128-row zero pad + scalar quantise of one row | **50 ms/token** | `[ph]` "A quantise+launch", 1.33 s/token build |
| C invalidate + dequantise | **9 ms/token** (≈0.1 ms per launch, 368 launches) | `[ph]` "C readback" |
| same two rows on the live 400 ms int8 token | quant+launch **15-20 ms**, readback **~1.5 ms** | `[prof]` |

So the host-side copies are **~1.5 %** of a 1.33 s token and a few percent of the live one. That is the honest
ceiling for this target as it is written: the money is in the 128-row pad (the shipping container registers
**0** kernel roles at M=16, so the pad is 8x larger than the work) and in the launch count, not in the memcpy.

Two facts from measuring the readback that decide how a zero-copy version must be built:

- `bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE)` is a real cache sweep (`_mm_mfence; clflush` per 64 B, then
  `_mm_mfence` — disassembled from the shipping runtime), not a copy. It is already paid on every launch.
- A dirty host cache line over the result BO **clobbers the NPU's output**: poisoning the row before the
  launch and not writing it back leaves the sentinel to be evicted over real results, which reads exactly like
  the dropped-write fault in engine#666. Any "skip the flush" arm has to respect that ordering.

Nothing here needs the file: this is all in-BO traffic.

## Target 3 — the ONNX execution provider's boundary

The EP reads and writes `OrtKernelContext` in place inside its own host nodes, and copies at the node boundary
(`DT_CopyTensors`, `onnx/ep/onebit_ep.cpp:206`): per fused QDQ GEMM node, an `M x K` f32 input and an `M x N`
f32 output — 20 KB + 0.6 MB per node for a 0.6B decode step at M=1. ORT's IO binding would let inputs and
outputs live in BO-backed `OrtValue`s and remove the boundary copies entirely; it exists in the ORT version
this tree builds against and nothing calls it. Not measured per-call on this box: the numbers above are the
bytes, and the per-call cost is what target 3 owes.

## The audit: every copy still on the serving paths

27B Qwen3.5 hybrid on the int8 NPU path, M=128, unless the row says otherwise. "Bytes/token" counts what is
read and written per decoded token; a row marked *no memcpy* is arithmetic over memory rather than a transfer.

| # | copy | where | bytes/token | ms/token | why it stays |
|---|---|---|---|---|---|
| 1 | A stage: `memset` the 128-row pad + int8-quantise one row, straight into the mapped BO | `npu/forward/engine-src/npu_engine_i8ctx_inc.h` `quantize_async` | 347 MB memset + 10.6 MB write, no memcpy | **50** | the pad, not the copy: an M=16 twin exists in the code but the shipping container has no M=16 roles. NPU-runtime work (target 2) |
| 2 | A flush: `sync(TO_DEVICE)` per launch | same file | 347 MB range swept | inside the 50 | cache writeback, not a copy; required while the CPU writes A and the array reads it in the same launch. Removing it needs the "who wrote it" analysis in target 2 |
| 3 | C invalidate: `sync(FROM_DEVICE)` per launch | `readback()` | 2.07 GB range swept | **9** (0.1 ms/launch) | cache sweep, not a copy; already paid. Cheap per byte, so not where the money is |
| 4 | C dequantise: int32 row 0 of C to a host float vector | `dequant_only` | 16.2 MB read + 16.2 MB written | inside the 9 | the next launch's A could be filled from C row 0 without a float round trip — that is target 2's "consumed in place" |
| 5 | LM head: 15 packed tiles, 15 launches, an invalidate per tile | `run_lm_head` | 1.0 MB of logits to host | **180** (`[ph]` "head") | the per-token 356 MB f32 tile memcpy is already gone on the shipped path; what is left is the device streaming the int8 head, not a copy |
| 6 | GDN recurrent state swept on the host, in place, per layer | `gdn_layer`, `rec_state_[l]` | **604 MB** read+written (48 layers x 3.14 MB x ~4 passes) | part of `[ph]` "host-inside-layers" **190** | the largest copy-shaped cost on this path and no zero-copy fixes it: a recurrence has to run somewhere. Removing it moves the recurrence into an NPU kernel — kernel work, private tree |
| 7 | QKV rope per-head row staging | `attention()`, `step()` | ~0.4 MB | ~0.3 | per-head rope needs contiguous rows; the copies are L1-sized |
| 8 | hybrid KV handoff, file route | `app/serve_hybrid.h`, `llama_state_seq_save_file` + `?action=restore` | 65 536 B/token + 151 MB once (27B) or 98 304 B/token (30B) | **28.8 / 61.8 / 198.5** at 307/4k/16k (27B) | replaced by target 1 wherever both sides can share; kept as the fallback for a decoder that cannot |
| 9 | KV cell adopt after the handoff | the pinned fork's `src/llama-kv-share.cpp` (`llama_kv_cells_copy`) | **0** | **0** | already zero copy: metadata only |
| 10 | ONNX EP node boundary | `onnx/ep/onebit_ep.cpp` (`DT_CopyTensors`, `:206`) | `M x K` in + `M x N` out per node | not measured here | target 3 |

Two conclusions the table forces, written down so they are decisions rather than leftovers:

- **Rows 1 and 3 total ~59 ms of a 1 330 ms token** on the measured build (rows 2 and 4 are inside them, so
they do not add): zero-copy on the NPU runtime is worth a few percent there, not the 20 % it sounds like. The lever that pays is the number of
  launches and the M=128 pad, and both are NPU-runtime work.
- **Row 6 is the biggest single cost and no memory recipe removes it.** A GDN recurrence is host arithmetic
  over 604 MB per token; only moving the recurrence into a kernel changes the number.

## Sharing a buffer between the NPU and the iGPU: what works, and the one rule

Two directions, both measured on this box (engine#753 and engine#812 carry the detail and the harnesses):

| step | result |
|---|---|
| HRX/KFD allocation exported as a dma-buf | fd in **0.044 ms** |
| import into amdxdna (`PRIME_FD_TO_HANDLE` + `GET_BO_INFO` + mapping) | **0.206 ms** (0.090 ms for a plain device allocation) |
| one buffer, iGPU writes / NPU reads and back | **MATCH both ways** over 64 MiB; **29.3 GB/s** through the shared mapping against **14.0 GB/s** for copy-then-read |
| NPU executes on an imported range, addressing the **exporter's** mapping | **byte-identical** to the same kernel on its own BO (one output hash over 4 runs / 12 arms) |
| NPU executes on an imported range whose address comes from **mmap of the import through amdxdna** | **fails**: `Populate ranges failed, ret -14` (EFAULT) at the next submit, and because that step is client-wide, every later submission in the process fails too |

One framing correction, because it costs a reader a day otherwise: *"XRT cannot bind a foreign dma-buf"* is
true of XRT and is not our blocker — this engine's NPU lane does not use XRT (`npu/rt.h`), and its own
submission path imports the fd with `PRIME_FD_TO_HANDLE` and binds a BO that points at the exporter's pages.
[docs/hetero-prefill.md](hetero-prefill.md) has the comparison.

The rule that follows, and the reason it is written here rather than left to be rediscovered: on a PASID-mode
driver, mapping an imported dma-buf through amdxdna invalidates its population and the submit path refaults
it through HMM, which an imported range cannot satisfy. **Address the exporter's own mapping.** The driver
patch that removes the invalidation is prepared and not installed (engine#812, and the patch set in
engine#1134); until the owner authorises loading it, the exporter-mapping route is the only one that executes.

That is also a contract question, and `tests/npu_import_contract.sh` (CTest `npu_import_contract`) is where
it is checked instead of remembered. The pinned `libamdf` advertises
`AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API` with device read/write on its import memory profile
(ordinal 1, `libamdf/src/xdna/umd/drm/memory_profile.c`), i.e. "import a foreign dma-buf and execute on
it". `libamdf`'s import path takes host access through an amdxdna `mmap` of the import, which is the
poisoning path in the table above, so on the shipping module that capability is advertised and refused. The
test reads the advertisement from the pin, measures the behaviour with `libamdf`'s own CTS (GPU export -> NPU
import -> NPU executes on the imported range, plus registered caller pages as the control), and keys the
expected pairing on the loaded module's `srcversion` against the fingerprint table in
`third_party/patches/amdxdna/README.md`. Run it with `ONEBIT_IMPORT_CONTRACT=advertised` for the acceptance
form: it is red while a module without `0006`/`0007` is loaded, and green when one that carries them is
(measured 2026-11-06 on `EE1F36A2B9BA294CE11AD80`, the c26 driver: binding 7/7 pass, the array executes
registered caller pages, and 8 of 8 imported-backing arms fail at `kernel_queue_create`).

The NPU BO to dma-buf direction still does not exist: the export `PRIME_HANDLE_TO_FD` returns ENOENT for the
handle the runtime produces, so an NPU-owned buffer cannot yet be handed to the GPU. Nothing on the serving
paths needs that direction today.

## How to re-measure

- Handoff: `ONEBIT_KV_SHARE=1 ONEBIT_PREFILL_DEVICE=<dev> 1bit serve` logs one line per handoff —
  `shared KV handoff: N cells, X ms, 0 bytes copied`. The file route's price is `POST /slots/<id>?action=save`
  then `?action=restore`, timed; the sizes in the table above come from fitting the state file's size against
  prompt length.
- NPU runtime: `NPU_PHASE_PROF=1` and `NPU_PROFILE_TOKEN=1` print the `[ph]` and `[prof]` rows quoted above.
- Devices are shared: GPU work through `~/.local/bin/gpu-hold`, NPU work through `npu-hold`, CPU-heavy builds
  through `thermal-run.sh`. Kernel internals stay in the private trees; nothing in this page depends on them.
