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
| 1 | hybrid KV handoff without the file | **shipped**, opt-in: `ONEBIT_KV_SHARE=1` ([docs/hybrid.md](hybrid.md), "The shared route"). 0 bytes copied; the cost is cell metadata, **0.12 ms at 307 cells to 5.56 ms at 16 399** |
| 2 | NPU runtime: activations live in BOs | **closed**. A is quantised on an AVX2 path straight into the mapped BO, the pad memset is gone, and both cache sweeps are narrowed to the rows the host touches: C invalidate **1 065.09 → 66.57 MB/token**, A writeback **1 908.62 → 321.59 MB/run** (engine#908, #873, #988, #1004). Both syncs are **required** — the kill switches emit garbage |
| 3 | ONNX EP IO binding onto BO memory | **closed for this tree**: the node boundary moves **0 bytes** (1 983 trace lines, never a `copy bytes=` line), and the inputs are the graph's own host tensors, which on XDNA2 *are* BO-backed. The largest pure copy left is the cached graph's own `Concat` — 458 KB/position, 41-58 us/position — an op/kernel problem, measured and handed to the ONNX lane |
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
| shared region, 4B, 307 / 4 105 / 16 399 prompt tokens | **0.12 / 1.99 / 5.56 ms**, `n_read` 0 | the number of **cells**, not bytes: ~0.34 us per cell |

Measured in the shape that ships (2026-11-06, build at pin `d70b67bf0`): a *separate* provider process, the
region exported over `SCM_RIGHTS`, an **AddressSanitizer-instrumented** provider mapping it and prefilling into
it, the decoder adopting the cells. **0.122 / 1.988 / 5.558 ms** to adopt 307 / 4 105 / 16 399 cells, with
`n_read` **0** and `prompt_n` **1** on every arm — the decoder decoded from the adopted cells instead of
re-processing the prompt — and **0 ASan reports** in the provider. So the handoff is cells bookkeeping and it
does grow with the prompt, just not with prompt *bytes*: 0.12 ms at 307 cells, 5.6 ms at 16 399. The shared
figures are on a 4B because that is the model the ASan harness drives (a 30B under the sanitizer prefills for
hours); the property being measured — cells, not bytes — does not depend on the model, and the file column
beside it is the 8.2-242.9 ms that same handoff costs at those prompt lengths on 30B-A3B.

The provider's own side of that handoff is **40.6 / 78.7 / 150.4 ms** to map and first-touch 648 MiB / 1.1 GiB
/ 2.8 GiB of region — a page fault per page it writes, not a copy, and inflated by the sanitizer; an
in-process run of the same adoption step measured 0.048-0.258 ms at 256-2 048 cells. The gate for this target
is text identity, not a tolerance: greedy text identical to `--device hrx` on every arm.

Two properties that make it usable rather than a demo: a decoder that refuses to export now falls back to the
state-file route for that request (engine#1203) instead of losing the handoff, and a decoder that *cannot* export (any backend but `ggml-hrx`:
the CPU, for instance) starts normally and warns once instead of dying at load (fork#212). `ONEBIT_KV_SHARE`
is read as "set and non-empty" on both sides: `=0` still enables the route, an unset variable is the opt-out.

What is **not** proven: the NPU is not yet the *provider* that writes that region — the provider in the gate
above prefills with the CPU. The piece that was missing under it, a device-side K/V writer, now exists in the
private tree (engine#753): the NPU writes a prefill's K/V rows straight into the region, both halves of the row
landing in the decoder's own tensors at the decoder's offsets and strides, and it is verified byte-for-byte
against a CPU reference, including the bytes it does not write (a writer that runs long must fail there). What
is left is the provider binary around it, which is the same handoff contract as the CPU provider's.

Two facts from that work belong here because they bound what a device provider can do:

- the region is the decoder's cache, so a provider writes **two destinations per layer, not one**: `cache_k_l<L>`
  and `cache_v_l<L>` are separate tensors and a token's K|V row is contiguous in the compute but not in the
  cache (row stride `n_embd_k_gqa`, K and V at different tensor offsets);
- it must write through **its own mapping** of the dma-buf. Asking the driver for a mapping of the import is the
  route that makes every later submission fail (engine#812); the pages are the same either way.

### The arm a device provider replaces, and the model to run it on

The gate needs a model that exists **both** as a GGUF (the HRX decoder, which is the exporting side) and as a
Q4NX pack (what a device provider prefills from). Pairing every GGUF staged under `~/models` (two levels deep)
against the packs in `~/models/npu` by architecture, layer count and hidden size gives **four** pairs, and the
first arm should be the cheapest of them rather than the one with the weights already staged:

| pair | GGUF | pack | why it is or is not the first arm |
|---|---|---|---|
| **Qwen3-0.6B** | `~/models/Qwen3-0.6B-Q4_K_M.gguf` (396 MB, 28 layers, hidden 1024) | `~/models/npu/Qwen3-0.6B`: 28 `attn` | **first arm**: one small region, ~5 s of device time, and no recurrent half to get right |
| ZR1-1.5B | `~/models/zyphra-zr1/ZR1-1.5B-bf16.gguf` (3.56 GB, 28 layers, hidden 1536) | `~/models/npu/ZR1-1.5B-npu-q4km`: 28 `attn` | a good second arm: still all attention, 9x the weights |
| Qwen3-Coder-30B-A3B | `~/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (18.6 GB, `qwen3moe`, 48 layers, hidden 2048, 128 experts) | `~/models/npu/qwen3-coder-30b-a3b-moe`: 48 `attn` | third: all attention like ZR1 and no recurrent half, but 47x the 0.6B's weights and a MoE plan, so a device arm here measures the MoE path before the handoff is even interesting |
| Bonsai-2-27B | `~/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.9 GB, `qwen35`, 64 layers, hidden 5120; the unrot q4_0 in the same directory is the same model) | `~/models/npu/bonsai2-27b-NPU`: 64 layers, **48 GDN + 16 attention** | last: the region is also the GDN recurrent state (fork#181), so the writer has 48 recurrent layers to get right as well as the attention cache |

The other staged GGUFs have no pack and are not candidates for this gate: Qwen3-4B (both quants), Qwen3.5-4B,
Qwen3.6-35B-A3B, zaya1-8b and bge-micro-v2.

Qwen3-0.6B's geometry was read off both sides rather than assumed: the GGUF has 28 layers, hidden 1024, 16 query
heads and **8 KV heads** of head_dim 128, i.e. `n_embd_k_gqa` **1024**, and the pack's `plan.json` has the same
28 layers with a fused qkv of `N=4096` = 2048 q + **1024 k + 1024 v**. A device writer emitting rows at stride
1024 f16 therefore lands in the decoder's own `cache_k_l*` / `cache_v_l*`.

Measured on that pair (CPU ASan provider, `gpu-hold` ticket, 5 s of device time; logs in the lane's probe dir):

| prompt tokens | regions | provider import | decoder adopt | `n_read` | `prompt_n` | greedy text | ASan |
|---|---|---|---|---|---|---|---|
| 307 | 1 | 1.276 ms | **0.109 ms** | 0 | 1 | `The tide comes in twice a day,` | 0 |
| 1 027 | 1 | 1.103 ms | **0.227 ms** | 0 | 1 | identical | 0 |

`n_read 0` is no prompt bytes moved; `prompt_n 1` is the decoder decoding from the adopted cells rather than
re-processing the prompt. These are the numbers a device provider has to reproduce, and the import leg is 30x
cheaper than on the 4B above (1.1-1.3 ms against 40-157 ms) because the region is one small cache -- which is
the measurable reason this pair is the first arm, and why the two other pairs in the table above are second and
last rather than first.

One requirement makes the difference between a gate and a demo: the layout header's `params` (`n_ctx`, `n_batch`,
`n_ubatch`, `n_seq_max`, `flash_attn`, `type_k`, `type_v`) are the **decoder's**, and a provider has to prefill in
the ubatch the cells are adopted at. A provider that prefills in a different ubatch writes the right rows of the
wrong stratification.

## Target 2 — the NPU runtime's copies, measured

Per launch the runtime quantises one row into the activation BO, flushes it, waits, invalidates the result BO
and dequantises row 0 back to host floats. What that cost *before* this target was worked is what the
2026-10-31 measurements below were taken on — they are the starting point, not the current state:

| | then (1 330 ms/token build) | now |
|---|---|---|
| A stage: 128-row zero pad + quantise of one row | 50 ms/token | quantise writes straight into the mapped BO on an AVX2 path (#908), and the pad rows are only re-zeroed when a previous launch wrote them (#873) |
| A writeback per launch | whole BO: 1 908.62 MB/run | rows the host touched: **321.59 MB/run** (#1004) |
| C invalidate per launch | whole BO: 1 065.09 MB/token | rows the host reads: **66.57 MB/token** (#988), 11.0-11.3 → **2.0-4.0 ms/token** on the same instrument |
| host-side share of a token | ~1.5 % | a few percent at most, and what is left is the launch count, not bytes |

Each of those was gated byte-identical at token, logprob and head level, on two containers and two rounds, and
the narrowed ranges were re-proven on a later driver (6/6 arms). So this target is **closed**, and the honest
residue is not a copy: when the pad tile was measured (2026-10-31) the pack registered **0** kernel roles at
M=16, so a decode launch padded one row to M=128 — 8x the work. The M=16 twins exist now (7 ELFs in the 27B
pack, 18 in the pack the fleet's serve runs, which is launched with `--kernels …/kernels-m16`), so what is left
there is which tile a decode step *selects*, which is the NPU runtime's business and not a memory fix.

Two facts from measuring the readback, which are why the sweeps were narrowed rather than deleted:

- `bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE)` is a real cache sweep (`_mm_mfence; clflush` per 64 B, then
  `_mm_mfence` — disassembled from the shipping runtime), not a copy. It is already paid on every launch.
- A dirty host cache line over the result BO **clobbers the NPU's output**: poisoning the row before the
  launch and not writing it back leaves the sentinel to be evicted over real results, which reads exactly like
  the dropped-write fault in engine#666. That ordering is also why both sweeps are still there: the arms that
skip them (`NPU_NO_A_FLUSH=1`, `NPU_NO_C_INVAL=1`) **emit garbage**, so the answer to "when does the cache model
need a sync" is "on every launch, over the rows the host touches" — measured, not assumed.

Nothing here needs the file: this is all in-BO traffic.

## Target 3 — the ONNX execution provider's boundary

The EP reads and writes `OrtKernelContext` in place inside its own host nodes, and copies at the node boundary
(`DT_CopyTensors`, `onnx/ep/onebit_ep.cpp:206`): per fused QDQ GEMM node, an `M x K` f32 input and an `M x N`
f32 output — 20 KB + 0.6 MB per node for a 0.6B decode step at M=1. That was the target: bind inputs and
outputs to BO-backed `OrtValue`s and remove the boundary copies. **It was measured and the boundary is already
empty**: a profile of the fused path never prints a `copy bytes=` line at all (1 983 trace lines), because the
group is handed the graph's own host tensors and XDNA2 BOs *are* host memory — binding would be the same bytes
to the same place. What a copy could still buy is the readback leg, **0.044 ms of a 4.32 ms call (0.9 %)**.
The largest pure copy left on this route is the cached graph's own `Concat` (458 KB/position, 41-58 us/position
at 4 threads, ~2.1x linear at 4 096 positions) — that is how `present` is produced, so it needs an op/kernel to
write the cache in place, not a binding change. Both numbers come from the ONNX lane's measurements.

## The audit: every copy still on the serving paths

27B Qwen3.5 hybrid on the int8 NPU path, M=128, unless the row says otherwise. "Bytes/token" counts what is
read and written per decoded token; a row marked *no memcpy* is arithmetic over memory rather than a transfer.

| # | copy | where | bytes/token | ms/token | why it stays |
|---|---|---|---|---|---|
| 1 | A stage: quantise one row into the mapped BO (was: `memset` the 128-row pad first) | `npu/forward/engine-src/npu_engine_i8ctx_inc.h` (`quantize_async` -> `prepare_activation`) | one KD row written; the pad memset is gone (#873), and the quantiser is an AVX2 path writing straight into the mapping (#908) | 50 on the pre-#908 build; not re-quoted | the pad, not the copy: at measurement time the pack had **0** kernel roles at M=16 (M=16 twins ship now), so this row is about which tile a decode step selects, not about a memcpy |
| 2 | A writeback: `sync(TO_DEVICE)` per launch | `sync_a_to_device()` | whole BO, 1 908.62 MB/run -> **321.59 MB/run**: the rows the host touched (#1004) | inside row 1 | cache sweep, not a copy, and **required** — `NPU_NO_A_FLUSH=1` emits garbage |
| 3 | C invalidate: `sync(FROM_DEVICE)` per launch | `readback()` | whole BO, 1 065.09 MB/token -> **66.57 MB/token**: the rows the host reads (#988) | 11.0-11.3 -> **2.0-4.0** per token | cache sweep, not a copy, and **required** — `NPU_NO_C_INVAL=1` emits garbage |
| 4 | C dequantise: int32 rows of C to a host float vector | `dequant_only` | `am x an` elements per launch (one row on decode), not the whole BO | inside row 3 | the host reads those rows once anyway; folding the next launch's quantise into it is kernel work, not a copy fix |
| 5 | LM head: 15 packed tiles, 15 launches, an invalidate per tile | `run_lm_head` | 1.0 MB of logits to host | **180** (`[ph]` "head") | the per-token 356 MB f32 tile memcpy is already gone on the shipped path; what is left is the device streaming the int8 head, not a copy |
| 6 | GDN recurrent state swept on the host, in place, per layer | `gdn_layer`, `rec_state_[l]` | **604 MB** read+written (48 layers x 3.14 MB x ~4 passes) | part of `[ph]` "host-inside-layers" **190** | the largest copy-shaped cost on this path and no zero-copy fixes it: a recurrence has to run somewhere. Removing it moves the recurrence into an NPU kernel — kernel work, private tree |
| 7 | QKV rope per-head row staging | `attention()`, `step()` | ~0.4 MB | ~0.3 | per-head rope needs contiguous rows; the copies are L1-sized |
| 8 | hybrid KV handoff, file route | `app/serve_hybrid.h`, `llama_state_seq_save_file` + `?action=restore` | 65 536 B/token + 151 MB once (27B) or 98 304 B/token (30B) | **28.8 / 61.8 / 198.5** at 307/4k/16k (27B) | replaced by target 1 wherever both sides can share; kept as the fallback for a decoder that cannot |
| 9 | KV cell adopt after the handoff | the pinned fork's `src/llama-kv-share.cpp` (`llama_kv_cells_copy`) | **0** | **0** | already zero copy: metadata only |
| 10 | ONNX EP node boundary | `onnx/ep/onebit_ep.cpp` (`DT_CopyTensors`, `:206`) | **0** on the fused path: it is handed the graph's own host tensors, and a 1 983-line profile never prints a `copy bytes=` line | **0.044**/call, the readback leg only | measured, and target 3 is closed: the copies it named are not there. The copy that remains is the graph's own `Concat` when it writes `present` — an op/kernel problem |

Two conclusions the table forces, written down so they are decisions rather than leftovers:

- **Rows 1 and 3 were ~59 ms of a 1 330 ms token** on the pre-optimisation build (rows 2 and 4 are inside them,
  so they do not add). That is why target 2 paid off as range narrowing rather than as an architecture change:
  #988 and #1004 took ~90 % of the traffic out of rows 2 and 3, which is why those rows read as bytes-per-run
  above now. What is left that pays is the launch count and the M=128 pad — kernel-set work, not a copy fix.
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
`AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API` with device read/write on the **dma-buf-fd entry** of its
import memory profile (ordinal 1, `libamdf/src/xdna/umd/drm/memory_profile.c`), i.e. "import a foreign
dma-buf and execute on it". The **opaque-fd entry** in the same profile is cloned from that one and then has
the flag cleared, carrying the direct-host provenance instead -- a different advertisement for the same
execution, which is why the acceptance below covers both transports and reports them apart. `libamdf`'s
import path takes host access through an amdxdna `mmap` of the import, which is the poisoning path in the
table above, so on the shipping module that capability is advertised and refused. The test reads the
advertisement from the pin, measures the behaviour with `libamdf`'s own CTS (GPU export -> NPU import -> NPU
executes on the imported range, plus registered caller pages as the control), and keys the expected pairing on
the loaded module's `srcversion` against the fingerprint table in `third_party/patches/amdxdna/README.md`.
Run it with `ONEBIT_IMPORT_CONTRACT=advertised` for the acceptance form: it is red while a module without
`0006`/`0007` is loaded, and green when one that carries them is (measured 2026-11-06 on
`EE1F36A2B9BA294CE11AD80`, the c26 driver: binding 7/7 pass, the array executes registered caller pages, and
8 of 8 imported-backing arms fail at `kernel_queue_create`).

The advertisement half is a parse of the pin, so it has a device-free test of its own: CTest
`npu_import_advert` (`tests/npu_import_contract.sh --self-test`) reads two fixtures — an advertising pin and
the same pin with the flag dropped from the import class' dma-buf entry — and requires `advertised` on the
first and *not* on the second. Both bounds it needs are load-bearing and were both wrong in the first
version: the ordinal-0 class advertises the same flag, and the import class clears it on the opaque-fd entry
it clones, so a whole-file search reports a narrowed pin as advertising (engine#1283's review).

The NPU BO to dma-buf direction still does not exist: the export `PRIME_HANDLE_TO_FD` returns ENOENT for the
handle the runtime produces, so an NPU-owned buffer cannot yet be handed to the GPU. Nothing on the serving
paths needs that direction today.

## How to re-measure

- Handoff: the cross-process harness is `t1_handoff_lengths.sh` + `t1_asan_import.py` in the zero-copy probe set (it
  exports, shares over `SCM_RIGHTS`, runs an ASan provider and adopts). `ONEBIT_KV_SHARE=1 ONEBIT_PREFILL_DEVICE=<dev>
  1bit serve` logs one line per handoff in the engine's own route —
  `shared KV handoff: N cells, X ms, 0 bytes copied`. The file route's price is `POST /slots/<id>?action=save`
  then `?action=restore`, timed; the sizes in the table above come from fitting the state file's size against
  prompt length.
- NPU runtime: `NPU_PHASE_PROF=1` and `NPU_PROFILE_TOKEN=1` print the `[ph]` and `[prof]` rows quoted above.
- Devices are shared: GPU work through `~/.local/bin/gpu-hold`, NPU work through `npu-hold`, CPU-heavy builds
  through `thermal-run.sh`. Kernel internals stay in the private trees; nothing in this page depends on them.
