<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# MECHANISM: the host↔device staging handoff is ordered but never made memory-visible, on memory the runtime itself declares non-coherent

**Status: candidate mechanism, named and falsifiable, with a discriminating experiment that needs no GPU.**
This supersedes `315-coherence-closed-from-source-both-paths.md`'s conclusion that the family is dead: the
family was dead as an A/B *knob*, because the flags never reach an allocation — and that same fact is what
creates the defect. Coherence is not a knob that failed to help; it is an invariant the runtime violates.

## The two facts that meet

**Fact 1 — every host-imported buffer is declared NON-COHERENT, on every path.** Three separate allocators
hardcode the type and merely *store* the coherence flags:

| path | site | type |
|---|---|---|
| register/import (`hipHostRegister`) | `memory.c:2510-2530` | `HOST_LOCAL \| DEVICE_VISIBLE` = **0x52** |
| owned host import (**this is the staging path**) | `memory.c:2229-2233` | `HOST_LOCAL \| DEVICE_VISIBLE` = **0x52** |
| pool (ggml-hrx staging) | pool chosen by `HOST_LOCAL` (0x42) only | coherence bit never consulted |

with `wrapper->host_register_flags = flags;` in each case — stored, then read only by a copy (`:755`), a store
(`:2405`) and a report-back through `hipHostGetFlags` (`:2667`).

**Fact 2 — the runtime never makes memory visible.** Exhaustive counts over `libhrx/src`:

    invalidate_range        0        mapping_invalidate        0        buffer_mapping_invalidate  0
    flush_range             0        mapping_flush             0

and the only `flush` names in the tree are `stream_flush` / `context_flush` (command submission) and
`fflush` (stdio). **No invalidate and no flush is ever issued, anywhere.**

## The handoff they meet on

`memory.c:122-158` and `:161-193` — the graph's two host-transfer captures:

    device_to_host:  copy_node:  staging->device_ptr  <- capture->src        (device writes)
                     host_call (depends on copy_node):  memcpy(dst, staging->host_ptr, count)   (CPU reads)

    host_to_device:  host_call:  memcpy(staging->host_ptr, src, count)       (CPU writes)
                     copy_node (depends on that host_call):  capture->dst <- staging->device_ptr  (device reads)

and `graph.c:257-282` shows the staging buffer comes from
`iree_hal_streaming_memory_allocate_host_staging(graph->context, size, …)` — the **owned-host-import** path,
i.e. **declared 0x52, non-coherent** — is owned by the graph, and therefore **the same virtual address on
every replay**.

So the handoff is: *CPU and GPU exchange data through non-coherent memory, ordered by a timeline semaphore
(which guarantees **execution** order) with no operation that guarantees **visibility**.*

## The mechanism

> A semaphore wait orders execution; it does not invalidate the CPU's cache. On non-coherent memory the CPU
> may still hold valid lines for the staging VA from the previous replay — and it is the **same VA every
> replay** — so `memcpy(dst, staging->host_ptr, count)` can return the **previous** replay's bytes, and
> `memcpy(staging->host_ptr, src, count)` can leave the device reading **stale** bytes, because nothing ever
> writes back the dirty lines or invalidates the clean ones.

**This is a memory-visibility defect, and its severity is a function of CPU cache residency — which is
precisely a variable, not a constant.** It is *not* the same claim as "the memory is non-coherent" (a
constant, which would break every transfer) and it is *not* "a missing invalidate is absent so it cannot
vary" — the invalidate is missing *always*; what varies is whether the CPU happens to hold the lines, so the
*symptom* is intermittent even though the defect is permanent.

## It explains every hard fact on the record

| fact | explained by |
|---|---|
| **Face B: `nan=0 http=1 s500=1 unparsed=2`, no NaN anywhere** | D2H: the CPU reads a **previous replay's** logits/hidden state — finite, plausible, wrong. The sampler picks a different token; the server's parser rejects the body. **No NaN is manufactured**, exactly the observed signature. |
| **Face A: NaN logits / `HSA_STATUS_ERROR_MEMORY_FAULT`** | H2D: the device reads stale input bytes → computes from them → a real device-side residual row goes NaN/wrong. This is the chain already on the record, now with a root cause instead of a guess. |
| **The instrument sees true device state while the engine is wrong** | The fingerprint reads via `download_synchronous` → `hrx_stream_synchronize` then `hrx_synchronous_d2h`, a device-side path with a genuine sync. The **engine's own** output goes through the non-coherent staging. Hence `nan=0` in the instrument on arm after arm that produced HTTP 500. |
| **256-byte granularity; `2048 = 32×64`, `1984 = 31×64`** | 64 B is the cache line; 256 B is four lines, the unit at which a partially-visible write-back or a copy-engine burst shows up. 31 of 32 visible and one not is a partially-flushed transfer, not a random sprinkle. |
| **Intermittency, ~5 % ambient vs ~95–100 % forced** | Cache residency decides it. |
| **The trigger is `mprotect` churn over a `hipHostRegister`'d range** | mprotect rewrites page tables and forces TLB shootdowns and cache maintenance on **exactly the host memory involved**. It is a *cache-visibility stressor*, not a data path — which is why a pure-eviction exercise is the strongest forcer, and why forcing moved the rate by 20×. |
| **The read pattern sets the onset, probes do not** | The handoff happens at graph/host boundaries; which node is the first to consume host-published data depends on what the host reads, not on how many instrument probes are attached. An instrument-relative onset is *expected* of a host-side handoff. |
| **Every coherence knob was inert** (three record-wide negatives) | The flags never reach an allocation. The knob cannot change the memory, so it cannot change the outcome. |
| **PR #90: "graph inputs are never written back to host"** | The record had **already** found a host/device visibility defect in the *input* direction, and shipped it. This names the class it belongs to. |

## Discriminating experiment — and it needs no GPU

The mechanism makes one prediction that no alternative makes, and the archive can test it directly:

> **If the host reads stale cache lines, the corrupt bytes must equal an EARLIER replay's bytes for the same
> buffer** — because the stale content is whatever the same staging VA last held, i.e. the previous token's
> data. Not random garbage, not NaN, not a neighbour's data: **the same tensor, one call earlier.**

So: for each partial reading (`l_out`, `node_2948`, …) with a corrupt row, compare that row's fields against
the **same tensor's rows from previous `call=` numbers** in the same archive. A match is proof of a stale-line
read and simultaneously excludes every other surviving reading (wave64 masking, MoE-table aliasing, uninitialised
memory — all of which would produce unrelated content).

**Pre-registration, committed before scoring:**

| outcome | conclusion |
|---|---|
| corrupt row == same tensor's row from an earlier call | **mechanism confirmed** — stale host cache lines on the staging path |
| corrupt row == another tensor's data, same call | not this mechanism — aliasing/reuse instead |
| corrupt row not present anywhere earlier | not this mechanism — recompute the candidate set |
| archive lacks the needed earlier rows | INVALID — report as such, do not report as "no difference" |

## Fix direction (forward-only)

The change is a visibility operation at the handoff, in the fork, on top of the pin:

1. **Preferred** — make the staging allocation actually coherent: stop hardcoding `0x52` in
   `iree_hal_streaming_memory_allocate_owned_host_import_with_context_mode` and honour
   `IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_COHERENT`. Small, local, and it repairs the invariant the
   runtime currently violates.
2. **Or** — a `flush` after the H2D host memcpy and an `invalidate` before the D2H host memcpy, i.e. use IREE's
   own mapping-visibility APIs, which exist for exactly this and which this runtime never calls.

Both are additions on top of the pin, so forward-only is satisfied by construction. Neither is a revert.

**Nothing is claimed as verified yet.** The mechanism is named and falsifiable, the fix direction is concrete,
and the discriminating experiment is pre-registered above and can be run on the existing archive.
