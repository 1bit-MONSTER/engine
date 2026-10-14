<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The staging/memory-visibility mechanism: named from source, and REFUTED on the one tensor family the archive can test

Pre-registered before scoring, then run. Recorded as a negative, not as "no difference".

## What was hunted and what was found

Following the `memory.c` host-transfer captures into `graph.c` and the allocators produced a concrete,
falsifiable mechanism (`315-MECHANISM-staging-handoff-not-memory-visible.md`):

> The host↔device handoff is **ordered** (a timeline semaphore) but never made **memory-visible**: all three
> host-import paths hardcode `HOST_LOCAL | DEVICE_VISIBLE` = **0x52** (non-coherent) while merely *storing* the
> coherence flags, and the runtime issues **no invalidate and no flush anywhere** (`invalidate_range` 0,
> `flush_range` 0, `mapping_invalidate` 0, `mapping_flush` 0 across `libhrx/src`). The staging buffer is graph-owned,
> so it is the **same VA on every replay** — the precondition for a stale CPU line to hold the *previous*
> replay's bytes.

Supporting source facts, all confirmed this session:

- `memory.c:122-158` (D2H): device copy writes `staging->device_ptr`, then a host callback does
  `memcpy(dst, staging->host_ptr, count)`.
- `memory.c:161-193` (H2D): host callback writes `staging->host_ptr`, then a device copy reads
  `staging->device_ptr`.
- `graph.c:257-282`: staging comes from `allocate_host_staging(graph->context, size, …)` → the
  **owned-host-import** path → declared **0x52**.
- Three separate allocators hardcode `0x52` and only store `host_register_flags`.
- Archive: `hrx-copy site=get_tensor_async tensor=result_output size=619520 host_ptr=0x7fb014700000` — the logits
  are read at a **constant, page-aligned** host address every replay, and `[hrx-buf] … host_visible=1
  memory_type=0x52 size=619520` is that same buffer. So the logits the engine reads **are** non-coherent host
  memory at a fixed VA.
- **And the record's own probe comment says it outright** (`llama-315copy/…/ggml-hrx.cpp:677-684`):
  *"(1) the read the probe has always done: plain memcpy for a host-visible buffer"* versus *"(2) the same tensor
  through the synchronising path that buffer_get skips for a host-visible buffer."* — an instrument built to
  test exactly this hypothesis.

## The discriminating experiment, and its answer

`hrx-ids-fp` = unsynchronized host `memcpy` of a host-visible buffer. `hrx-ids-sync` = the same tensor through
`hrx_synchronous_d2h`. Both record `min/max/oor/dup_pairs/first8`. **If the host read is stale, the two must
differ.**

    log files scanned      : 943
    hrx-ids-fp  records    : 253,644
    hrx-ids-sync records   : 253,598
    paired (same call+name): 253,598
    pairs where the two DISAGREE: 0

Across the arm population (866 arms carrying the probe): **1,160,376 paired reads, 0 differences.**

**So the mechanism is REFUTED on the MoE id tensors.** The synchronization skip in
`ggml_backend_tensor_get` for host-visible buffers is real (source-confirmed) but produces **no observable
divergence** on a million-plus reads. Any stale-line effect there is either absent or below this instrument's
resolution.

## The precise gap this exposes — and it is the whole reason the mechanism is not yet testable

The sync twin is implemented **only in the i32 branch** of the probe:

    if (tail_n == 0 && node->type == GGML_TYPE_I32) { … (1) memcpy … (2) hrx_synchronous_d2h … }

**So the f32 tensors — `l_out`, `node_2948`, `result_output` — have no sync twin at all**, and those are
precisely the tensors that carry the NaN and the 256-byte-granular corruption. The record contains 1.16M pairs
for the family that never corrupted and **zero** for the families that did.

**Therefore: the staging/memory-visibility mechanism is neither confirmed nor refuted where it matters.** It is
refuted on the ids family, which is evidence *against* it but not decisive, because the ids tensors are written
by a different producer (argsort) and are i32.

## So the next step is one instrument change, not a campaign

Extend the existing fp/sync twin from the `GGML_TYPE_I32` branch to the f32 branch — the same two reads, the
same reported fields, now over `l_out`, `node_2948` and `result_output`. This is a small edit to an **existing,
already-verified** probe; it needs **one arm**, not a campaign, and it converts the mechanism from untestable to
decided.

Pre-registered prediction, committed before the arm runs:

| outcome on f32 tensors | conclusion |
|---|---|
| fp != sync, and the differing region is 256-byte aligned | **mechanism CONFIRMED** — the host read is stale at the measured granularity, and the fix is the visibility operation |
| fp == sync on a faulting arm | **mechanism REFUTED for the f32 path** — the corruption is real device state, and the surviving candidate set is the wave64 / cache-granule / MoE-table readings |
| the arm does not fault | INVALID for this question — report as such, do not report as "no difference" |

**This is the first point in the investigation where a single, cheap, already-scaffolded measurement decides
the mechanism rather than narrowing a family.**
