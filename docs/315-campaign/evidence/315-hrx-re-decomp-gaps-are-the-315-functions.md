<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The hrx-re decompilation dump fails on exactly the functions #315 needs (44 priority failures)

Read from `~/amd-oss/hrx-re/decomp/` — the Ghidra pipeline that was loading the strixhalo box. This is a
**gap analysis of the dump**, produced because the coherence question could not be settled from the source
tree and the dump was the next place to look.

## Coverage

| library | decompiled JSONs | status |
|---|---|---|
| `libhrx` | 219 | **116 ok, 103 failed** (`decompile failed rc=1`) — 47 % failure |
| `loomc` | present | shards still running when this was read |
| `amdhip64` / `hipbinding` | present | — |

So roughly half of libhrx is decompiled, and the useful question is *which* half — because a dump that is 53 %
complete is not uniformly useful; it is useful or not depending on whether the gaps cover your question.

## The gaps cover them

**44 of the 103 failures match the #315 vocabulary** (map / protect / reserve / unmap / page / queue / sync /
fence / semaphore / stream / wait / signal / evict / restore / coher / host). The priority subset:

    hrx_allocator_virtual_memory_map          hrx_allocator_virtual_memory_protect
    hrx_allocator_virtual_memory_unmap        hrx_buffer_unmap
    hrx_device_synchronize                    hrx_event_synchronize
    hrx_fence_extend / insert / release / retain / signal / wait
    hrx_semaphore_query / release             hrx_hal_semaphore_lists_initialize
    hrx_host_allocator_clone / free / free_aligned / malloc / malloc_aligned
    hrx_host_allocator_malloc_uninitialized / realloc / realloc_aligned / system
    hrx_queue_host_call_thunk                 hrx_graph_host_callback
    hrx_buffer_table_reserve_insert / cancel_reserved_insert

**Several of these are the mechanism question itself:**

- **`virtual_memory_map` / `_protect` / `_unmap`** are the VMM path the #315 trigger drives directly — the
  trigger is `mprotect` churn over a `hipHostRegister`'d range (`~/wt/evictonly.hip`), and a restored queue
  or mapping is what the whole fault is attributed to. All three failed.
- **`hrx_device_synchronize` and `hrx_event_synchronize`** are the read-side barriers every ordering
  hypothesis depends on. Both failed.
- **`hrx_host_allocator_*`** (nine functions) is the host allocation path, i.e. where a non-coherent mapping
  would come from. All nine failed.
- **`hrx_fence_*`** is the ordering primitive family. Six failed.

## Why this matters for the goal

The coherence candidate was resolved **only because `hrx_allocator_allocate_buffer`, its callee
`hrx_mem_pool_backing_retain_allocation_source`, and `hrx_buffer_create_from_hal` were among the 116 that
*worked*** — that is what produced the falsification
(`315-coherence-candidate-falsified-by-decomp.md`). The same dump, one function further along the chain
(`virtual_memory_map`), is **empty**, which is precisely why the mapping half of the question could not be
closed and had to be left as an explicit gap.

So the dump's value for #315 is currently **one question answered and one blocked, by which functions
happened to decompile** — not by any property of the defect.

## The concrete ask, for the hrx-re lane

Re-run the decompiler on the **44 priority failures**, `virtual_memory_*`, `hrx_fence_*`,
`hrx_semaphore_*`, `hrx_*_synchronize` and `hrx_host_allocator_*` first. They failed with `rc=1` uniformly, so
the likely causes are address/analysis issues rather than per-function difficulty (a thunk, an undefined
instruction, or a function the auto-analyser did not enter) and the usual remedies — force-a-function at the
address, add the entry point, re-run auto-analysis over that range, or decompile from the call site — are the
ones to try. `hrx_allocator_virtual_memory_map` is the single highest-value target in the dump: it is the
only thing standing between "the coherence bit cannot change the pool" (established) and "the coherence bit
cannot change anything at all" (not established).

**Nothing in this document infers a mechanism from the gap.** A missing decompilation is a missing
measurement, and the failure list is useful only as a work queue.
