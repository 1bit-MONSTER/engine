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

---

# Why the dump misses them, and the fix: the queue is built from EXPORTED symbols, and DWARF is present

Chasing the register-coherence question to the end produced the cause of the whole gap pattern, and a one-line
class of fix for the RE lane.

## The missing functions are internal, and the queue only sees exported symbols

| function | where it actually lives | exported? | in the queue? |
|---|---|---|---|
| `iree_hal_streaming_memory_register_host` | **`libamdhip64.so`** (2 name occurrences) | **no** | no |
| `iree_hal_streaming_memory_allocate_owned_host_import_with_context_mode` | **`libamdhip64.so`** | **no** | no |
| `iree_hal_streaming_memory_allocate_host_staging` | **`libamdhip64.so`** | **no** | no |
| `hrx_allocator_virtual_memory_map` | `libhrx.so` (3 occurrences) | no | no |

    nm -D --defined-only libamdhip64.so.7.9999.0 | grep -E 'register_host|owned_host_import|allocate_host_staging'
    -> empty                    (none of them is a dynamic symbol)

    wc -l pipeline/queue-hip.tsv   -> 481        (the queue the pipeline actually drives)

and the queue's own first lines are `libamdhip64.so … <address> <name> <weight>` — name-and-address pairs,
which is what an exported-symbol enumeration produces.

**So the pipeline decompiles the exported surface and cannot see internal functions.** That single fact
explains the pattern found earlier: not "the decompiler struggled with the hard functions" — the hard
functions were never in the queue. It also reclassifies the 103 libhrx failures: the earlier reading
("they failed with `rc=1`") was about functions that *were* queued; these four were never attempted at all.

## The fix: the binaries carry DWARF

    readelf -S libhrx.so.0.1.0        -> no .symtab, but .debug_info, .debug_str, .debug_abbrev,
    readelf -S libamdhip64.so.7.9999.0 -> .debug_loclists, .debug_rnglists, …   (DWARF present)
    strings libhrx.so.0.1.0 | grep -cE '^(iree|hrx)_[a-z_]+$'  -> 11078 plausible C symbol names

So the internal function **names are in the binaries**, and with DWARF the addresses and source lines are too.
The queue can therefore be built from `.debug_info` (or `readelf --debug-dump=info`, `objdump --dwarf=info`,
or a DWARF library such as pyelftools) instead of from `.dynsym` — which would expose every internal function,
including all 44 libhrx priority targets and these three.

**If DWARF-derived addresses are not usable directly**, the fallback is the call-site route: the in-tree
`hipHostRegister` (which decompiled OK) calls `iree_hal_streaming_memory_register_host`, so its address is
recoverable from that call instruction, and the same holds for the others from their callers.

## What this does and does not change for #315

- **Does not** answer the register-coherence question — nothing here is a mechanism.
- **Does** explain why the question is unanswerable from the current dump and names the exact change that
  would make it answerable. If the recommendation lands, `iree_hal_streaming_memory_register_host` alone
  decides whether a default `hipHostRegister` (which is what the #315 trigger uses, with `flags = 0`) yields
  coherent or non-coherent memory.
- **Also corrects a process point from earlier in this document:** "103 failures with `rc=1`" and "never
  queued" are different defects with different fixes, and the priority list should be split accordingly —
  retry-the-decompile for the queued-and-failed, add-to-the-queue-by-DWARF-or-call-site for the never-attempted.
