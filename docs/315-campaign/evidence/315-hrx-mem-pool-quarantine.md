<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# New in-scope lead from the REA decompilation: the HRX **memory pool quarantine/eviction** path

Found by reading the decompilation dumps now on the box (`~/amd-oss/hrx-re/decomp/`, 83 MB of REA output from
a separate, much larger effort). This is a **runtime mechanism** adjacent to the #315 trigger that this goal's
own reading had not reached, because the work so far stayed on the engine side (read path, host accesses,
AQL submission policy, graph reorder, writeback).

## What the decompilation shows

`hrx_mem_pool_trim` (`~/amd-oss/hrx-re/decomp/libhrx/hrx_mem_pool_trim.json`, REA evidence record with
`authority: shipped-artifact`, `confidence: observed`) contains a production pool implementation whose
vocabulary is exactly the trigger's:

    iree_hal_pool_stats_t local_98;
    local_98.bytes_quarantined       = 0;
    local_98.quarantine_eviction_count = 0;
    local_98.inflight_allocation_count-related gate
    local_98.over_budget_count / wait_count / exhausted_count / reuse_count / reuse_miss_count

    iree_slim_mutex_lock(&pool->mutex);
    if (pool->hal_pool != NULL) iree_hal_pool_trim(pool->hal_pool, 2, 0);
    hrx_mem_pool_backing_trim(&pool->device->mem_pool_backing, &pool->retention, min_bytes_to_keep);
    ...
    iree_hal_pool_query_stats(pool->hal_pool, &local_98);

So the runtime has:

- a **quarantine** state with an explicit **`quarantine_eviction_count`** and `bytes_quarantined` — memory is
  quarantined and then **evicted**, which is the word the #315 trigger uses;
- an **`inflight_allocation_count`** tracked per pool;
- budget/exhaustion accounting (`over_budget_count`, `exhausted_count`, `wait_count`, `budget_limit`).

## The ordering gate, which is the interesting part

The aggressive trim path is guarded:

    if ((min_bytes_to_keep == 0) && (pool->hal_pool != NULL) && (pool->inflight_allocation_count == 0)) {
        ... full trim ...
    }

i.e. **the full trim only runs when nothing is in flight**. That is a deliberate ordering discipline, and its
existence is informative in two directions:

1. It confirms that "do not reclaim memory out from under in-flight work" is a real invariant the runtime
   *tries* to maintain — which makes an eviction-during-flight event a **violation of an intended
   invariant**, not merely an unlucky interleaving. That is a sharper framing than "the window opens on
   preemption".
2. It gives a concrete place to look for how the invariant can fail: the counter must be accurate across
   every path that starts or completes work, and the trim must be reached only through this gate.

## Why this matters for #315

The trigger is forced eviction, and the pooled evidence is strong (ambient ~5% vs ~95% under the forcer). If
memory is reclaimed while an allocation is in flight, the consumer can observe replaced or stale contents —
which is the fault's shape, and matches the peer's n=14 constraint that the failure is a **terminal
whole-buffer event** rather than a changed computation.

## Honest standing

- This is a **lead read from a decompilation**, not a measurement of #315, and the decompilation belongs to a
  different effort with its own objective.
- `hrx_mem_pool_trim` is one of only **14** decompiled `libhrx` functions (of 943 across the shipped
  artifacts), so this is a single window onto the pool code, not a map of it.
- Nothing here is shown to be **reachable** in a faulting arm: whether `inflight_allocation_count` can be
  non-zero at the moment the kernel evicts, and whether the quarantine eviction applies to the buffers the
  fault corrupts, are both open.
- The decompilation carries `limitations` fields per record; those should be read before relying on any
  individual inference from it.

## Cheapest next steps, if box time is available

1. Enumerate the pool-stats counters at runtime in a faulting arm — `quarantine_eviction_count`,
   `bytes_quarantined`, `inflight_allocation_count` — to see whether quarantine eviction coincides with the
   fault. That is a counter read, not a new experiment, if the stats are exposed.
2. Read the remaining pool functions from the preserved queue (`~/amd-oss/hrx-re/pipeline/queue-full-943.tsv`
   is resumable) — specifically anything touching `inflight_allocation_count`, which is the invariant's
   linchpin.
