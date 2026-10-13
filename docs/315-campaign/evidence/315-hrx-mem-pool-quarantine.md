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

---

# The allocate side: the invariant looks SOUND, which narrows this lead

Followed up by reading `hrx_mem_pool_allocate_buffer` (`~/amd-oss/hrx-re/decomp/libhrx/`), which is the other
decompiled function that touches `inflight_allocation_count`. Together with `trim`, it defines the pool's
centrality invariant.

## The shape

    pool->inflight_allocation_count = pool->inflight_allocation_count + 1;   // increment UNDER the mutex
    iree_slim_mutex_unlock(&pool->mutex);                                    // release it
    ...
    status = iree_hal_pool_allocate_buffer(...);                             // work happens OUTSIDE the lock
    ...
    iree_slim_mutex_lock(mutex);                                             // re-acquire
    pool->inflight_allocation_count = pool->inflight_allocation_count - 1;   // decrement

That is the correct pattern: the counter is maintained under the mutex, the blocking allocation is performed
outside it, and the decrement pairs with the increment. And `trim`'s aggressive path is gated on
`inflight_allocation_count == 0`, so the two halves are consistent by construction.

## Checked for the obvious defect, and it is not there

The classic failure for this pattern is a **leak on an error path** — an early return between the increment
and the decrement, which would leave the counter permanently high (blocking trim forever) or, in the other
direction, allow trim while work is in flight.

Counted and located the sites in that function:

    increments: 2      decrements: 2
    the one goto (LAB_0017653c) is reached at offset ~2380, AFTER the decrement at ~1021

So on the path I could read, the decrement precedes the jump and the count is balanced. **No leak found at
this site.**

## What that does to the lead

It **weakens** it as "the counter is buggy" and leaves the more interesting remainder intact: the invariant
exists and is maintained here, so an eviction-during-flight event would have to arise either from another
path that touches the counter (only two functions in the decompiled set do, and 51 pool/alloc functions
remain undecompiled), or from **outside the pool entirely** — i.e. the KFD-level eviction the trigger
actually uses does not go through `hrx_mem_pool_trim` at all, in which case this quarantine machinery is not
the affected path and the lead is about a different layer's notion of quarantine.

That second possibility is now the more likely one: the forcer (`evictonly`) drives **KFD** eviction via
`mprotect` churn on a `hipHostRegister`'d range, which never calls into HRX's pool trim. So the pool's own
quarantine counters may simply not be on the trigger's path.

## Honest standing after this read

    gate exists and is correct            (this file, previous section)
    allocate/trim pair is consistent      (this section)
    whether KFD eviction reaches it       UNKNOWN, and the forcer's mechanism suggests it does NOT

So the lead is **not** "the pool leaks its in-flight count" — that is unsupported — and the live question is
whether this quarantine machinery is on the trigger's path at all. Answering that is a runtime counter
question (pool stats in a faulting arm), which is the cheap next step already named.

---

# The pool's DOCUMENTED retirement/reuse contract — a concrete stale-read surface

Checked whether the quarantine counters could simply be read at runtime (the "cheap next step" named above).
**They cannot**, and finding that out exposed something better.

## The counters are NOT exposed

`hrx_mem_pool_get_attribute` exists and is public, but the attribute enum is closed:

    HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 0
    HRX_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 1
    HRX_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2
    HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD = 3
    HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT = 4
    HRX_MEM_POOL_ATTR_RESERVED_MEM_HIGH = 5
    HRX_MEM_POOL_ATTR_USED_MEM_CURRENT = 6
    HRX_MEM_POOL_ATTR_USED_MEM_HIGH = 7

No `quarantine_eviction_count`, no `bytes_quarantined`, no `inflight_allocation_count`. The quarantine
counters the decompiled `trim` reads come from `iree_hal_pool_query_stats` into a **local struct** and never
leave the library. So the counter test needs an API change to the runtime, which is out of scope.

## What the header documents instead — the retirement/reuse contract

`libhrx/include/hrx_runtime.h`, verbatim:

    // Records a logical allocation backed by |pool|. Logical usage is distinct
    // from physical TLSF reservations, which can remain retained for later
    // stream-ordered reuse.
    hrx_mem_pool_record_logical_allocation(pool, size);

    // Records retirement of a logical allocation backed by |pool|. The physical
    // reservation remains owned by its buffer until that buffer is released or
    // reused.
    hrx_mem_pool_record_logical_free(pool, size);

    // Trims unused backing storage toward the pool's configured release threshold.
    // This is intended for stream-ordered free completion paths.
    hrx_mem_pool_release_unused(pool);

and the attribute enum names the reuse policy explicitly — `REUSE_FOLLOW_EVENT_DEPENDENCIES`,
`REUSE_ALLOW_INTERNAL_DEPENDENCIES`, `REUSE_ALLOW_OPPORTUNISTIC`.

**So the pool deliberately retains and reuses physical memory across stream ordering.** Logical free and
physical release are separated in time, by design, and `release_unused` is documented as belonging on
*stream-ordered free completion paths*.

## Why this is the concrete stale-read surface

That separation is exactly the shape a stale read needs:

    a buffer is logically retired  ->  its physical reservation is RETAINED  ->  reused for something else
                                                                                    |
    a consumer still holding the old expectation reads the NEW contents  <------------+

and the reuse policy is **event-dependency-based** (`REUSE_FOLLOW_EVENT_DEPENDENCIES`), i.e. correctness
rests on the pool correctly following event dependencies when it decides a reservation is reusable. If an
eviction event displaces or delays the dependency bookkeeping — which is the trigger's class — a reservation
could be treated as reusable earlier than the consumer finished with it.

This is a **premise**, not a finding: nothing here shows the dependency tracking is wrong, and it is a
different layer from the KFD eviction the forcer actually drives.

## Honest statement of the lead after this read

    quarantine/trim invariant        exists and is correctly maintained (previous sections)
    counters observable at runtime   NO - needs an API change, out of scope
    documented reuse contract        separates logical free from physical release, event-dependency based
    is it on the trigger's path      UNKNOWN, and the forcer drives KFD, not pool trim

So the useful residue is a **named, documented contract** (logical-free vs physical-release, ordered by event
dependencies) that a stale read could violate — and the question of whether HRX's pool is even involved when
KFD evicts underneath it. Both need either an API addition or a different instrument; neither is in scope as
a change.
