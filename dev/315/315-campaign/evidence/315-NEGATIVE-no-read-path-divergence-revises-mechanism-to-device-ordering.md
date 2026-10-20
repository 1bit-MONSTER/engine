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

# NEGATIVE against the visibility hypothesis, and the mechanism revised to a DEVICE-SIDE ordering defect

The `315-MECHANISM-kv-not-visible-to-decode.md` hypothesis predicts that a plain-memcpy read and a synchronised
read of the same buffer will **disagree** at the divergent step. The record has **two** instruments that make
exactly that paired comparison, and **neither has ever observed a divergence**.

## The two paired-read instruments, both negative

| instrument | paired reads | records | divergences |
|---|---|---|---|
| `hrx-dl-fp` | `probe_hash` vs `copy_hash`, with `hashes_equal` | **146** | **0** — every record is `hashes_equal=1` |
| `hrx-ids-fp` vs `hrx-ids-sync` | unsynchronised host memcpy vs `hrx_synchronous_d2h` | **253,598** | **0** |

**253,744 paired readings across two independent instruments, zero disagreements.** So on everything the record
has ever compared, **the two read paths agree**.

**This is real counter-evidence and it is recorded as such.** It does not refute the hypothesis outright — the
comparisons are on the weight-download path and on i32 routing tensors, and **no arm has ever made the comparison
on the KV cache at a divergent step** — but it removes the reason to expect a divergence, and a hypothesis whose
central prediction has been tested 253,744 times elsewhere with zero successes should not be promoted on the
strength of a step it was never measured at.

## What follows: the wrong KV values are genuine device state

If the two reads agree, then a wrong value read at a divergent decode step is **what the device actually holds**.
So the fault is **on the device**, and the visibility framing is the wrong one. Re-reading the measured facts
under that constraint:

- **forced eviction 15.1×** — eviction changes **timing** and memory pressure on the device, which widens a race
  window; it is not evidence about CPU caches;
- **index 0 immune, 0/2259** — index 0's logits are computed **in the same pass that writes the K/V**, so no
  cross-op dependency exists to be mis-ordered; every index ≥ 1 depends on the prefill's KV writes;
- **front-loaded over ~7 decode steps, 4× under eviction and *zero* early events in control/quiet** — exactly what
  a dependency whose window closes as the pipeline drains looks like;
- **finite, clean-prefixed, no crash** — a stale read of genuinely-written memory, not poison;
- **no invalidate anywhere in the runtime** — true, but now **irrelevant**: it would matter for CPU-visible
  staleness, and no such staleness is observed.

## The revised mechanism (device-side ordering)

> **The prefill writes the prompt's K/V; the decode steps read it across a captured-graph node boundary. The
> dependency between them orders *execution* but does not reliably order *memory visibility on the device*, so
> under pressure a decode step can read K/V before the prefill's writes have become visible to it. The values it
> reads are real numbers from an earlier state, so the logits are finite and plausible; the generation is correct
> until the affected position is attended, then diverges.**

This is the same shape as before with the CPU-cache claim removed, and it now points at the graph executor's
dependency/barrier logic rather than at allocation flags.

## Where the record already contains a candidate defect in exactly that logic

`graph_exec.c`'s barrier insertion:

    for each dependency d of node n:
        idx = node_index_map[d->node_index];
        if (idx == UINT32_MAX) continue;          // <- dependency SKIPPED: no hazard, so no barrier
        if (set_contains(barrier_index_set, idx)) { record_dependency_barrier(); ... }

and the graph is partitioned with **semaphores at partition boundaries** intended to carry the cross-partition
ordering (`graph_exec.c:1781-1792`: *"We need semaphores at partition boundaries for synchronization. Multi-stream
partitions need join semaphores"*).

Two things I established earlier and did **not** finish: the `UINT32_MAX` skip is *correct* for nodes genuinely
outside the map (`node_index_map_count = max_node_index + 1`, so it should never fire), **but the lookup in
`graph_exec.c:1527` is done inline with no bounds check** where the helper `graph_analysis.c:139` has one — and
the capacity-overflow path in the node-index set is fail-safe (more barriers), so it is not the defect.
**The cross-partition semaphore path is the part I have not audited, and it is now the prime suspect**: the
prefill and the decode are different graph executions, and the ordering between *them* is not obviously covered
by either the in-partition barriers or the partition-boundary semaphores.

## What this changes about the plan

The confirming arm changes shape. It is no longer "do the two reads disagree" — that has been answered
negatively everywhere it has been asked. It is:

**instrument the K/V buffer at the divergent decode step and compare it against the same buffer's content
immediately after the prefill wrote it.** If they differ, the write was lost or late — a device-side ordering
defect, and the fix is a barrier or a semaphore in the graph's prefill→decode handoff. If they agree, the K/V is
intact and the fault is downstream of it, in the attention or the residual path.

That is a strictly stronger test than the one this directory pre-registered, and it was only findable by taking
the negative seriously.
