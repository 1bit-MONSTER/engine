<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The AQL publish boundary, read at source — and how the graph reorder interacts with it

Found by reading the runtime after another agent's full HRX decompilation/dossier
(`~/work/hrx-system/HRX-SYSTEM-DOSSIER.md` §5.2) pointed at the exact files. This is read at **source** in
`~/work/hrx-system/runtime/src/iree/hal/drivers/amdgpu/`, which is the vendored IREE fork the engine's
`hrx-system` submodule tracks.

## The publication contract, verbatim

`util/aql_ring.h`, `iree_hal_amdgpu_aql_ring_commit`:

    // Commits a packet by writing its header + setup as a single atomic
    // store-release. This makes the packet visible to the CP: the release
    // semantics ensure all prior writes to the packet's fields (and to kernarg
    // memory) are ordered before the CP can read them.
    ...
    iree_atomic_store((iree_atomic_int32_t*)packet, (int32_t)header_setup,
                      iree_memory_order_release);

and `iree_hal_amdgpu_aql_ring_doorbell`:

    // The doorbell is purely a wakeup hint — the CP will process any packet
    // whose header is valid, regardless of whether the doorbell has been rung.

**Two consequences that matter for this issue:**

1. A packet becomes **live to the CP at commit**, not at the doorbell. The doorbell only wakes the CP;
   it does not gate visibility.
2. Therefore, during the commit loop the CP may already be executing packets that have been committed
   while later packets in the same submission are still being constructed.

## Where commits happen, and in what order

`host_queue_command_buffer_block.c`:

    494  commit   (queue-device-event start timestamp packet)
    584  for (uint32_t i = 0; i < emitted_packet_count; ++i)
    587    if (header_type(packet_headers[i]) != INVALID)
    589      iree_hal_amdgpu_aql_ring_commit(packet, packet_headers[i], packet_setups[i]);
    621  commit   (barrier packet)
    625  commit   (profile harvest packet)
    661  iree_hal_amdgpu_aql_ring_doorbell(...)        <- ONE doorbell for the whole batch

So the payload packets are committed **sequentially in emission order**, and there is a single doorbell
at the end. That is the correct batch discipline — but it means **emission order is the order the CP
observes**, because each commit publishes immediately.

## The link to the graph reorder

`record_prepared_command_list(..., ordered.empty() ? prepared.commands : ordered, ...)` is called from
`command-program-executor.cpp:1479` with the list returned by `order_prepared_commands_for_graph`. So when
the reordering engages, **the list recorded — and therefore emitted, and therefore committed in that
order — is the reordered list.**

Established independently by the zero-arm probe (`315-graph-reorder-lever.md`): GLM's decode has a
26-command routed-FFN graph that IS reordered (23 -> 20 barriers) in every layer, transient extent
237,568 B against the 16 MiB gate.

Putting the two together:

    reorder decides the emitted order
      -> emission order is the commit order (`:584-590`)
      -> commit is the point of CP visibility (`aql_ring.h` commit contract)
      -> so the CP may consume reordered packets while later ones are still being committed

The ordering *the graph was recorded in* is deliberately not the ordering the CP sees, for about half of
GLM's per-layer graphs, and the CP can begin on a later-committed packet early. That is exactly the shape
a preemption/eviction window needs to turn into a stale read: the window opens between a packet's commit
and the arrival of a packet whose dependency the reorder moved.

## What this does NOT establish

- It does **not** show a missing dependency. The reorder preserves hazard pairs by construction (it only
  picks a command once all its recorded dependencies are scheduled — `lookahead_order`), so this is not a
  claim that the reorder breaks program correctness.
- It does **not** show the CP-in-flight-during-commit behaviour is unsafe in general. The commit contract
  is a standard AQL publication; the question is whether **eviction during the window** changes what a
  consumer reads, which is what the A/B tests.
- It is **not** a cause. It is the concrete mechanism by which the reorder lever reaches the device, which
  is what made the A/B worth running at all rather than inert.

## Why it sharpens the experiment

It gives the A/B a specific, checkable expectation rather than a direction: if the reorder family carries
the fault, the effect should be attributable to the **26-command graph**, and disabling reordering
(`GGML_HRX_GRAPH_ORDER=program`) should remove it while leaving the 9-command attention graph unchanged
(it is already program-ordered, so the knob cannot affect it).

## Provenance of the read

The files were located via another agent's dossier at `~/work/hrx-system/HRX-SYSTEM-DOSSIER.md`
(copied into this evidence tree as `315-hrx-system-dossier-by-peer-agent.md` for durability), whose §5.2
names `util/aql_ring.h:185-203,255-262,277-296` and `host_queue.c:96-99`. Every citation above was then
**re-read at those files directly** in `~/work/hrx-system/runtime/src/iree/hal/drivers/amdgpu/` rather than
taken from the dossier's prose, and the line numbers in this file are the ones observed on re-read (the
commit loop at `:584-590`, the single doorbell at `:661`, the commit contract in `util/aql_ring.h`).

That dossier is a peer agent's work product, not this goal's evidence, and it is cited as a pointer only.
Its own §11 defects table concerns HIP-layer diagnostics and ABI coverage, not this question.
