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

# How the two live threads compose (recorded to avoid duplicated work)

Two agents are working this goal on the same branch. This records the split, both parties' state, and how
the two findings constrain each other — because the composition is stronger than either thread, and neither
should be re-run by the other.

## Thread A — the recorded-text constraint (peer; COMPLETE at n=14)

Across 14 arms including a foreign replication (`315gord1b`), **every faulting arm's output for the
identical prompt matches the clean arms' answer**, with 0 of 6 faulting arms differing, and the
request-by-request trace shows the first requests character-identical before the arm dies.

The constraint this yields: **the fault is a terminal whole-buffer event, never a changed computation.**
That eliminates the stale-index / wrong-expert class for this workload. Its recorded limit: short
(96-token) answers, one prompt.

## Thread B — the deferred host writeback (this session; static chain COMPLETE, probe queued)

`add_host_writeback` merely queues a host `memcpy` (`command-program-executor.cpp:152`) and
`mark_stream_synchronized` performs it later (`:38`). The queued source is a
`HOST_LOCAL | DEVICE_VISIBLE` = `0x52` **non-coherent** mapping (`runtime/host-memory.cpp:1287-1310`), and
the logits are structurally expected to take this path (`staging.download = access.write &&
!binding.graph_input` holds; the trace records `host_visible=1 direct_host_binding=0`). The replay queues
the download and **returns without flushing it** (`:2047`, `insert_barrier=false`). The code states the
hazard itself: "Uploading the same host memory before that has happened would read the previous
contents … so nothing else guarantees the ordering."

The instrument exists and has never run: `GGML_HRX_WRITEBACK_FINGERPRINT=1` logs `rec_hash` (at queue time)
vs `sync_hash` (at execution), with **`rec_eq_sync=0`** meaning the queued source changed while queued.

## Why they compose (the peer's argument, and it is correct)

    Thread B answers WHERE the bad read is queued.
    Thread A answers WHEN it must fire and WHAT SHAPE it must have.

A deferred read of a staging buffer that eviction moved inside the window since its device copy gives
**exactly** the shape Thread A requires: a whole-buffer terminal event, not a changed computation — because
the queued memcpy copies a whole region that was correct when the device wrote it and stale by the time it
is read. A per-request or index-level corruption would show as differing output, which Thread A's n=14
rules out.

Neither thread alone is a cause. Together they are the **tightest hypothesis pair in the record**: a
located, deferred, non-coherent read whose failure mode matches the measured shape of the fault.

## What each party agreed to do

- Thread A will not run more arms for its constraint: n=14 with a foreign replication is enough, and more
  would only re-confirm.
- Thread B's probe (`~/wt/315wb-probe.sh`) is queued, awaiting release of the GPU by Track A's campaign
  series. Thread A independently verified that all four arm builds carry the knob
  (`315coherent-build`, `315instr3-build`, `315ids-build`, `315fix-build`), so the probe will run.

## The single measurement that would settle Thread B

One GLM decode with `GGML_HRX_WRITEBACK_FINGERPRINT=1`, reporting the `rec_eq_sync` tally. If
`rec_eq_sync=0` occurs at all during decode, the deferred window is real and populated; under the eviction
rig it would then be the leading cause candidate, with the discriminating fix being to flush the writeback
before the consuming sync rather than at the conditional upload guard.
