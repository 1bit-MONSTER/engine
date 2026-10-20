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

# FALSIFIED: the deferred-host-writeback candidate — zero pending writebacks across a faulting arm

`315-candidate-deferred-host-writeback-publication.md` named a mechanism and pre-declared its falsifier:

> **Falsifier:** if `has_pending_host_writebacks()` is false on every read in a faulting arm, this candidate
> is dead and the writer is something else.

The falsifier fired on the first arm.

## The measurement

`patch-writeback-read-probe.py` reports `graph_replay_state.has_pending_host_writebacks()` at
`backend_get_tensor_async` — the engine's own async logits read, the one that does **not** synchronise —
gated on `GGML_HRX_WRITEBACK_READ_PROBE`. Build `~/wt/315fp-build`, marker `hrx-wb-read` verified in
`libggml-hrx.so.0`.

Run `315wbe`, `-ub 512`, in-regime, **with the record's replicated eviction forcer running concurrently**
(`~/wt/evictonly 500`, which launches no kernels and only mprotect-churns `hipHostRegister`'d memory — the
intervention the record measured at 7/10 faults against 0/22 controls). Two earlier runs at the ambient
rate produced 9 clean arms and no faulting arm at all, which is the expected outcome at ~5%/arm; the
forcer was used to reach a faulting arm in one, which is a **diagnostic** use and not an acceptance cell.

```
315HUNT_FAULT arm=1 out=~/wt/rt-det/315wbe-1-GLM-4.7-Flash-Q4_K_M
  arm 1 verdict: gtt_start 98 nan=1 hsa=0 utf8=0 parse=0 wb_records=0
  probe census:  reads = 626    pending_writebacks=1  →  0
```

**A faulting arm — Face A (`nan=1`), in-regime at `gtt_start 98G` — with 626 reads and not one of them
carrying a pending host writeback.**

## What that means

The candidate's entire content was: *the read of the host destination happens while the CPU `memcpy` that
publishes it is still queued, so the reader gets the previous contents.* If the queue is empty at every
read of a faulting arm, that cannot be what the reader is doing. **The candidate is dead, by its own
pre-registered test, and it is not weakened but removed** — this is the direct measurement the record
lacked, not an argument.

Notes that keep the falsification honest:

* The probe is a **read-side observation only**; it changes no ordering, so it cannot have suppressed the
  fault it measured. The arm faulted with the probe on.
* `pending_writebacks=0` is not "the probe never ran": 626 records were produced, and the mechanism's own
  text (`command-program-executor.cpp:873-882`) plus the earlier 9-arm run confirm the counter is wired to
  `pending_host_writebacks_`.
* One arm is one arm. It is a falsification because the prediction was **universal** ("still queued at the
  read"), and a single counter-example refutes a universal claim — not because n is large. A run that
  showed `pending=1` would have been a confirmation of the same strength.

## Where the hunt goes

Not back to visibility or coherence — those remain closed by §13 (coherent `hv56=1` still faulted) and
§59/§55-56 (read-path ordering). The remaining candidates are the ones
`315-what-the-synchronise-drains.md` lists after eliminating the `buffer_stream` path:

1. `GraphExecutor::execute` returns without waiting, and the fingerprint/read sequence is
   `execute → backend_synchronize → read` — so anything read that is not on `stream` is unordered.
2. A second writer of the same arena slot (§53/§59 established the slot is reused).
3. The replay path — least likely, since §4047348 traces it to a full wait.

And it re-points at the two things the record has measured but not explained: the f32 twin's
`memcpy_stable=0` divergence on `result_norm` (`315-read-twice-result-buffer-moves-under-the-reader.md`),
whose writer this note has now ruled *out* being the deferred writeback, and §51's rank-sort defect, whose
precondition (NaN at the argsort's input) is still unmeasured.
