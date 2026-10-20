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

# PRE-REGISTRATION: does a synchronise BETWEEN two reads stop the `result_norm` change?

Written while no box run is authorised, so the experiment is fixed before it is launched. It targets the one
**measured, unexplained fact** this issue still has.

## The fact it explains

`result_norm` changes between two **consecutive identical plain reads** in a faulting arm: `memcpy_stable=0`
on 75/75 (`315-read-twice-result-buffer-moves-under-the-reader.md`) and 128/128
(`315-cellA-measured-noncoherent-divergence.md`). Three reads at the same call gave three different hashes.
Two explanations have been removed by measurement:

* **not the deferred host writeback** — 626 reads at the engine's own logits read in a faulting
  eviction-forced arm, **0** with a pending writeback
  (`315-candidate-deferred-writeback-FALSIFIED.md`);
* **not cache behaviour of a non-coherent persistent mapping** — the same twin diverges at a measured `0x52`
  (13.7 %) and at a measured `0x56` (14.8 %), so the memory type is not the variable
  (`315-cellB-divergence-independent-of-memory-type.md`).

What is left is: **something writes that range between two adjacent reads**, and the question is whether that
writer is ordered by the stream the harness already synchronises before reading.

## The experiment

One instrument change on the existing twin build, and one eviction-forced faulting arm (the trigger the
record replicated at 7/10 vs 0/22). No new build configuration, no regime change.

**Instrument:** in the f32 twin, insert a `backend_synchronize(backend)` **between read #1 and read #2**, and
report a third hash. The line already carries `hash` (the synchronising read), `memcpy_hash` (plain read #1)
and `memcpy2_hash` (plain read #2); this adds `hash_after_sync` = a plain read taken **after** a fresh
synchronise.

**Configuration:** `GGML_HRX_TENSOR_FP=result_norm,GGML_HRX_BUFFER_TRACE=1`, `-ub 512`, in-regime, eviction
forcer running, stop at the first fault. **Diagnostic cell** — no arm of it may count toward any campaign.

## Pre-declared readings

| observation in the faulting arm | reading |
|---|---|
| read#1 ≠ read#2 **and** `hash_after_sync` == read#2 | the writer **is** ordered by the stream: the reads were racing work the sync would have covered, and the earlier divergence is a read-vs-`stream` race. This makes `GraphExecutor::execute`'s non-waiting return the live candidate. |
| read#1 ≠ read#2 **and** `hash_after_sync` ≠ read#2 | the writer is **not ordered by `context->stream` at all** — a fresh synchronise does not stop it. That removes the `execute`-returns-without-waiting candidate and points at a second writer of the arena slot (§53/§59) or something outside the stream. |
| read#1 == read#2 (no divergence) | the arm is uninformative for this question; record it and note that the divergence did not occur under this instrument, which would itself be new (it occurred in every previous faulting arm). |

The second row is the interesting one: it is the only reading under which the remaining `stream`-ordering
candidate dies.

## Falsifier

Any faulting arm in which `hash_after_sync` differs from read#2 **and** a subsequent instrumented check shows
the sync completing normally — i.e. where the writer can be shown to be a *third* party rather than an
ordering gap.

## Cost and what it buys

One instrument edit, one rebuild, and **one** faulting arm (the forcer produced a fault in 1 arm for both the
writeback probe and the cell-A run). If it lands on the first row, the issue has a named ordering gap; on the
second, one more candidate is removed and the arena-reuse family becomes the survivor. Either way it is a
bounded result, which is what the re-scope proposal asks criterion 1 to accept.
