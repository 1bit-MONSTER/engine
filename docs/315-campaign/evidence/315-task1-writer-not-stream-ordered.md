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

# RESULT (task 1): the writer of `result_norm` is NOT ordered by `context->stream`

`315-PREREG-sync-between-reads.md` ran and landed on its **second** pre-declared row, which removes the last
surviving ordering candidate.

## The measurement

Build `~/wt/315fp-build` with the sync-between-reads instrument, verified by content — `hrx-f32-sync-after=1`
and `memcpy_stable=1` in the source **before** configuring and in `libggml-hrx.so.0` **after** building. Run
`315sr`, `-ub 512`, in-regime, eviction forcer, `GGML_HRX_TENSOR_FP=result_norm,GGML_HRX_BUFFER_TRACE=1`.
**The arm faulted** (Face A), so this is a faulting arm.

| quantity | value |
|---|---:|
| records | 312 |
| diverged (`memcpy#1 != memcpy#2`) | **82** |
| of those, `after_sync_stable=1` (a fresh sync stops it) | **0** |
| of those, `after_sync_stable=0` (a fresh sync does **not** stop it) | **82** |
| fresh sync changed the value (`memcpy#2 != after_sync`) | **82** |

Sample, verbatim (a converged record):

```
[hrx-f32-sync-after] call=94 name=result_norm bytes=8192 memcpy_hash=f6f120302c6fe587
                     memcpy2_hash=f6f120302c6fe587 after_sync_hash=f6f120302c6fe587 after_sync_stable=1
```

## The pre-declared reading, applied

> read#1 ≠ read#2 **and** `hash_after_sync` ≠ read#2 → the writer is **not ordered by `context->stream` at
> all** — a fresh synchronise does not stop it. That removes the `execute`-returns-without-waiting candidate
> and points at a second writer of the arena slot (§53/§59) or something outside the stream.

**82 of 82 divergences are that row.** So:

* **`GraphExecutor::execute` returning without a wait — while `backend_synchronize` covers only
  `context->stream` — is REMOVED as the explanation.** Waiting for the stream does not stop the write, so
  the write is not on that stream.
* The survivor is the other candidate `315-what-the-synchronise-drains.md` lists: **a second writer of the
  reused arena slot** (§53/§59 established the slot is reused), or something else outside `context->stream`
  entirely.

## The fact worth isolating: the sync itself changes the value

`memcpy#2 != after_sync` in **all 82** divergent records — so `ggml_backend_synchronize` is not merely
failing to stop the movement, it is *an event after which the bytes differ*. That is consistent with what
`mark_stream_synchronized()` does — it **publishes** queued host writebacks, i.e. it runs the deferred CPU
`memcpy` — but this run cannot say that is what happened here, because the writeback probe measured the
*engine's* read (`backend_get_tensor_async`) and this is the *fingerprint's* sync. Two readings, and the
measurement does not separate them:

1. the sync publishes a queued writeback into this range (`mark_stream_synchronized`); or
2. the sync's wait takes time, and during it the (unstream-ordered) writer writes again.

Both are testable from this data if a future instrument reports the writeback queue at this point. Recorded
as the next isolating question, not as a claim.

## Bounds

* One faulting arm; the diagnostic eviction-forced cell, so **no arm of this run counts toward a campaign.**
* 230 of 312 records did **not** diverge, so the divergence is intermittent within an arm, as before.
* It does not show the divergence *causes* the fault — nothing here ties the two; the arm faulted and 82
  records diverged, and the divergence is not present in every faulting arm (this instrument saw it here, but
  the earlier arms' rates differ).
* The instrument's own validity: the converged records read identical hashes on all three reads, so the
  instrument is not manufacturing divergence.

## The falsifier, restated as the task contract requires

From `315-PREREG-sync-between-reads.md`:

> Any faulting arm in which `hash_after_sync` differs from read#2 **and** a subsequent instrumented check
> shows the sync completing normally — i.e. where the writer can be shown to be a *third* party rather than
> an ordering gap.

**Not triggered.** This run has the first half (`hash_after_sync != read#2`, 82 of 82) but **not** the second:
it carries no instrumented check of the sync's completion, so it does not establish that the writer is a
third party rather than an ordering gap. The pre-registered reading for that first half is the one applied
above — the writer is not ordered by `context->stream`, which removes the `execute`-returns-without-waiting
candidate — and the falsifier's second half is what would additionally have excluded *any* ordering gap.
