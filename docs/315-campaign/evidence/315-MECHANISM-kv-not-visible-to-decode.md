<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# MECHANISM (Face B): the prefill's KV writes are not reliably visible to the decode steps that read them — dose-response 15×, decaying over ~7 steps, immune at index 0

This document assembles the strongest chain the record supports and separates **measured** from **inferred**,
item by item. It is the first mechanism candidate that explains all seven of the phenomenon's properties at
once, and it is falsifiable with one arm.

## Measured (archive-only, no GPU)

**1. The trigger is causal and quantified.** Per-decode-step hazard by cell, over 2,372 requests and 98,325
run-steps, comparing every run against the global consensus for its own prompt:

| cell | arms | runs | run-steps | events | hazard | early hazard (idx ≤ 7) |
|---|---|---|---|---|---|---|
| **eviction (forced)** | 6 | 95 | 4,184 | 5 | **0.001195** | **0.004043** |
| quiet | 8 | 420 | 18,032 | 3 | 0.000166 | 0.000000 |
| probe-`p*l` | 35 | 1,433 | 63,458 | 9 | 0.000142 | 0.000705 |
| control | 6 | 286 | 12,651 | 1 | 0.000079 | 0.000000 |

**Forced eviction is 15.1× the control hazard.** This is the record's original trigger result (28/29 eviction
arms vs 4/57 controls) reproduced at a per-step resolution and with a rate.

**2. The fault is in decode and the prefill's own output is immune.** Index 0 diverges **0 times in 2,259 runs**;
all 18 events are at index ≥ 1. Index 0 is the logits the **prefill** produces; every index ≥ 1 is a decode step.

**3. Under forced eviction the hazard is front-loaded.** Early hazard 0.004043 vs the cell's own 0.001195 overall
— **4×** — and the aggregate early excess is significant (11 of 18 events in steps 1–7 against 3.8 expected,
p ≈ 0.002). **In the control and quiet cells the early hazard is exactly zero**: without forced eviction the
faults happen later, if at all.

**4. The fault is finite and enters at a clean boundary.** All 18 events have `nan = 0`, a prefix that
bit-matches the consensus, and a generation that runs to full length. Nothing accumulates; nothing crashes.

**5. The runtime performs no cache management at all, and the memory is declared non-coherent.** Exhaustive
counts over `libhrx/src`: `invalidate_range` 0, `flush_range` 0, `mapping_invalidate` 0, `mapping_flush` 0; the
only `flush` names are `stream_flush`/`context_flush` (command submission) and `fflush`. And every host-imported
buffer is hardcoded `HOST_LOCAL | DEVICE_VISIBLE = 0x52` — **non-coherent** — on all three allocation paths,
with the coherence flags merely stored. The record's own backend comment records that the tensors on this path
*"live in a host-visible, non-coherent arena"*.

## Inferred (a hypothesis, stated as one)

> **The prefill writes the prompt's K/V into that non-coherent host-visible arena. The following decode steps
> read it. Between the two there is an execution-order dependency but no operation that makes the writes
> visible — no invalidate, no flush, anywhere in the runtime. So a decode step can attend to KV bytes the
> prefill wrote but which the CPU/GPU have not made visible, and it computes logits from a mixture of current
> and stale K/V. Those values are real numbers from a real earlier write, so the logits are finite and
> plausible; the generation is correct while the affected position is not yet being attended, then diverges.**

This accounts for each measured property, and no other candidate on the record accounts for more than two of
them:

| property | explained by |
|---|---|
| forced eviction 15× | eviction pressure is exactly what determines whether pending writes have landed |
| **index 0 immune, 0/2259** | index 0 is computed **in the same pass that writes the KV** — no cross-pass read to lose |
| decay after ~7 steps | the prefill's writes complete; once landed, the read is correct |
| front-loading only under eviction | in control/quiet the writes land promptly, so exposure is rare and late |
| finite, no NaN | stale K/V are valid numbers, not poison |
| clean prefix | the affected position is not attended until later steps |
| HTTP 500 only 2 of 18 | a 500 needs the degeneration to persist long enough for the parser to reject it |
| no invalid UTF-8 ever | the garbage is U+FFFD, a **valid** codepoint; no UTF-8 check can fire |

## The discriminating experiment

The hypothesis predicts something no alternative does: **at the divergent decode step, the KV bytes read
through the plain host memcpy and through `hrx_synchronous_d2h` should differ on the faulting arm.**

- **differ** → an ordering/visibility defect, as hypothesised, and the fix is the visibility operation or a
  coherent KV allocation;
- **agree** → the wrong values are genuine device state, the race is on the device side, and this hypothesis is
  refuted.

The record identifies the arm, request and decode step for all 18 events for free, and
`docs/315-campaign/tools/patch-input-f32-sync-twin.py` (verified: applies cleanly, idempotent, compiles with the
build directory's exact flags) reads exactly that pair, with a content hash so an interior divergence is
visible. **One arm, on a step the archive already names.**

## Fix direction (forward-only)

Both candidates are additions on top of the pin, neither is a revert:

1. **Honour the coherence flag** — stop hardcoding `0x52` in
   `iree_hal_streaming_memory_allocate_owned_host_import_with_context_mode` and pass
   `IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_COHERENT` for the KV/staging allocations. Small, local, and it
   stops the runtime from declaring memory non-coherent while assuming it is visible.
2. **Or** add the visibility operations the runtime never performs: an `invalidate` before a host read and a
   `flush` after a host write, using IREE's mapping-visibility APIs, which exist for exactly this and which this
   runtime never calls.

## Honesty note

This is the most complete candidate the record has produced, and it is still a **hypothesis**: items 1–5 in the
measured section are facts, the mechanism paragraph is inference, and it rests on a read-path divergence that
**has not been observed on any f32 tensor** because no arm has ever made that comparison. Two earlier candidates
from this same line of work were over-claimed and retracted (the layer-41 onset, and the fixed 32-entry
boundary); this document states its inferential step explicitly so that it can be the third thing checked rather
than the third thing retracted.
