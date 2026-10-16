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

# PRE-REGISTRATION: the f32 twin against a measured memory type — a two-cell comparison

Written before either cell is run. Both builds already exist on the box; nothing new is compiled.

## Why this, and what it decides

`315-f32-twin-result-divergence-on-the-faulting-arm.md` established that the f32 read-path twin
diverges 42 times on a faulting arm and 0 times in 554 clean-arm records, but it could not say *why*,
because the two reads are sequential and because neither arm recorded the memory type of the region being
read. A later reading of the record's own buffer traces showed the same two host-visible sizes appearing
under **both** memory types — 44 allocations at `0x52` (non-coherent) against 96 at `0x56` (coherent) —
with no 8192-byte host-visible allocation, so `result_norm` is a view inside one of those buffers and its
type is a **variable, not a constant**.

That makes one comparison decisive, and it has never been run:

> **Run the same f32 twin, on the same box, in the same regime, against a configuration whose regions are
> `0x52` and one whose regions are `0x56`. If divergence follows the memory type, the read-path
> divergence is a coherence defect. If it does not, the memory type is not what the twin is measuring.**

## The two cells

| cell | build | host-visible memory type | note |
|---|---|---|---|
| **A** | `~/wt/315fp-build` (the twin build, default) | expected `0x52` | the cell the 42-divergence result came from; its type must be **measured**, not assumed |
| **B** | the unconditional-`HOST_COHERENT` build already on the box (`~/wt/315coherent-build`, the lever-1 cell) | expected `0x56` | the twin must be **re-applied to that tree** if it does not already carry it |

Both cells run `GGML_HRX_TENSOR_FP=tail:16` (position, so the logits node is covered regardless of name —
the name filter is an exact per-item match and missed it in the two arms that produced the result) and
`GGML_HRX_BUFFER_TRACE=1`, so the achieved memory type is **recorded in the same log as the divergence**.

`GGML_HRX_BUFFER_TRACE=1` is not optional. Without it this experiment is the previous one again: a
divergence count with no memory type attached.

## Pre-declared readings

| observation | reading |
|---|---|
| divergences present in A and absent in B, with `0x52` and `0x56` respectively recorded in the same logs | **the divergence is a coherence defect**, and the memory type is the variable that carries it. This is the first statement in the record tying the twin's divergence to a measured type. |
| divergences in **both** cells | the divergence is not the memory type. It is then either independent of coherence, or both cells achieved the same type — which the trace in the same log settles, so the two cases are separable within this run rather than by argument. |
| divergences in **neither** cell | the 42 divergences were a property of the earlier arms' configuration (the `-ub 512` high-rate cell or the regime), not of the memory type. That does not refute the earlier result; it relocates what it is about, and the earlier run's own logs are then the only record of it. |
| the achieved type in A is `0x56` (not `0x52`) | **cell A is void** — the two cells are the same configuration and any A/B difference is noise. Record it and re-run A on a build whose regions are confirmed `0x52`. |

## What this does NOT decide, stated in advance

* **It does not test lever 1.** Section 13 falsified *unconditional `HOST_COHERENT` as a fix for the
  fault*, with the output region measured coherent (`hv56=1`) in the arm that faulted. This experiment
  asks a narrower question — whether the **twin's divergence** tracks the memory type — and it is
  deliberately not framed as a fix test. A positive result here would **not** overturn section 13; it
  would say that coherence is the variable behind the divergence, which is a different claim from
  coherence being sufficient for the fault.
* **It does not establish causality for the NaN.** The divergence is on `result_norm`, one tensor upstream
  of the logits, and the logits read is only now being covered by position.
* **n is small by construction.** This is a diagnostic, two cells, not the ≥60-arm campaign. If it comes
  back positive the campaign question is whether a `0x56` configuration also clears the **fault**, which
  is section 13's question and already has a recorded answer to be re-read rather than re-run blind.
