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

# Cell B vs cell A: the read-to-read divergence does NOT track the memory type

The two-cell comparison `315-two-cell-memory-type-preregistration.md` pre-registered, run. The
coherence explanation my own correction introduced (`315-persistent-mapping-correction.md`) is closed.

## The two cells

Both cells: **the same build** (`~/wt/315fp-build`), **the same instrument** (the f32 fp/sync twin plus the
read-twice control), `-ub 512`, in-regime, `GGML_HRX_BUFFER_TRACE=1` so the type is measured, and the
eviction forcer running so a faulting arm arrives in one. They differ in **one environment variable**:

| cell | configuration | **measured** host-visible type | records | diverged | rate | fault |
|---|---|---|---:|---:|---:|---|
| **A** | default | `direct_host_binding=0  memory_type=0x52` (non-coherent) | 932 | 128 | **13.7%** | `nan_guard=1` |
| **B** | `GGML_HRX_USE_UNIFIED_MEMORY=1` | `direct_host_binding=1  memory_type=0x56` (coherent) | 426 | 63 | **14.8%** | `nan_guard=1` |

All divergences in both cells are `memcpy_stable=0` (128/128 and 63/63).

`GGML_HRX_USE_UNIFIED_MEMORY` gates `device_ctx->use_direct_host_bindings` (`ggml-hrx.cpp:1175`, logging
*"direct coherent host bindings enabled"*), so the coherent cell is reached on the same build with the
instrument untouched — the A/B difference is the memory type and nothing else.

## Pre-declared reading applied

The pre-registration's table:

> divergences in **both** cells → the divergence is not the memory type. It is then either independent of
> coherence, or both cells achieved the same type — which the trace in the same log settles, so the two
> cases are separable within this run rather than by argument.

**Divergences in both cells, and the same-log traces show the types are different (`0x52` vs `0x56`).** So
the second branch is excluded by measurement: the divergence is **independent of coherence**.

The rates are indistinguishable at this n (13.7% vs 14.8%; Fisher exact on 128/932 vs 63/426 is far from
significant), so this is not "coherence reduces it" either. Both cells also faulted, which is consistent
with §13's finding that a coherent allocation does not remove the fault.

## What this settles, and what it hands back

* **Closed:** the third explanation for `memcpy_stable=0` that `315-persistent-mapping-correction.md`
  introduced — that the CPU's cached view of a non-coherent *persistent mapping* changes between two reads.
  Coherent memory has no such mechanism, and the divergence is unchanged. The persistent-mapping observation
  itself remains true (`buffer_get` does read a mapping created at allocation and never invalidated); it is
  simply not what moves between the reads.
* **Restored, but narrowed:** `memcpy_stable=0` again means **the buffer changed between two identical
  reads**, i.e. something wrote it — with the writer now having two fewer candidates: not the deferred
  host writeback (falsified by a counter, `315-candidate-deferred-writeback-FALSIFIED.md`) and not cache
  behaviour of a non-coherent mapping (falsified here by the A/B).
* **Handed back:** the remaining writers are the ones `315-what-the-synchronise-drains.md` lists —
  `GraphExecutor::execute` returning without a wait while `backend_synchronize` covers only
  `context->stream`; a second writer of the reused arena slot (§53/§59); the replay path (least likely,
  §4047348 traces it to a full wait).

## Bounds

Both cells are **diagnostic** (eviction-forced, which the record measured at 7/10 faults against 0/22
controls). **No arm of either cell may be counted toward the objective's ≥60 in-regime clean-arm campaign.**
n is two arms per cell; what makes this decisive is that the prediction was a *ratio comparison between two
configurations*, and the measured types differ by exactly the bit under test while the ratio does not move.
