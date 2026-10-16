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

# CONTROL RESULT: the buffer moves between two consecutive identical reads — the reader races its writer

Sequel to `315-f32-twin-result-divergence-on-the-faulting-arm.md` (the 42-divergence result) and to
`patch-f32-read-twice-control.py` (the control, whose reading was declared before this run). It settles
the limit that result could not: **staleness or a race.**

## The measurement

Same build lineage, same regime, same `-ub 512`, same name list — the only change is that the twin now
takes a **second plain memcpy** immediately after the first and before the sync read, reported as
`memcpy2_hash` / `memcpy_stable`.

| arm | verdict | records | divergences | `memcpy_stable=0` | `memcpy_stable=1` |
|---|---|---:|---:|---:|---:|
| `315f32c-1` | clean | 672 | 0 | – | – |
| `315f32c-2` | truncated (`TimeoutError`, 12/66 requests — invalid, not a pass) | 681 | 0 | – | – |
| `315f32c-3` | **FAULT** (`HRX returned NaN logits: count=154880 of 154880, contiguous=yes`) | 669 | **75** | **75** | **0** |

**Every one of the 75 divergences is `memcpy_stable=0`.** Verbatim:

```
[hrx-f32-sync] call=56524 name=result_norm bytes=8192 nan=0 of 2048 ne0=2048 first=-1 row=-1
  hash=c03cfd2fe2e7d1c1 memcpy_hash=c887214edc68a3ca nan_agree=1 hashes_equal=0
  memcpy2_hash=3c73bc8f3afb8383 memcpy2_agree=1 memcpy_stable=0
```

**Three reads of the same tensor at the same call, three different hashes.**

## What that means

The pre-declared reading is unambiguous:

* `memcpy#1 == memcpy#2` and `!= sync` ⇒ a **stale host cache line**: the path is the difference.
* `memcpy#1 != memcpy#2` ⇒ **the buffer moved between two identical reads**: the writer is the difference.

The observed case is the second, in all 75 records, with zero of the first. So the read-path divergence is
**not** the engine reading stale coherent-memory bytes. It is **a read that is not ordered against the
producer**: between two back-to-back plain reads of host memory, the contents changed.

## The consequence that matters: the record's own synchronise does not order the reads

The fingerprint does not read an in-flight buffer by design — `ggml-hrx.cpp` (`graph_compute`) calls

```cpp
if (result.success()) {
    // engine#315: order the fingerprint's own reads. ... without this the per-node lines are not a
    // snapshot of one instant and the per-node attribution is invalid.
    ggml_backend_synchronize(backend);
```

and that call is the sole ordering the instrument relies on. **This measurement shows it is not
sufficient**: `result_norm` is still being written while the host reads it, three times over, in a
faulting arm. So the premise the fingerprints were built on — *their own comment's "snapshot of one
instant"* — does not hold on this box in this regime.

This is a constraint on the record, stated as such rather than as a conclusion about the fault:

* **It does not, by itself, name the fault's mechanism.** The reading is on `result_norm`, one tensor
  upstream of the logits, and the logits node still has not been observed in a faulting arm (the name
  filter misses it; `tail:<N>` is the fix, recorded earlier).
* **It does bear on §59's first result.** §59 read "ordering the reads changes nothing" as closing the
  visibility/ordering family. If the ordering primitive available to that instrument does not actually
  drain the stream, that negative is weaker than it was read — the same class of problem the record
  already knows how to catch (it is why the i32 branch's comment insists a read "would report the
  sentinel as never written" on an undrained stream). **§59 should be re-read with this in view**, and it
  is not being retracted here on the strength of one arm.
* **It does not revive lever 1.** Coherence is not what this measures: a stale cache line is excluded by
  the data, not adopted.

## Honest bounds

* `n = 1` faulting arm. The other two arms produced no divergences (one clean, one truncated), and 75/75
  versus 0/0 is a clean split but a small one.
* All 75 divergences are on `result_norm`, all `nan=0 of 2048` on both sides — so the disagreement is in
  **finite bytes**, and nothing here shows the NaN originating in this tensor.
* The control read is taken *after* the first and *before* the sync read, so what it proves is that the
  buffer changes on the timescale of two adjacent reads — consistent with the writer being live
  throughout, but the exact producer was not identified.

**Next measurement, in order:** (1) re-run with `GGML_HRX_TENSOR_FP=tail:16` so the **logits** node is
covered by position in a faulting arm, and `GGML_HRX_BUFFER_TRACE=1` so the tensor's memory type is
recorded; (2) identify what is writing `result_norm` while the host reads it, which is the question the
`memcpy_stable=0` result turns this into.
