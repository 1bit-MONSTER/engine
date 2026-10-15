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

# RESULT: the f32 read-path twin diverges on the faulting arm and only there

Read against the pre-registration committed before the run
(`315-f32-sync-twin-preregistration.md`). The pre-declared reading for `hashes_equal=0` on a faulting arm
was: *"the read path into the engine is not ordered against the device's write"*. That is what was
observed. Two limits are recorded below and neither is smoothed over.

## The measurement

Build `~/wt/315fp-build` at pin `e44c9d01a` with `patch-input-f32-sync-twin.py` applied (marker verified
in `libggml-hrx.so.0`), in-regime via `315-instr-capture.sh`, `-ub 512`,
`GGML_HRX_TENSOR_FP=result_norm,result_output,l_out,ffn_out`.

Each record compares, for the same tensor at the same call:
`hash` = the same bytes through `hrx_synchronous_d2h`; `memcpy_hash` = the plain
`ggml_backend_tensor_get` memcpy that a host-visible buffer gets (no iface call, no synchronisation).

| arm | verdict | gtt_start | f32 records | **`hashes_equal=0`** | tensors with records |
|---|---|---:|---:|---:|---|
| `315f32-1` | **clean** | 110G | 554 | **0** | result_norm |
| `315f32-2` | **FAULT** (`HRX returned NaN logits: count=154880 of 154880, contiguous=yes`) | 110G | 426 (1278 by name) | **42** | result_norm only |

**Zero divergence in 554 clean-arm records; 42 in the faulting arm, every one on `result_norm`.** This
is the first time this twin has been taken on any f32 tensor — the record had 1,160,376 paired reads for
the i32 family and **zero** for the f32 family that carries the corruption.

A representative divergent record, verbatim:

```
[hrx-f32-sync] call=36574 name=result_norm bytes=8192 nan=0 of 2048 ne0=2048 first=-1 row=-1
               hash=9cb2244e4ae24383 memcpy_hash=8892a34c1b335845 nan_agree=1 hashes_equal=0
```

`nan_agree=1` on all 42: both reads report the same NaN count (zero). **The disagreement is in finite
bytes, not in the NaN extent.**

## Limit 1 — the divergence is NOT on the tensor that carries the NaN

`result_output` is the logits tensor (619,520 B = 154,880 floats), and it is exactly the tensor the
guard reports as wholly NaN. **It produced zero fingerprint records in the faulting arm.** `l_out` and
`ffn_out` produced zero as well — only `result_norm` matched, 1278 records. So:

* the twin fired on `result_norm`, one tensor upstream of the logits;
* the logits read itself was **not observed** in this arm, so this result says nothing directly about
  the read of the tensor that corrupts.

`result_output` *does* match the name list — the clean smoke test produced `name=result_output
bytes=619520` records — so its absence here is an arm-specific fact (the guard logs `continuing=no`, the
server aborts at the fault) and is recorded as an **instrument gap in this arm**, not as agreement.

### The gap, diagnosed from the record afterwards — and the fix

`result_output` is reachable by the instrument, and the record already contains a whole-tensor-NaN
reading of it. In a run that selected nodes by **position** rather than by name (`tail:<N>`, which
bypasses the name filter — the probe's own note says position is "the only way to walk the head-end
pipeline without guessing tensor names"):

```
[hrx-tensor-fp]  call=103644 name=result_output op=MUL_MAT bytes=619520 nan=154880 of 154880 ne0=154880 first=0 row=0
[hrx-finite-fp]  call=103644 name=result_output finite=0 nan=154880
```

So the logits tensor is **wholly NaN, `finite=0`**, which is what the guard reports — and the probe *can*
see it. Two consequences for the next run:

1. **Use `tail:<N>` (or `head:<N>`), not the name list, for the logits.** The name filter is an exact
   match per comma-separated item (`hrx_fp_name_listed`, `ggml-hrx.cpp:604-621`), while the graph's
   layer tensors carry a `-<layer>` suffix — the record's own fingers show `ffn_moe_topk-9`,
   `l_out-0`, `ffn_moe_weighted-46`. That fully explains why `l_out` and `ffn_out` produced nothing here;
   for `result_output` the exact name does match in other runs, so its absence in these two arms is not
   explained by the filter and is stated as unresolved rather than guessed at.
2. **The logits reading must be taken in a faulting arm.** The record's `result_output` line above is
   from an arm whose `nan=154880 of 154880` matches the guard, but it is a *post-graph by-name* read and
   therefore carries §53/§59's aliasing caveat. The f32 twin's own comparison is immune to that caveat
   (both reads target the same buffer at the same call), which is why extending the twin to the logits
   node by position — rather than adding another by-name read — is what the next instrument change should do.

## Limit 2 — the two candidate causes are still not separated

The two reads are **sequential**, so a divergence means the buffer changed between them:

* **(a)** a stale host cache line on the plain-memcpy path — the path is the difference; or
* **(b)** the device writing `result_norm` while the host reads it — the buffer is the difference.

The pre-registration declared this test cannot separate them, and it does not. Both would be a read not
ordered against its producer, but only (a) is a memory-visibility defect.

### A code fact that bears on the split, read after the run

The fingerprint does **not** read an in-flight buffer. `ggml-hrx.cpp`, in `graph_compute`, gates the whole
fingerprint on a successful execution and then synchronises **before** any read:

```cpp
if (result.success()) {
    // engine#315: order the fingerprint's own reads. ... without this the per-node lines are not a
    // snapshot of one instant and the per-node attribution is invalid.
    ggml_backend_synchronize(backend);
    const char * want = std::getenv("GGML_HRX_TENSOR_FP");
```

So both reads are taken with the device drained **as far as `ggml_backend_synchronize` drains it**, which
makes cause (b) — a writer racing the reader *between the two reads* — materially less likely than the
pre-registration allowed for. It does not eliminate it: whether HRX's `backend_synchronize` is a full
stream drain is exactly what sections 55-56 argue at length for the *consumer's* path, and this note does
not re-derive it for the probe's. The read-twice control remains the arbiter, and it is already running.

It also sharpens what a divergence would mean. `result_norm` is read through the host-visible path: the
probe's own comment records that the tensors it names on the MoE path live in the host-visible
**non-coherent** arena (44048640 bytes, from the buffer trace), which is the memory type where a missing
invalidate or flush is a defect rather than a no-op.

**The control that separates them already exists and is not yet applied**: `patch-f32-read-twice-control.py`
(commit `71f4101`) adds a second plain memcpy immediately after the first and before the sync read,
reporting `memcpy2_hash` / `memcpy_stable`. Reading: `memcpy#1 == memcpy#2` and `!= sync` ⇒ (a);
`memcpy#1 != memcpy#2` ⇒ (b). It is validated (anchors matched against the live source; the patched TU
passes `-fsyntax-only` under the build's own flags) and was deliberately held back so this run's
pre-registration would not be mixed with a changed instrument.

## What this does and does not establish

* **Establishes:** on this box, in this regime, the faulting state and f32 read-path divergence coincide
  exactly in the two arms measured (0/554 vs 42/426), on a tensor the engine reads through the
  unsynchronised memcpy path. The earlier "the two read paths agree" negative is a statement about the
  **i32** family; it does not transfer to f32, and the record never claimed it did — this is the gap it
  named.
* **Does not establish:** that the divergence is staleness rather than a read racing the writer (Limit 2),
  that it occurs on the logits tensor (Limit 1), or that it is causal rather than co-symptomatic. Two
  arms is two arms; the clean/faulting split is clean but n = 2.

**Next run, in order:** (1) apply the read-twice control and re-run in-regime until the same fault, to
settle Limit 2; (2) fix the `result_output` gap (use `tail:<N>` rather than the name list, so the logits
node is covered even when the arm aborts) so the logits read is observed in the faulting arm.
