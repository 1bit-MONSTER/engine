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

# PRE-REGISTRATION: the f32 read-path sync twin, read before the fault is caught

Written and committed **before** the hunt below produced a faulting arm, so the reading cannot be
fitted to the result. The hunt was already launched when this was written; what is pre-committed here
is the *interpretation*, not the run.

## Why this measurement, and why now

`315-mechanism-refuted-on-ids-family-and-the-precise-gap.md` identifies the precise gap:

> The sync twin is implemented **only in the i32 branch** of the probe… **So the f32 tensors —
> `l_out`, `node_2948`, `result_output` — have no sync twin at all**, and those are precisely the
> tensors that carry the NaN and the 256-byte-granular corruption. The record contains 1.16M pairs for
> the family that never corrupted and **zero** for the families that did.

The twin that produced the record's "the two read paths agree" negative compares, for the same tensor at
the same call:

1. the plain `ggml_backend_tensor_get` memcpy — what a host-visible buffer gets, **no iface call and no
   synchronisation**;
2. the same bytes through `hrx_synchronous_d2h`.

That comparison has never been taken on any f32 tensor. `patch-input-f32-sync-twin.py` adds it with a
**full-content hash** rather than only a NaN count, so a divergence confined to a tensor's interior is
visible.

## What was built and what is being run

* Source: `~/wt/llama-315copy` at pin `e44c9d01a`, patch `patch-input-f32-sync-twin.py` applied
  (anchors all matched; `hrx-f32-sync` = 1, `head_n` = 5 in the .cpp). Backup of the pre-patch file at
  `~/wt/ggml-hrx.cpp.bak-pre-f32twin`.
* Build: `~/wt/315fp-build`, `make -j16`, rc=0, marker verified present in `bin/libggml-hrx.so.0`
  (`hrx-f32-sync` = 1) — an uninstrumented binary would silently measure nothing.
* Emitted line: `[hrx-f32-sync] call=… name=… bytes=… nan=… of … ne0=… first=… row=… hash=…H
  memcpy_hash=…H nan_agree=0|1 hashes_equal=0|1`, where `hash` is the **synchronising** read and
  `memcpy_hash` is the plain one.
* Run: `315-fp-hunt.sh <315fp-build/bin> 15 315f32` with
  `EXTRA_ENV="GGML_HRX_TENSOR_FP=result_norm,result_output,l_out,ffn_out"` and `UB=512`, in-regime via
  `315-instr-capture.sh`. It stops at the first arm with a fault on either face.

**Instrument smoke test (clean, single short request), recorded because it fixes the baseline:**
`[hrx-f32-sync] call=94 name=result_output bytes=619520 nan=0 of 154880 ne0=154880 … hash=744e1b4b2f61b983
memcpy_hash=744e1b4b2f61b983 nan_agree=1 hashes_equal=1` — 20 lines, every one agreeing. So on a clean
run the two paths are identical; any divergence found in a faulting arm is not baseline behaviour.

## Pre-declared readings

| observation on the **faulting** arm | reading |
|---|---|
| `hashes_equal=0` on a tensor that also carries NaN / the corruption | **The plain memcpy read returns different bytes than the synchronising read of the same tensor at the same call.** The read path into the engine is not ordered against the device's write. The memory-visibility family is live **on the f32 family that actually corrupts** — the one place it has never been tested. |
| `nan_agree=0` with `hashes_equal=0` | Same, and the disagreement includes the NaN extent itself: the engine is reading a materially different buffer than the synchronised view. |
| every line `hashes_equal=1 AND nan_agree=1`, including on the faulting arm | **The two read paths agree on the f32 tensors too.** The read-path/staging-visibility family is refuted on the family that matters, not only on the i32 family — and it closes for the same reason §46/§55/§56 closed it, now with the measurement that was missing. |
| no `hrx-f32-sync` lines at all on a faulting arm | The instrument did not reach the tensor (name mismatch or the call did not hit the printed path). **Not a result** — the arm is uninformative and must be re-run with `head:<N>`/`tail:<N>`. |

**The stop rule is the hunt's own:** it stops at the first fault, so the faulting arm is whatever it
catches. No arm pruning, and a clean-only hunt (no fault within 15 arms) is recorded as
`315HUNT_DONE` — an uninformative outcome about the fault, **not** evidence for either reading above.

## Declared confounds, and what each does and does not permit

1. **The two reads are sequential**, so a divergence means the buffer *changed between them*. That is
   either (a) a stale host cache line, or (b) the device writing the buffer concurrently with the host's
   read. **This measurement cannot separate (a) from (b)** — but it does not need to for the claim it is
   being used for: both are a read that is not ordered against the producer, and both are defects. What
   it *would* refute is the claim that the read is safe.
2. **Arena reuse** (§53, §59) means a by-name read may return another tensor's bytes. That does **not**
   affect this test: the twin compares two reads of the *same buffer at the same call*, so aliasing
   changes what both reads return together. This is the one instrument in the record that is immune to
   the aliasing objection.
3. **`UB=512`** is the issue's own high-rate configuration (11/12 on the record), used so a fault is
   cheap to reach. If a divergence appears only under it, that is a statement about that configuration
   and must be re-checked at the default `-ub` before being reported as the in-regime mechanism.
4. **Hash equality does not prove the reads are correct** — only that they agree. Agreement refutes
   staleness; it does not certify the data.

## The falsifier

One line: a faulting arm in which every `[hrx-f32-sync]` record has `hashes_equal=1` and
`nan_agree=1` falsifies "the f32 read path is the mechanism". Conversely a single `hashes_equal=0`
record on a NaN-bearing tensor confirms it for that tensor.
