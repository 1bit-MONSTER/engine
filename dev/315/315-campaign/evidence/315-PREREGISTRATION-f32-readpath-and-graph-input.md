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

# PRE-REGISTRATION: the f32 read-path twin and the graph input — the two measurements that can still settle #315

**Committed before any arm runs. Nothing in this directory is scored against these criteria, and no arm is
pruned. An arm that does not fault is INVALID for these questions and will be reported as such, never as "no
difference".**

## Why these two and not something else

Three instrument-integrity defects have already voided most of the record's per-node evidence, and a fourth —
buffer aliasing — has voided the rest:

1. **Buffer aliasing.** The fingerprint reads tensors **after compute** (its own code comment says so:
   *"fingerprint named graph tensors after compute"*), and it reads a **buffer**, so every tensor name bound to
   a shared allocation reports one allocation's final content. Measured: `node_2948` ≡ `ffn_shexp-41`
   (identical hash `65f4b97ca74c`), `l_out-42 ≡ l_out-44`, and 47 `l_out-N` names collapse to 4 hashes. **So
   "the onset is layer 41" is not a measurement** and has been retracted.
2. **No f32 sync twin.** The i32 branch reads each tensor twice — plain memcpy, and `hrx_synchronous_d2h` — and
   comparing them produced the record's "the two read paths agree" negative. **The f32 branch has no twin**, and
   f32 tensors are the ones that carry the fault.
3. **The graph input has never been observed.** `inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
   n_embd_inp, ubatch.n_tokens)` is created **unnamed**, so the name list cannot match it and `tail:<N>` never
   reaches it — it sits at the **head** of `graph->nodes`. Across 567 observed tensor names there is no `inp_*`,
   no embedding, no token tensor; the earliest thing ever observed is `l_out-0`.
4. **`fp_nan_only`** hides clean tensors, so no arm has ever shown what is *upstream* of a NaN.

Together these mean the record has no valid per-node localization, no f32 read-path comparison, and no
observation of the one tensor upstream of everything — while the objective's own mechanism chain is
*"graph input published to host by deferred non-coherent staging writeback → re-uploaded to device → device
computes from stale input"*, and the record's own PR #90 was an **input-path** defect.

## The instrumentation

`dev/315/315-campaign/tools/patch-input-f32-sync-twin.py` — verified: applies cleanly to
`ggml-hrx.cpp` (4 anchors, exactly one occurrence each), is idempotent, and the patched translation unit
**compiles with the exact build flags of the real build directory** (`-fsyntax-only`, exit 0).

It adds two things, both inert unless their env var is set:

- **`hrx-f32-sync`** — for every f32 tensor the existing fingerprint already reports, read the same bytes
  through `hrx_synchronous_d2h` and print `nan`, `first`, `row`, a **full content `hash`**, and the comparison
  flags `nan_agree` and **`hashes_equal`** against the plain-memcpy line. A content hash, not just a count, so a
  divergence confined to the tensor's interior is visible.
- **`head:<N>`** — select the first `N` nodes, symmetric with the existing `tail:<N>`, so the **unnamed graph
  input** is reachable. Combine with `GGML_HRX_TENSOR_FP_NANONLY` **unset** so clean tensors print too.

## The arms, and what each decides

**Arm 1 — read-path twin on the tensor that carries the fault.**

    GGML_HRX_TENSOR_FP="tail:16"        (or a name list)
    GGML_HRX_TENSOR_FP_NANONLY unset
    -> compare [hrx-tensor-fp] vs [hrx-f32-sync] for the same call+name

**Arm 2 — the graph input.**

    GGML_HRX_TENSOR_FP="head:8"  GGML_HRX_TENSOR_FP_NANONLY unset
    -> is the input clean or NaN, and does its content hash agree between the two read paths?

**Arm 3 — the in-regime trigger**, so the arm is inside the declared regime gate rather than ambient, and the
question is asked where the fault actually appears.

## Scored outcomes, fixed now

**On the f32 twin (Arm 1):**

| observation | conclusion |
|---|---|
| `hashes_equal=0` on a faulting arm, and the differing region is 256-byte aligned | **read-path staleness CONFIRMED** for f32 — the plain memcpy returns different bytes than a synchronised read; the fix is the visibility operation or a coherent staging allocation |
| `hashes_equal=1` on a faulting arm, with `nan>0` | **read-path staleness REFUTED for f32** — the corruption is genuine device state, and the read path is not the mechanism |
| `hashes_equal=0` but the arm is clean | reported as such: a read-path divergence that does not produce a fault is a different finding and does not satisfy the criterion |
| arm does not fault | **INVALID** for this question. Report as INVALID. |

**On the graph input (Arm 2):**

| observation | conclusion |
|---|---|
| input `nan > 0` while the first reader is clean | the input is the seed; the mechanism is upstream of the graph |
| input clean, and `l_out-0` / the first named node is NaN | the seed is inside layer 0's computation, not the input |
| input clean, everything downstream clean, yet the arm faults | the fault is not in the forward path at all — it is in the decode loop or the output path, and Face B's reclassification applies |
| arm does not fault | **INVALID.** Report as INVALID. |

## Standing corrections this pre-registration carries

- **"Arena knobs and transient reuse are already falsified on the record"** (the objective's own premise) is
  **contradicted by measurement**: reuse is not merely present, it is pervasive and it is what made the
  read-pattern/onset result look meaningful. Any acceptance argument that depends on reuse being falsified
  should be revisited with the user.
- **Face B's stated definition** ("invalid-UTF-8 output surfacing as HTTP 500 parse failures") has **zero
  occurrences** on the record: invalid UTF-8 in 0 arms, HTTP 500 in 0 arms. The operative marker is
  `common_chat_peg_parse: unparsed Content-only output`, and roughly half of those arms are the model declining
  to emit JSON rather than any corruption. **This needs `/goal-tweak` before a ≥60-arm campaign can be scored
  against it** — it is the user's decision, not mine.

## What is NOT claimed

No mechanism is named. No fix is proposed. This pre-registers two instruments and the outcomes that would
confirm or refute the only two hypotheses the record can still test, and it states the criteria before the data
exists.
