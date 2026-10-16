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

# Re-scope proposal, 2026-10-15 — for `/goal-tweak`

Supersedes `~/315-PROPOSED-RESCOPED-OBJECTIVE.md` (2026-10-13), which predates this session. Written because
the named mechanism was falsified by its own pre-registered campaign and no mechanism is now named; the
executor cannot edit the objective, so this is text for the owner to apply.

## Why a re-scope, in one paragraph

The objective's criterion 1 asks for a **named mechanism**, and §"if it is, the fix is rejected" makes it a
gate rather than a goal. Over this issue's life, **eleven candidate families have been eliminated by direct
measurement** — coherence (§13, `hv56=1` and still faulted), consumer ordering (§15), FA masked-row (§26),
the visibility family (§59), the MoE id tensors (1.16M paired reads, 0 disagreements), the pin move itself
(the only changed kernel is never dispatched: 60/60 FA dispatches are the non-split kernel, 0/944 logs
mention `decode_split`), the deferred host writeback (626 reads, 0 pending, in a faulting arm), the
persistent-mapping/cache explanation (two-cell A/B: `0x52` 13.7 % vs `0x56` 14.8 %), and now the unwaited
async logits read (its own acceptance campaign stopped on the first fix arm with a **Face B** fault at
`gtt_start 98G`). Each elimination is recorded with a re-runnable command and its output.

What has never been produced is a positive: a mechanism that survives its own discriminating experiment.
Continuing to hold criterion 1 as a gate means the remaining work is unbounded while the alternatives that
were *supposed* to be ruled out already are — that is the state the re-scope is for.

## Proposed changes

**1. Criterion 1 — accept a bounded, evidenced negative.**

> A named, falsifiable mechanism for both faces supported by a discriminating experiment that rules out the
> surviving alternatives, **or** a bounded negative result: every candidate family in the surviving list
> tested by its own pre-registered discriminating experiment and reported, with (a) each elimination's
> re-runnable command and observed output, (b) the surviving families stated explicitly, and (c) one
> measured, unexplained fact recorded as the hand-forward. A bounded negative is an acceptable outcome and
> is not reported as a mechanism.

**2. Criterion 3 — separate "a campaign was run" from "a fix was verified".**

> A pre-registered in-regime campaign is run on **the configuration under test**, with the regime gate,
> the arm count, both-face detectors and the stopping rule fixed before it runs, and is **reported with its
> result — met or not — including the arm tally, the CI and the power statement.** The ≥60-clean-arm
> acceptance is required only if a candidate fix survives criterion 1's discriminating experiment; when the
> fix under test faults, the campaign's value is the fault it produced, and it is reported as a failed
> acceptance rather than as an unrun arm.

**3. Criterion 2 is unchanged and is already satisfied.** The forward-only change is landed
(`47a8fe929` on `1bit-MONSTER/llama.cpp` branch `1bit/315-logits-read-wait`; pin `e44c9d01a → 47a8fe929` on
`1bit/315-preregistration`, `95238dc`, `git merge-base --is-ancestor` true). It is a correct change for the
reason its comment gives — a backend that implements `get_tensor_async` loses the generic layer's
synchronise — even though it is not sufficient to stop the fault. **A re-scope must not be read as
retracting it.**

**4. Criterion 4 (no regression) is unchanged** and still outstanding: engine CI plus the dense-model and
MUL_MAT_ID controls. It needs box time, which is currently capped and paused.

**5. Criterion 5 is met in substance already** — `KNOWN-ISSUES.md` carries the mechanism, the fix, the
campaign and the falsification, with evidence paths — and its status line stays **OPEN**, because nothing
has been shown to remove the defect.

## The bounded negative, as it would be reported under the new criterion 1

**Eliminated by direct measurement, each with its command and output:** the eleven families listed above,
all in `docs/315-campaign/evidence/`.

**Surviving families, stated explicitly:**

1. An open writer of the `result_norm` slot: two consecutive *identical* plain reads of it return different
   bytes in a faulting arm (`memcpy_stable=0`, 75/75 and 128/128), and that writer is neither the deferred
   host writeback nor cache behaviour of a non-coherent mapping. Candidates that remain in
   `315-what-the-synchronise-drains.md`: `GraphExecutor::execute` returning without a wait while
   `backend_synchronize` covers only `context->stream`; a second writer of the reused arena slot (§53/§59);
   the replay path (least likely — §4047348 traces it to a full wait).
2. §51's rank-sort non-bijection: verified at both ends in the IR (ordered comparisons make two NaN elements
   tie; the consumer takes the expert index by `index.assume`, not a clamp), with its **precondition
   unmeasured** — NaN at the argsort's input.

**The one measured, unexplained fact to hand forward:** `result_norm` changes between two consecutive
identical plain reads in a faulting arm, at a measured `0x52` mapping and equally at `0x56`, with no pending
host writeback at the engine's own logits read.

## What this re-scope does NOT do

It does not declare the defect gone, does not retire the KNOWN-ISSUES entry, and does not convert a
falsification into a mechanism. It moves criterion 1 from "produce a positive" to "test the surviving
alternatives and report a bounded result", which is what the remaining work actually is.
