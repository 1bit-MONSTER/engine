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

PROPOSED RE-SCOPED OBJECTIVE for goal mv0f7orc-9rf73p (engine#315).
Prepared 2026-10-13 for the user to apply, because the goal tools expose no direct
objective-edit call: the revision is intended to be applied with /goal-resume
(or an equivalent revision path). Nothing here changes the goal by itself.
-->

# Proposed revised objective

Solve 1bit-MONSTER/engine issue #315: name the mechanism behind the intermittent in-regime HRX
GLM-4.7-Flash faults (Face A: NaN logits / HSA_STATUS_ERROR_MEMORY_FAULT; Face B: invalid-UTF-8 output
surfacing as HTTP 500 parse failures), land a forward-only fix in 1bit-MONSTER/engine, and verify it
with a pre-registered in-regime arm campaign of >=60 arms **in the faulting band** showing zero faults on
both faces.

## Three changes, each forced by measured evidence

1. **Cause hunt widened to the ggml-hrx runtime/recorder lane and the in-repo loom kernel corpus.** The
   trigger is already named and replicated; the cause was shown not to be host-side, not in the logits
   read path, and not in the runtime's AQL submission policy. The recorder §33 left as the standing
   question lives in `ggml/src/ggml-hrx/runtime/graph-record-order.cpp`, which the existing boundaries
   *already* include. See `315-rescope-ground-truth.md`.
2. **Criterion 3 redefined to the faulting band** (`gtt_start >= 98G`), because the campaign of record
   (0/70) ran entirely at 97G and ">=60 in-regime arms, zero faults" anywhere in [95,110] is satisfiable
   without exercising the defect.
3. **Criterion 1 re-stated** to separate what is PROVEN (the trigger) from what must be NAMED (the cause),
   so it is falsifiable rather than open-ended. A bounded negative result is an acceptable outcome.

## Established — do not re-litigate

- **Trigger, named and replicated**: forcing KFD evict/restore of in-flight work raises the rate from
  0/22 to 7/10 (Fisher exact two-sided p = 3.6e-5), both faces, verified with the issue's own watcher
  (4134 Q_EVICT / 4168 Q_RESTORE in 25 s vs 0/0 quiet).
- The per-arm eviction **count** correlation is falsified (109.5 vs 254.5) — a count is not a measure of
  band occupancy.
- The **read-path candidate is refuted** by its own pre-registered A/B (cell e: 0/4 FAULT knob-off vs 0/2
  FAULT knob-on, both Face B, knob verified live), and the refutation is not dismissible (staging pool is
  `HOST_LOCAL|HOST_COHERENT|DEVICE_VISIBLE`, `host_queue_staging.c:427`).
- The **whole host-side family is closed**: no host access to the non-coherent `0x52` mapping is on the
  decode path.
- The **runtime's submission policy is closed** as a site for missing dependencies
  (`aql_block_processor.c`: the runtime does not infer memory hazards).
- **One overclaim retracted**: the pooled 97G-vs-98G census does not survive a within-console control
  (`ctl1`: 9.1% vs 6.2%) and must not be cited as a rate.

## Success criteria (re-scoped)

- **A named, falsifiable CAUSE for both faces**, in the widened scope, with a discriminating experiment.
  The named surface: the recorder's `collect`/`update`/barrier ordering and the 16 MiB
  `kMaxTransientExtent` cap in `ggml/src/ggml-hrx/runtime/graph-record-order.cpp`. A bounded, evidenced
  negative result is acceptable.
- **A forward-only change landed on a branch** (ancestry check true, not a revert). The refuted read-path
  patch must NOT be landed.
- **Pre-registered campaign**: >=60 arms, every arm inside the faulting band (`gtt_start >= 98G`, justified
  from the trace rather than the gate label), zero faults both faces, CI and power statement. Arms must
  include the eviction condition or demonstrate they reach the band.
- **No regression**: engine CI green; dense-model control and MUL_MAT_ID op-test unchanged.
- **KNOWN-ISSUES.md** carries the evidence and the met criterion (or the entry retired).

## Boundaries (revised)

- **In scope**: the strixhalo box; the `~/wt` harness (`315-disc.sh`, `evictonly`, `rt-det`); the engine
  pin and docs; and the llama.cpp **ggml-hrx** side — **extended** to `ggml/src/ggml-hrx/runtime/`,
  `ggml/src/ggml-hrx/dispatch/`, and the in-repo corpus
  `ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops/*.loom` (59 files).
- **Out of scope**: AMD driver/kernel changes; the NPU/IRON/Chess lane; the **external** loom/IREE
  compiler toolchain (`~/amd-oss/hyperloom-toolchain`, TheRock/loom trees) — reading an emitted `.loom`
  in the corpus is in scope, changing the loom compiler is not; rewriting recorded studies; any backward
  pin move.
- Still out of scope: declaring the issue solved on "zero NaN" alone.

## Constraints (unchanged, plus additions)

All original constraints stand (pre-registration, <=12 h box time with flock, both-face detection,
re-runnable commands at named revisions, forward-only pins, do not weaken instrumentation). Added:

- **Measure remaining box-time budget from the record before starting GPU work** and prefer bounded code
  reading, which needs no GPU, where it can answer the question.
- **One tag, one driver.** A second Pi session (`01a12f94`, tmux `pi-315`) pursues this same goal; see
  `~/wt/315-COORDINATION.md`.
- **Record corrections in place rather than deleting.** This record already carries two self-corrections
  and a retracted overclaim, and that standard is maintained.

## Verification contract (re-scoped)

- Campaign: >=60 arms in the faulting band, zero faults both faces, log paths and rate with CI.
- Forward-only: `git merge-base --is-ancestor <prev> <new>` is true; cite submodule pin before/after.
- Cause: cite file:line for the recorder's collect/update/barrier ordering and state in falsifiable terms
  whether a pair can be omitted, with reasoning a reviewer can re-run.
- Baseline-vs-fix on the same harness; engine CI success; dense-model control unchanged.

## Why this is worth doing rather than closing

The widened surface is *already in scope*, is bounded code (no regime, no model, no toolchain change), and
has a real discriminating test. It is the one path that could turn the trigger-level mechanism into a
cause and unlock criterion 2.
