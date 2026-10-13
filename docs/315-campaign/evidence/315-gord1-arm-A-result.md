<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Graph-reorder A/B arm A (315gord1a) — the control, 4/4 eviction arms faulted, BOTH faces

Pre-registered at `101ab04` (`315-gord1-preregistration.md`). Partial-result note: this file is written
with arm A complete and arm B not yet run (it was deferred because a peer agent's campaign `315ids1` held
the GPU; the launcher waits for it to release).

## Invocation (re-runnable)

    BIN=/home/bcloud/wt/315coherent-build/bin \
    EXTRA_ENV="GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1" \
    bash ~/wt/315-disc.sh 4 315gord1a

Reordering **ON** (shipped behaviour; `GGML_HRX_GRAPH_ORDER` unset). Console:
`strixhalo:~/wt/315gord1a-console.log`. The arm server's environment was checked to confirm the knob was
absent.

## Result — the driver's own summary

    cell q: OK=4 FAULT=0 OOR=0 INVALID=0
    cell e: OK=0 FAULT=4 OOR=0 INVALID=0      <- EVERY eviction arm faulted
    cell c: OK=4 FAULT=0 OOR=0 INVALID=0

Per-arm, all in-regime at `gtt_start` 97-98G, zero OOR, zero INVALID:

    round 1 arm q: OK     gtt 97G->98G   side=0/0
    round 1 arm e: FAULT  Face A (nan=1)  gtt 98G->79G  side=1/1
    round 1 arm c: OK     gtt 97G->98G   side=12/12
    round 2 arm e: FAULT  Face A (nan=1)  gtt 98G->79G  side=1/1
    round 2 arm c: OK     gtt 97G->98G   side=12/12
    round 2 arm q: OK     gtt 97G->98G   side=0/0
    round 3 arm c: OK     gtt 97G->98G   side=12/12
    round 3 arm q: OK     gtt 97G->98G   side=0/0
    round 3 arm e: FAULT  Face B (500)     gtt 98G->98G  side=1/1
    round 4 arm q: OK     gtt 97G->98G   side=0/0
    round 4 arm e: FAULT  Face B (500)     gtt 98G->98G  side=1/1
    round 4 arm c: OK     gtt 97G->98G   side=12/12

## Both faces, in one arm

The four cell-`e` faults split **2 Face A / 2 Face B**:

    r1e: Face A — "HRX returned NaN logits: count=154880 of 154880 ... continuing=no"
    r2e: Face A — same
    r3e: Face B — requests 5,  Content-only 500
    r4e: Face B — requests 3,  Content-only 500

This is the strongest form of the control: the intervention that is supposed to produce the fault
produces **both** acceptance-criterion faces in a single arm run, at 4/4 in the test cell, with every
control arm clean.

## Reading against the pre-registration

The pre-registration's branch structure compares arm B against this. Arm A is exactly the "control that
must fault" it named: cell `e` faults, cells `q` and `c` do not, all arms in-regime with no OOR/INVALID.
Arm A is therefore **valid**, and the comparison with arm B will be informative.

## What this does NOT establish

Nothing about the hypothesis. Arm A only establishes that the rig reproduces on this build with the knob
unset. The question — whether disabling the reorder removes the fault — is arm B's, and is not yet run.
