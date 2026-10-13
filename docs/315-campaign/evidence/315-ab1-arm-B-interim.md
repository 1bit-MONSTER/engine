<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# A/B arm B — INTERIM: the fix's first eviction-cell arm faults (Face B)

**Status: INTERIM.** Arm B (`315ab1b2`) is still running (round 1 of 4 complete at the time of
writing). This file records the first scored cell-`e` arm so the observation cannot be lost or
re-read after the fact; the full tally and the final reading are recorded when the run completes.

## What is already in

Same binary, same frozen three-cell rig as arm A, knob **ON** (`GGML_HRX_COHERENT_READ=1`), box lock
held, single driver (the collision that killed `315ab1b` is documented separately and this re-run
carries a start guard).

    BIN=/home/bcloud/wt/315coherent-build/bin \
    EXTRA_ENV="GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1 GGML_HRX_COHERENT_READ=1" \
    bash ~/wt/315-disc.sh 4 315ab1b2

    round 1 arm q: OK     nan=0 req=66                       gtt 97G->98G  side=0/0
    round 1 arm e: FAULT  nan=0 http=1 s500=1 unparsed=2     gtt 98G->99G  side=1/1   <- FACE B

Arm validity checked, not assumed:

- **in-regime**: `run.log` records `gtt_start 98GiB mode performance cell=e side_pids=2081364`;
- **the intervention was applied**: `side_pids` names a live `evictonly 500`;
- **the fix path was live**: the arm server's own environment (`/proc/<arm-server>/environ`) carried
  `GGML_HRX_COHERENT_READ=1` while the arm ran;
- **the failure is Face B, by the recorded signature**: `detector.json` gives
  `http_failures: 1`, `first_failure: "HTTP 500"`, `truncated: true`, `requests: 9` (of the expected
  66), and the server logged
  `"code":500,"message":"The model produced output that does not match the expected Content-only format"`.

That is the same Face-B signature the eviction cell produced on the unfixed build (disc1/disc2 and
arm A).

## Why this matters for the pre-registered reading

The pre-registration (`fc7ffd5`) fixed four branches before the run. The branch this first data point
lands in is:

> **B ≈ A (both fault in cell `e` at a similar rate) -> H-fix REFUTED. The `memcpy` is not the site;
> the stale data enters elsewhere.**

Arm A's cell `e` is 4 FAULT / 0 OK. Arm B's first cell-`e` arm is also a FAULT. If the remaining
rounds confirm it, then routing the host-visible read through `hrx_synchronous_d2h` **does not remove
the fault**, and the mechanism chain recorded in `f5ce617`/`5446f19` (bare `memcpy` on a non-coherent
mapping as the corruption site) is falsified as the *cause*.

## What this would and would not falsify

- **Would falsify**: `buffer_get`'s bare `memcpy` as the site where the stale/corrupt data enters, and
  therefore the prepared one-hunk fix as a repair.
- **Would NOT falsify**: criterion 1's trigger-level mechanism. Forced KFD eviction of in-flight work
  raising the fault rate (7/10 vs 0/22, p=3.6e-5, both faces) is an independent, replicated
  intervention result and is untouched by this. The cause site reopens; the trigger stands.
- **Would NOT falsify** the observation that the read path *is* incoherent (`0x52`, no `HOST_COHERENT`)
  — only that fixing it at `buffer_get` is sufficient.

## Not yet claimed

A single round is not "B ≈ A". The claim waits on arm B's full cell-`e` tally. If later rounds are
clean, the reading changes to mixed and must be reported as such rather than resolved post hoc.

---

# FINAL (2026-10-13 ~10:56) — completed at 2 of 4 rounds; the refutation is established

The run was stopped after round 2 rather than continued. Recording why, and what the stopped run does and
does not support.

## Final scored arms

```
  round 1 arm q: OK     nan=0 req=66                     gtt 97G->98G   side=0/0
  round 1 arm e: FAULT  nan=0 http=1 s500=1 unparsed=2   gtt 98G->99G   side=1/1   <- FACE B
  round 1 arm c: OK     nan=0 req=66                     gtt 97G->98G   side=12/12
  round 2 arm e: FAULT  nan=0 http=1 s500=1 unparsed=2   gtt 98G->100G  side=1/1   <- FACE B
  round 2 arm c: FAULT  nan=1 req=34 transport=1         gtt 97G->79G   side=12/12  <- FACE A
  round 2 arm q: OK     nan=0 req=66                     gtt 97G->98G   side=0/0
```

    cell e (the test cell):  0 OK / 2 FAULT   both Face B
    cell q (quiet):          2 OK / 0 FAULT
    cell c (CPU pressure):   1 OK / 1 FAULT   the fault is Face A

## The pre-registered reading

Unchanged from the interim note: arm B's cell-`e` arms **fault with the fix engaged**, while arm A's
cell `e` is 4/4 FAULT. That is the refutation branch fixed at `fc7ffd5`:

> **B ~= A -> H-fix REFUTED. The memcpy is not the site; the stale data enters elsewhere.**

The knob was verified live during the arms by reading the arm server's own `/proc/<pid>/environ`. The
refutation is not dismissible — the vacuity objection (that `hrx_synchronous_d2h` might itself be
incoherent) was checked at source and disproved (`315-ab1-fix-path-is-coherent.md`).

## Why it was stopped at 2 rounds

- **The verdict was settled.** Cell `e` faulted in every round run, so rounds 3-4 could only narrow a
  rate that is not in question. The pre-registration's "B ~= A" branch is met on the evidence already in.
- **Box-time budget.** The record puts this goal turn at ~10-11.1 h of the stated <=12 h ceiling before
  this run; two more rounds would add ~20 min against a margin that is at best thin.

Stopping a pre-registered run early is a deviation and is recorded as one: the pre-registration specified
4 rounds unless the result was already clear, and this invokes that clause explicitly rather than
silently. **The cost is stated**: arm B has 2 cell-`e` arms against arm A's 4, so the comparison is 2-vs-4
in cell `e`, not round-for-round. The direction is unambiguous (arm B faults in 100% of its cell-`e`
arms) but the sample is half the planned size, and no rate estimate should be computed from it.

## What is established, and what is not

- **Established**: the read-path fix does not stop the fault; the corruption is not the CPU observing a
  stale logits page.
- **Established separately**: the `cell c` Face-A fault, whose GTT trace (97 -> 106G) is recorded in
  `315-ab1-cell-c-fault-drift.md` as a single-arm observation.
- **Not established**: the cause site. Still open, with the read path eliminated.
- **Not usable as a rate**: 2 arms. Nothing in this file should be quoted as a frequency.
