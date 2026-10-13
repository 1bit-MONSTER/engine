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
