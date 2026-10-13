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

# Criterion-4 control re-run: ALL EIGHT HRX TESTS PASS, matching the baseline

Ran 2026-10-13 17:09 on the current pin (`e44c9d01a`), from `~/wt/v315-build`, under
`systemd-run --user --scope -p MemoryMax=6G`. This is the objective's **"no regression: engine CI green, and
the dense-model control plus the MUL_MAT_ID op-test unchanged relative to baseline"**.

## Result

    --- test-hrx-buffer            PASS   (5 passed, 0 failed)
    --- test-hrx-mul-mat-id-k32    PASS
    --- test-hrx-mxfp4             PASS
    --- test-hrx-moe-split         PASS
    --- test-hrx-attention-sink    PASS
    --- test-hrx-decode-stride     PASS
    --- test-hrx-hadamard          PASS   ("test-hrx-hadamard: PASS" printed by the binary)
    --- test-hrx-fa-masked-v       PASS

**8 of 8 pass.** The `test-hrx-ops` and `test-hrx-loom-jit` binaries the baseline notes as not compiling at
this revision are absent from the build, consistently with that note.

## Direct comparison against the recorded baseline

| measure | baseline (2026-10-09) | today (2026-10-13) | verdict |
|---|---|---|---|
| HRX buffer cases | `5 passed, 0 failed, 5 total` | `5 passed, 0 failed, 5 total` | **unchanged** |
| individual cases | `backend_buffer`, `host_buffer`, `host_transfer`, `host_staging`, `host_weight_cache` all `[PASS]` | same five, all `[PASS]` | **unchanged** |
| MUL_MAT_ID `mxfp4` nmse | 1.23e-05 … 1.58e-05 | 1.34e-05 … 1.61e-05 | **same range, no drift** |
| kernels selected | `ggml_mul_mat_id_decode_f32_wave64`, `ggml_mul_mat_id_f32_f32_wmma` | same two | **unchanged** |

The `nmse` figures are the important half: they are a numeric fingerprint of the reduce and id-resolution
path, so "PASS" plus an unchanged `nmse` range means the op-test has genuinely not drifted, not merely that
it avoided failing.

## What this does and does not establish

- **Establishes**: the engine's own regression controls are **unchanged on the shipped pin today**. That is
  criterion 4's claim about the controls, and it is now measured rather than assumed.
- **Does not establish criterion 4 as MET**: the criterion pairs this with a fix to compare against
  ("baseline-vs-fix on the same harness"), and **no fix has landed** — the one prepared candidate is refuted.
  With nothing landed, these tests confirm the baseline the future fix must match, and nothing more.
- Also **not run**: the "dense-model control". The objective names it alongside the op-test; this run covers
  the HRX test suite and the MUL_MAT_ID op-test. The dense-model control should be named explicitly before it
  is claimed either way.

## Harness note

Two attempts were needed and both failures are recorded rather than hidden: the first waited on the wrong
clear-window rule, and the second **matched no tests at all** because I used guessed binary names
(`test-hrx-backend-buffer` etc.) for cases that actually live inside `test-hrx-buffer`
(`315-engine-ci-baseline.md`). The corrected run is `315crit4c.sh`.

---

# The dense-model half of criterion 4: staged

The objective's criterion 4 names **"the dense-model control plus the MUL_MAT_ID op-test unchanged relative
to baseline"**. The op-test half is done (8/8 HRX tests pass, `MUL_MAT_ID` nmse baseline-matched, above). The
dense half is now staged.

## What is staged

`~/wt/315dense.sh` — runs a **dense** model on HRX0 under the same detector shape the arms use and records
whether it faults:

    model  : Qwen3-8B-Q4_K_M (4795 MB) - a DENSE model, not MoE, so it lacks the routed-FFN/id-producing
             structure the fault's leading family involves
    device : HRX0, -ngl 99, -fa on, c 4096, --no-webui
    load   : 50 mixed-length requests (the detector's shape), max_tokens 32, temperature 0
    detects: nan_guard lines, HSA_STATUS_ERROR/Queue error, Content-only parse failures, and a response count
    gate   : waits for a sustained 120 s clear window (another agent launches campaigns back-to-back)

## Why it is a meaningful control

The recorded finding is that dense models **do not** exhibit this fault while GLM (MoE, routed FFN) does. So a
**clean** dense run is the expected baseline, and its value is as a *negative control* for the harness: it
shows the detector, the regime composition and the build do not fault everything in sight, which is what makes
the GLM arms' high eviction-cell rate attributable to the model/shape rather than to the apparatus.

## Status

Staged, not run — arm B of the alignment test holds the box. It will be run after arm B releases it, and its
result recorded either way.
