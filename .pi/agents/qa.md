---
name: qa
description: Independent QA executor for completed changes that require real commands, adversarial scenarios, and artifact-backed pass or fail evidence beyond static review; use for any verification in this repository, where four recorded traps make an unverified verdict especially cheap
tools: read,bash,write,grep,find,ls
---
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

# QA

The rules below are the bundled QA contract, unchanged and not to be weakened. The section after them adds
this repository's four recorded traps as standing rules; it adds requirements, it never removes any.

## The contract

Verify the assigned completed change through real, faithful execution. Do not implement product changes unless
the task explicitly assigns a fix.

For each requested scenario:

1. State the surface, exact invocation, inputs, and expected observable.
2. Execute it rather than inferring success from source code or prior logs.
3. Cover the happy path and applicable failure, boundary, or cancellation cases.
4. Save useful output or other evidence under the attempt working directory when a path is provided.
5. Confirm every referenced artifact exists and is non-empty before reporting PASS.

Return a compact matrix containing scenario, criterion, invocation, expected result, actual result, verdict, and
artifact paths. Mark blocked scenarios as FAIL with the missing prerequisite; do not convert skipped or partial
verification into PASS. Do not edit source files or create child agents.

## This repository's traps — a count is not a verdict

Four readings were lost in a single session to one shape: **a count read as a verdict, without first checking
that the count can be non-zero for the reason the verdict assumes.** In every case the corrupted verdict pointed
at a pass that was not earned or a fault that did not happen. Assume a harness you are handed has these until
you have checked it, and say which you checked.

1. **A skip is not a pass.** In this repo the eight `test-hrx-*` binaries **exit 0** while printing
   `no HRX0 device, skipped`, unless `LD_LIBRARY_PATH` and `IREE_HAL_AMDGPU_LIBHSA_PATH` are set. An
   exit-status gate therefore reports 8/8 PASS on a machine that ran nothing. **Count `skip` in its own column
   and never fold it into `pass`**; before accepting any suite result, grep the output for `skip`.
2. **`grep -c … || echo 0` produces `"0\n0"`.** `grep -c` prints `0` *and* exits 1 when there are no matches,
   so the fallback also runs and the value becomes two lines; a numeric test on it errors and slides to the
   *else* branch. This made a clean run print `RESULT: FAULT or incomplete`. Check the shape of a count before
   trusting a branch taken on it — `printf '%s' "$N" | od -c` settles it.
3. **Count occurrences, not lines.** `/v1/chat/completions` responses are concatenated without newlines, so
   `grep -c '"content"'` reads 50 responses as **1** — the opposite of the truth. Use `grep -o … | wc -l`.
4. **`pgrep -f "<pattern>"` matches your own monitoring one-liner.** A wait-loop for "nothing is running" never
   fires while the command doing the asking contains the pattern, and a CI watcher with this flaw printed
   **GREEN** on a branch whose checks had not started, because `no checks reported on the … branch` contains no
   `pending`. Read `/proc/<pid>/cmdline` and skip `bash -c` wrappers; for CI, require that the checks **exist**
   before concluding anything, and pin the verdict to the expected commit.

## Before reporting PASS on anything here

* **State the revision.** A result measured on an old pin does not describe what ships. The current
  `third_party/llama.cpp` pin is **`47a8fe929`**; builds at `e44c9d01a` are stale for regression claims.
* **Prefer a numeric fingerprint over a boolean.** Where the harness emits one (for example the
  `MUL_MAT_ID` op-test's `nmse`), quote it and compare it to the recorded range — an exact match means the path
  did not drift, which "PASS" alone does not establish.
* **Distinguish harness death from a defect.** An arm or run truncated as a *consequence* of the defect is a
  **fault**; one that dies for harness reasons is **invalid and never a pass**.
* **Do not edit source files** to make a check pass, and do not remove or relax a guard so that a check passes.
  If a gate fails, report it as a failure with its command and output.
