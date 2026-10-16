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

# PRE-REGISTRATION: the in-regime acceptance campaign for the logits-read wait

Committed **before** the campaign runs. The mechanism and the landed fix are in
`315-MECHANISM-unwaited-async-logits-read.md`; the fix is fork commit `47a8fe929`, and the engine pin on
`1bit/315-preregistration` moves `e44c9d01a → 47a8fe929` (`95238dc`), forward-only.

## What is being tested

**One prediction, stated before the result:** with the completion wait restored on the logits path, the
faults stop. If they do not stop, the mechanism is wrong regardless of how well it explains the record.

## Cells

| cell | build | purpose | arms |
|---|---|---|---|
| **F — fix** | clean, `47a8fe929`, **no instruments** | the acceptance cell | **≥60, in-regime** |
| **C — control** | clean, `e44c9d01a` (the pin's previous revision), no instruments | positive control: shows this harness in this regime can still fault | as many as the box budget allows, same regime, same window |

Both cells run on **the same harness** (`315-detector.py` shape set), **the same regime gate**, and
interleaved in one window, so a clean fix cell cannot be explained by a regime or harness difference.

`GGML_HRX_BUFFER_TRACE` and any fingerprint instrument are **off** in both cells: this measures the fault,
not the mechanism.

## Regime gate

Frozen at `gtt_start ∈ [95, 110] GiB`, the gate the record already uses, read from the harness's own
`gtt_start` line. **Every counted arm must be inside it**; an arm outside is recorded as out-of-regime and
does not count toward either cell's n (neither as clean nor as fault).

Composition caveat, already measured (§58): a neighbour moves GTT in ~17 GiB steps, so 80 GiB of
neighbours lands the arm at ~98 GiB (in gate) and one more neighbour lands it at 115 GiB (out). **98 GiB is
this box's in-regime point** and is the expected landing.

## Both-face detectors, and what counts as a fault

* **Face A:** the sampler guard line `HRX returned NaN logits` in the arm's `server.log`; and
  `HSA_STATUS_ERROR` / queue-error text (the `hsa` counter the harness already computes).
* **Face B:** `detector.json`'s `utf8_failures` and `parse_failures` — `detector.json` is authoritative,
  not the console and not the face tally (the record's own rule, because a Face-B fault's text is invalid
  UTF-8 and makes `grep` treat the log as binary).
* An arm that dies for a **harness** reason (`transport_failures`, `TimeoutError`, an arm the detector
  reports as truncated) is recorded as **invalid**, never as a pass — the record's own rule, and it is why
  the previous runs recorded truncated arms separately.

**Fault = any of the four counters non-zero on a complete arm.**

## Stopping rule (pre-registered)

Stop cell F at the **first fault**, exactly as the record's acceptance rule does. A fault in cell F ends the
campaign as **not met** and the arm is reported with its `gtt_start` and face. Cell C is not stopped by a
fault — a fault there is the positive control succeeding.

## Acceptance, and the power statement

**Acceptance: ≥60 counted in-regime arms in cell F, zero faults on both faces.**

Power, stated the way the objective asks: with zero faults in 60 arms, the one-sided 95 % upper confidence
bound on the fault rate is `1 − 0.05^(1/60) ≈ 4.9 %`. So **60 clean arms ⇒ ~95 % confidence the true
in-regime rate is below 5 %.** The bound is reported as this number, not as "the fault is gone".

**The vacuous-pass guard, pre-registered:** a clean cell F is only meaningful if the harness can still
fault in this regime. If **cell C also produces zero faults in ≥20 complete in-regime arms**, the campaign
is reported as **underpowered** — *not* as a pass — because a harness that cannot fault cannot show a fix.
This is the trap the issue records as already sprung once.

## Budget

The objective caps strixhalo at **≤12 h total**. Cell F at ≥60 arms is ~5-6 h at the measured
~5 min/arm; cell C is capped at whatever remains, with a floor of 20 arms for the vacuous-pass guard. If
the budget cannot fund both, cell F's ≥60 is funded first and the shortfall in cell C is reported as a
stated limitation rather than silently dropped.

## What will be reported

Every arm with its `gtt_start`, tag and verdict; the counted n per cell; faults by face with the faulting
arm's own log path; the CI above; the control cell's result; and the exact commands and revisions to
re-run it. Any arm excluded, and why.
