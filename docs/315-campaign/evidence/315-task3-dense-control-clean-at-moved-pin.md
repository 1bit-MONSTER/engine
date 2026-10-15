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

# task 3, dense half: CLEAN at the moved pin — and the script's verdict was wrong a third time

## The control

Qwen3-8B-Q4_K_M (a **dense** model, not MoE), on the new-pin build:

    build   ~/wt/315logwait-build/bin          source pin 47a8fe929
    model   ~/models/Qwen3-8B-Q4_K_M.gguf      5027783488 bytes
    device  HRX0, -ngl 99, -fa on, -c 4096, -np 1, --no-webui
    load    50 mixed-length requests, max_tokens 32, temperature 0
    run     06:38:42 -> 06:41:10, server up at gtt=37GiB

Run with `~/wt/315dense3.sh`, which is the recorded `315dense2.sh` with the build pointed at the new-pin
build and the response counter fixed from the recorded line-counting defect before the run.

## The measurement — clean, and matching the baseline exactly

| measure | recorded baseline (2026-10-13, old pin) | at the moved pin `47a8fe929` | verdict |
|---|---|---|---|
| responses | 50 of 50 | **50** | unchanged |
| distinct content | 1 | **1** | unchanged |
| NaN guard | 0 | **0** | unchanged |
| HSA_STATUS_ERROR / Queue error | 0 | **0** | unchanged |
| Content-only parse failures | 0 | **0** | unchanged |

Every one of the fifty responses carries identical content, so the model is deterministic and coherent under
the same detector shape the faulting arms use, on the build that carries the moved pin.

## The script said `RESULT: FAULT or incomplete`. It was wrong, and here is exactly why

```
RESULT: FAULT or incomplete - inspect /home/bcloud/wt/315dense-newpin/server.log
```

The verdict line is computed as

    if [ "$NAN" -eq 0 ] && [ "$HSAE" -eq 0 ] && [ "$CONTENT" -eq 0 ] && [ "$RESP" -ge 45 ]; then CLEAN ...

and each of the first three is built with

    NAN=$(grep -c "HRX returned NaN logits" "$D/server.log" 2>/dev/null || echo 0)

`grep -c` prints `0` **and exits 1** when there are no matches, so the `|| echo 0` fallback runs as well and the
variable becomes **two lines**. Reproduced directly on the run's own log:

```
NAN=[0
0]  HSAE=[0
0]  CONTENT=[0
0]  RESP=[50]
[ "$NAN" -eq 0 ] -> FALSE (non-integer: multiline value)
grep -c "HRX returned NaN logits" server.log   # prints 0, exits 1
```

The numeric comparison errors out on the newline, so the `&&` chain is false and the script takes the `else`.
**The fault verdict is produced by the counting mechanism, not by anything in the run.** Counting each thing
directly, on the same log, gives 0, 0, 0, 50, 1 with a 402-line server log.

## This is the third instance of one class, and the second one this turn

The record already logs two of these (`315-crit4-control-result.md`: the line-based `"content"` count that read
50 responses as 1, and the same line-based count again earlier). This turn added two more:

1. **the op-test half**: without the HRX runtime environment all eight tests exit **0** while printing
   `no HRX0 device, skipped` — an exit-status gate would have reported **8/8 PASS** on a machine that ran
   nothing (see `315-task3-op-test-unchanged-at-moved-pin.md`);
2. **this run**: `grep -c ... || echo 0` yields `"0\n0"`, a numeric test on it errors, and a clean run prints
   `FAULT or incomplete`.

Both have the same root: **a count is read as a verdict without checking that the count means what the verdict
assumes.** Recorded plainly because the pattern, not the individual bug, is what keeps costing this
investigation its readings — and because in both cases the misleading verdict pointed at a *fault* that
measurement does not support.

No instrumentation was weakened: the script is left as it ran so the verdict it produced stays reproducible,
and the correction is the direct measurement recorded above.
