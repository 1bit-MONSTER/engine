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

# FALSIFIED: the unwaited-async-logits-read mechanism, by its own pre-registered acceptance campaign

`315-campaign-preregistration.md` stated one prediction before the run:

> **One prediction, stated before the result:** with the completion wait restored on the logits path, the
> faults stop. If they do not stop, the mechanism is wrong regardless of how well it explains the record.

The campaign stopped on the first fix arm, per its own pre-registered stopping rule.

## What happened

Run `315acc`, launched 2026-10-15T06:08, `MINCLEAN=60 CONTROL_EVERY=3 REGIME_LO=80 REGIME_HI=92`,
`315-paired.sh 60 2 2048 <control bin> <fix bin> 315acc`. Console:

```
control_bin /home/bcloud/wt/315fixlog-build/bin   -> version: 10613 (e44c9d01a)
fix_bin     /home/bcloud/wt/315logwait-build/bin  -> version: 10394 (47a8fe929)
detector_sha256 314da4a70f578b85b247d036367be574241f8f0d677a6eb8ee45c60e0c48716e
box lock acquired; neighbours up; gtt 80G -> 98.5G (in gate)

  pair 1 arm c: OK    nan=0 req=66 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0
  pair 1 arm l: FAULT nan=0 req=9  ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0
after pair 1: fix-cell clean=0 fault-or-invalid=1 (target >=60 clean, cap 60 pairs)
STOP: a fix-cell arm faulted or was invalid — acceptance requires zero faults
```

**The fix arm is the one that faulted.** Its `detector.json`:

```json
{ "requests": 9, "ok": 8, "utf8_failures": 0, "parse_failures": 0, "http_failures": 1,
  "transport_failures": 0, "identical_distinct": 1, "first_failure": "HTTP 500", "truncated": true }
```

and its `server.log` carries the Face-B signature verbatim:

```
0.53.275.611 W srv operator(): got exception: {"error":{"code":500,"message":"The model produced output
             that does not match the expected Content-only …"}}
```

with invalid-UTF-8 text immediately before it. `HRX returned NaN` appears **0** times, so this is **Face B**,
not Face A. `run.log`: `gtt_start 98GiB` — **in gate**.

## Why this counts as a fault and not as an invalid arm

The pre-registration says an arm that dies for a **harness** reason (transport failure, timeout, truncation)
is invalid and never a pass. This arm is `truncated: true` — but the truncation is a **consequence of the
defect**: the server returned HTTP 500 on the model's own unparseable output and the detector stopped there.
A harness death is one where the harness fails independently of the model; here the model failed and took
the arm with it. Recording it as invalid would be exactly the "vacuous pass" the pre-registration's guard
exists to prevent, from the other direction.

## The verdict

**The acceptance criterion is NOT MET, and the mechanism is falsified as a fix.** The prediction was
universal — *the faults stop* — and a single in-regime fault refutes it, the same way the deferred-writeback
candidate was refuted by one counter-example. The campaign did not need its 60 arms to reach that
conclusion; the pre-registered stopping rule ended it at the first one by design.

Honest bounds, so the negative is not overstated:

* **One fault does not measure a rate.** The control arm was clean and this fix arm faulted, which at the
  ambient ~5 % is entirely unremarkable as a *rate* comparison (1 fault in 2 arms). What it does settle is
  the acceptance question and the mechanism question, because both are about a universal claim, not a rate.
* **It does not show the fix is worthless.** Waiting for the copy is still the correct thing to do for a
  backend that implements the async read — `llama_context::decode`'s own comment says so. What it shows is
  that this wait is **not sufficient** to stop the fault, so the unwaited read is not the mechanism.
* **It does not revive any earlier candidate.** Coherence stays closed (§13), the deferred writeback stays
  falsified, and the read-path families stay closed as recorded.

## What remains standing after this

The mechanism hunt returns to the candidates in `315-what-the-synchronise-drains.md`, plus the two
measured-but-unexplained facts this session produced:

1. `result_norm` changes between two consecutive identical plain reads in a faulting arm
   (`memcpy_stable=0`), and that is **not** the deferred writeback and **not** coherence (cells A/B:
   `0x52`/`0x56`, 13.7 %/14.8 %).
2. §51's rank-sort precondition — NaN at the argsort's input — is still unmeasured.

**The fix commit and the pin move stand as landed work** (`47a8fe929`, pin `e44c9d01a → 47a8fe929`): the
change is correct for the reason its comment gives, and it is forward-only. It is simply not the fix. The
objective's criterion 2 asks for a forward-only change to have been landed, and that is true; criterion 3's
campaign is what fails, and it fails with a measured fault rather than an unrun arm.
