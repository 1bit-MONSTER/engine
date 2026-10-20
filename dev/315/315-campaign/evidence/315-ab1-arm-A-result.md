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

# A/B arm A — the control that must fault (315ab1a), 4 complete rounds

Pre-registered at `fc7ffd5` (`dev/315/315-campaign/evidence/315-ab-preregistration.md`), which fixed the
binary, the rig, the four reading branches and the cell-`e` comparison BEFORE the run. This file
records the arm-A result only, written after the arm finished and before arm B's result is known.

## Invocation (re-runnable)

    BIN=/home/bcloud/wt/315coherent-build/bin \
    bash ~/wt/315-disc.sh 4 315ab1a

Patched binary, knob **OFF** (no `GGML_HRX_COHERENT_READ`), i.e. pre-fix behaviour, on the same frozen
three-cell eviction rig used by disc1/disc2. Box lock held on fd 9 (`.cache/lax-decode/box.lock`).
Console: strixhalo `~/wt/315ab1a-console.log`.

Session header: tag=`315ab1a`, rounds=4, gtt_before=26G, mode=performance, bin version 10613,
`extra_env='GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1'` (no knob),
model_sha256 `29837ed2c0fc5f51981adf8ac8083fcf80743c598381f13e9f06cbad0498b174`,
detector_sha256 `9d46ec453fc9640e33e6d6eed857c3e263bfa3df9c1c151a352e1f6f62a66a48`.

## Result — the driver's own summary

    cell q: OK=3 FAULT=1 OOR=0 INVALID=0
    cell e: OK=0 FAULT=4 OOR=0 INVALID=0
    cell c: OK=4 FAULT=0 OOR=0 INVALID=0

Per-arm:

    round 1 arm q: OK     nan=0 req=66   gtt 97G->98G   side=0/0
    round 1 arm e: FAULT  nan=1 req=19   gtt 98G->79G   side=1/1   <- FACE A
    round 1 arm c: OK     nan=0 req=66   gtt 97G->98G   side=12/12
    round 2 arm e: FAULT  nan=0 http=1 s500=1 unparsed=2  side=1/1  <- FACE B
    round 2 arm c: OK     nan=0 req=66                    side=12/12
    round 2 arm q: FAULT  nan=0 http=1 s500=1 unparsed=2  gtt 97G->102G  side=0/0  <- QUIET CELL
    round 3 arm c: OK     nan=0 req=66                    side=12/12
    round 3 arm q: OK     nan=0 req=66   gtt 97G->98G   side=0/0
    round 3 arm e: FAULT  nan=0 http=1 s500=1 unparsed=2  side=1/1  <- FACE B
    round 4 arm q: OK     nan=0 req=66   gtt 97G->98G   side=0/0
    round 4 arm e: FAULT  nan=0 http=1 s500=1 unparsed=2  side=1/1  <- FACE B
    round 4 arm c: OK     nan=0 req=66                    side=12/12

**Cell `e` faulted in every round (4/4), across BOTH faces** (Face A on round 1, Face B on rounds 2–4),
while cell `c` (CPU pressure, no GPU) was clean 4/4. Arm A is therefore a valid control: the rig
reproduces on the patched binary with the knob off.

## The quiet-cell fault, read against the GTT traces

Cell `q` faulted once (round 2, Face B) with `side=0/0` — no side load at all. It is not rig noise:
comparing every quiet arm across the three runs of this rig,

    run       quiet arms   their GTT behaviour             quiet-cell faults
    disc2          6       all 97G->98G (stable)                     0
    ab1a           4       97->98, 97->102, 97->98, 97->98           1  <- the drifting one

the faulting quiet arm is the one whose GTT **drifted +5 GiB** (97G -> 102G); every quiet arm at stable
GTT is clean. That is what the mechanism predicts and what `evictonly` is a *proxy* for: the operative
variable is **memory pressure that makes the kernel evict in-flight work**, not the forcer process
itself. A quiet arm whose occupancy climbs into the band can reach the same condition with no side load.

This sharpens what the A/B tests rather than weakening it. The pre-registered branch of interest is
unchanged: **arm B clean in cell e while arm A faults in cell e.** Arm A's cell `e` is 4/4 FAULT.

## What this does NOT establish

This is the control. It says nothing about whether the fix works; that is arm B (`315ab1b`), same
binary with `GGML_HRX_COHERENT_READ=1`, whose result is recorded separately when it completes.
