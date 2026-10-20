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
# Pre-registration: the alignment question ON GLM, with a working metric (`315galign1`)

Written before the run. Follows the peer's `315align2` design but fixes its three defects.

## Why a rerun is warranted
The peer's run tested `GGML_HRX_FA_PARTIAL_ALIGN` 256 vs 4096 under the real trigger, which is the right
idea, but three defects each independently prevent it from answering the question about issue #315:

 1. **wrong model** - it ran `Qwen3-0.6B-Q4_K_M` (`315align2.sh:10`); #315 is about
    `GLM-4.7-Flash-Q4_K_M`, and every recorded arm, face and rate in the issue is GLM;
 2. **dead fault metric** - `distinct_content` is 0 on every arm because the parser expects SSE `data:`
    lines while `/completion` returns one newline-free JSON object; and
 3. **`align=256` is a DECLINE, not a stress** - `dispatch-flash-attention.cpp:381-386` refuses the
    decode-split kernel for any non-4096 value and falls back.

Defect 3 cannot be fixed without a code change, so this run tests what the knob CAN do: **the decode-split
kernel bypassed (256) vs present (4096)**, on GLM, under the eviction trigger.

## Hypothesis
**H-align.** If the fault is carried by the decode-split FA kernel's dispatch, then bypassing it (align=256)
should change the fault rate. Since the frozen rig faults ~85% in cell `e`, a bypassed kernel that still
faults at ~85% would show the fault does NOT need that kernel.

## Design
    model  : GLM-4.7-Flash-Q4_K_M        (the issue's model - defect 1 fixed)
    driver : the frozen three-cell rig, 2 rounds, tag 315galign1
    arms   : cell e (evictonly + compaction) at align 4096 and align 256; cell q at both, as controls
    metric : the console's own nan=/parse=/transport= columns PLUS the detector (defect 2 fixed by using
             the frozen rig's classification rather than a bespoke parser)

## Branches, fixed before the run
    align256-e faults at ~85%, like align4096-e   -> the decode-split kernel is NOT required; H-align dead
    align256-e clean while align4096-e faults     -> the kernel IS required; alignment becomes the lead
    both clean in cell e                          -> UNINFORMATIVE (induction failed); not evidence either way

## Bounds
 - `align=256` bypasses rather than stresses, so a null result speaks to "which kernel", not to
   "which alignment". Stating it plainly: this cannot test the layout-sensitivity question at all.
 - one model, one prompt shape, 2 rounds: enough to see 85% vs 0% but not to resolve small differences.
