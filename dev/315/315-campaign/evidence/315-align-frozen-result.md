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

# The FA alignment test on the FROZEN rig: A and B are IDENTICAL — alignment is not required

The pre-registered test of the last open family, run on the driver that actually reproduces the trigger
(the earlier attempt used a bespoke composition that stayed mostly clean and was therefore uninformative).

## Result

Same binary, same frozen three-cell rig, only `GGML_HRX_FA_PARTIAL_ALIGN` differing:

                 cell q            cell e            cell c
    arm A     2 OK / 1 FAULT    0 OK / 3 FAULT    3 OK / 0 FAULT      align 4096 (default)
    arm B     2 OK / 1 FAULT    0 OK / 3 FAULT    3 OK / 0 FAULT      align 256 (#140 oracle layout)

**Byte-for-byte the same tally.** Cell `e` — the forced-eviction test cell — faulted **3 of 3 arms in both
arms**, at `gtt_start` 98 GiB, zero OOR, zero INVALID, with `side=1/1` on every cell-`e` arm. Both arms also
show the same single cell-`q` fault (the recorded drift-into-band effect, `side=0/0`).

## Validity, verified rather than assumed

- **1096 vs 256 genuinely applied**: each arm's session header records the value
  (`GGML_HRX_FA_PARTIAL_ALIGN=4096` for A, `=256` for B), and both composed to GTT 98-100 GiB before their
  first arm.
- **Both arms in-regime** at `gtt_start=98G` on every scored arm; no OOR or INVALID in either.
- **The forcer was live** on every cell-`e` arm (`side=1/1`).

## The pre-registered reading

    A ~= B (both at the rig's ~95%, overlapping)
        -> the alignment does not carry the fault, and the FA family closes.

Met exactly. And per the pre-registration's own power note, this is a **one-sided** conclusion:

> **the FA layout/dispatch class is not REQUIRED** for the fault — `align=256`, the layout the #140 oracle
> used to *expose* the divergence, faults at the same 3/3 as the default.

It is **not** a measured null: 3 cell-`e` arms per alignment cannot bound a *rate* difference (distinguishing
95% from 60% needs ~20 per condition). So this lands at the **same strength as the coherence and `GRAPH_SYNC`
rows**: the candidate is excluded as sufficient, not shown to be irrelevant.

## What it resolves about the earlier confusion

The FA family was left OPEN because (a) the first attempt used a composition that did not reproduce the
trigger and (b) I had briefly misread its arms as all-faulting via a `md5sum`-over-streaming parse error. This
run removes both problems: the frozen driver reproduces at 3/3, and the verdict comes from the rig's own
classifier rather than a hand-rolled one.

## Net effect on the closed set

    logits read path                REFUTED       its own pre-registered A/B
    host side                       CLOSED        none on the decode path
    runtime AQL submission policy   CLOSED        the runtime does not infer hazards
    graph reorder                   REFUTED       cell e 4/4 in BOTH arms
    deferred host writeback         CLOSED        binding host_data == nullptr
    memory-pool quarantine/reuse    CLOSED        no-op trim, fresh-only acquire, immediate release
    FA alignment/dispatch (#140)    NOT REQUIRED  this run: 3/3 both alignments, frozen rig

**The cause remains unlocated**, now with the FA family excluded as sufficient on the rig that reproduces the
fault — which is a stronger exclusion than the earlier, mis-parsed one.
