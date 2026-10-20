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

# Does the 0/70 campaign contradict the eviction faults? No — it has no eviction cell

Prompted by the cell-`c` drift finding (`315-ab1-cell-c-fault-drift.md`): if memory pressure walking GTT
into the eviction band is what faults, then a campaign with **no forcer** may never reach the band and
its 0-fault result would say nothing about the defect. Checked the campaign of record's own cell labels
rather than assuming.

## What the 70-arm campaign actually ran

From `7d74d5c` (315cwsr70, campaign of record):

    fix-cell (arm l)   OK=70  FAULT=0  OOR=0  INVALID=0   (all 70 at gtt_start=97G)
    control-cell (c)   OK=24
    scorer total       94 = 70 fix + 24 control

The cells are **`l`** (the "fix" cell of the cwsr design) and **`c`** (control). There is **no cell
`e`** — the eviction forcer is not part of this campaign. The run's own closing line says so, and the
commit records it:

> "both cells ran the same revision e44c9d01a" … "A clean trace is NOT evidence about the fault's
> mechanism."

## Why that matters, in light of the drift measurement

The eviction-cell faults (disc1/disc2: 7/10; arm A: 4/4; arm B: FAULT in every cell-`e` round so far)
all come from arms where something drives GTT **into the band**. The drift trace shows how a cell gets
there and how a cell can stay out of it.

A campaign of unforced in-regime arms at `gtt_start 97G` is therefore measuring the **ambient** rate,
not the faulting regime. 0/70 at ambient pressure is entirely consistent with 7/10 under forced
eviction and with the cell-`c` fault — they are different regimes, not contradictory results.

## What this does and does not change

- **Criterion 3 as written is not met by 315cwsr70 in the sense that matters.** The objective asks for
  "≥60 arms … zero faults across both faces" as evidence the defect is gone. A 70-arm run with no
  eviction cell shows the defect did not fire *at ambient pressure on the same revision*; it does not
  show the fault is absent, and the same commit already concluded "no mechanism is named, no fix is
  landed."
- **It is not a false result and should not be discarded.** It is a valid rate measurement for the
  ambient setting, correctly labelled, and the earlier session explicitly refused to over-read it.
- **It sharpens what criterion 3 needs**: arms that reach the band. A campaign on a *fixed* build must
  include eviction (`e`) arms — or demonstrate the fixed build no longer faults when pressure is driven
  into the band — otherwise a 60-clean-arm bar can be satisfied by never testing the condition.

## Consequence for this A/B

It means the arm-B result cannot be answered by more ambient arms. The relevant statistic is
**cell `e` on the fixed build**, which is exactly what arm B measures — and it has faulted in every
round so far. That is the evidence criterion 2 turns on.
