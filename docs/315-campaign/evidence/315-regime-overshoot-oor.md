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

# Regime overshoot: composition reached 115-116G, above the declared gate

A second agent's run (`315galign3`, `GGML_HRX_FA_PARTIAL_ALIGN=256`, 2 rounds) produced **two arms that are
`OOR`** — out of the declared regime — because the composition overshot:

    round 1 arm q: OOR  gtt_start=115G  gtt_now=117G  side=0/0
    round 1 arm e: OOR  gtt_start=116G  gtt_now=115G  side=1/1

against the rig's gate of `[95, 110]` GiB (`GLO=95`, `GHI=110` defaults in `315-disc.sh`). Both arms also
produced short request counts (`req=9` and `req=1`), so they are invalid on two counts.

## Why this happens

The composition is a **projection, not a measurement**. `315-regime.sh:107` reports:

    regime: composition neighbours=N ctx=... gtt_before_arm=X G (LO=80 HI=92, projects to >=95 after the
    detector loads)

with `REGIME_LO=80`/`REGIME_HI=92` controlling when to stop adding neighbours, and the gate itself applied
at `GLO=95`/`GHI=110`. The gap between the composition target (projecting to ">=95") and the gate's upper
bound (110) is what a warm detector can close: if the arm's own load adds more GTT than projected, the arm
lands above 110 and is scored `OOR`.

## Why it matters for this goal's criteria

Criterion 3 asks for **>=60 arms, every arm inside the declared regime gate**. An overshoot of this size shows
that "in-regime" is a property of how the composition happened to land, not of the declared band — the same
class of issue as the retracted `gtt_start` census, from the other direction:

- the earlier problem: arms *below* the faulting band were counted as in-regime;
- this problem: arms *above* the gate are produced and then discarded, so the reachable in-regime window is
  narrower than the declared `[95, 110]`.

Both argue the same thing: **a gate defined as a band on `gtt_start` is the wrong instrument**. What the
campaign should declare is the **cell** (the forced-eviction intervention) and the *rate* it is testing, with
`gtt_start` reported as a covariate — which is what `315-quiet-vs-forcer-by-level.md` and the corrected
KNOWN-ISSUES entry now say.

## Also worth noting

Both `OOR` arms here still showed the fault signature in one case (`arm e: nan=1`), i.e. the fault can occur
above the gate too. That is consistent with the fault tracking the **forcer** rather than the band, and it is
another reason the gate should not be the organising concept.

This file records an observation from another agent's run, not a result of its own: `OOR` arms are invalid by
the rig's own classifier and nothing here is scored.
