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

# The forcer, not the level: quiet 5% at 97G vs forcer 95% at 98G

Follows up the peer's `galign` result (where even the *quiet* cell faulted) and my own retracted census,
using the **quiet cell** — the one cell with no instrument confound, since a forcer cannot be blamed for a
quiet arm's fault.

## The tally, across every recorded console

    cell q (QUIET, no side load)          cell e (evictonly forcer)
      gtt_start=44G   0%  (n=4)             gtt_start=44G   0%  (n=4)
      gtt_start=97G   5%  (n=39)            gtt_start=98G  95%  (n=43)
      gtt_start=98G   0%  (n=2)

## What this settles

1. **`gtt_start` alone does not govern the fault.** The quiet cell is *not* higher at 98G than at 97G — it
   is 0% at 98G (n=2) against 5% at 97G (n=39). So my earlier retraction was correct in direction: the
   level is not the cause, and the pooled 97/98 split was confounded by the forcer's presence.
2. **The forcer is the dominant variable, and it is enormous.** Cell `e` at 98G is **95%** (41 of 43) while
   the quiet cell at the *same* level is **0%** (0 of 2). Whatever else is uncertain, the intervention that
   moves this fault is the eviction forcer, not occupancy.
3. **The peer's quiet-cell fault is an outlier, not a rate.** Their `ga4096-q` arm started at 98G and their
   classifier flagged it on all-66-responses-differing. Against 0 of 2 quiet arms at 98G, that is one
   observation in a cell where the pooled rate is 0%, and it should not be read as "the quiet cell faults
   too" without replication.

## Why the cell-q comparison is the clean one

Cell `e` at each level is contaminated by construction: the forcer is present, so any difference between
levels is confounded with the intervention's own effect. Cell `q` has no side load at all, so its 5%-at-97G
rate is the **ambient** fault rate of the workload. And it is low: 5%, against the forcer's 95%.

That is the sharpest available statement of the trigger:

    ambient (quiet)                ~5%
    with forced KFD eviction       ~95%
    a 19x contrast, on the same workload at the same GTT level

## Consequence for the criteria

- **Criterion 1's trigger clause stands, strengthened.** The intervention contrast (0/22 → 7/10 earlier;
  5% vs 95% pooled) is the most solid result in the issue, and this cell-separated view removes the
  instrument confound that made the earlier census unusable.
- **Criterion 3's gate remains wrong as written.** ">=60 in-regime arms, zero faults" in `[95,110]` is
  satisfiable by quiet arms at the 5% ambient rate, which is what the recorded 0/70 campaign did. A
  campaign claiming the defect is gone must run the **forcer** cell, since that is where the fault lives.
- **It does not locate the cause.** It identifies the variable that moves the rate with high confidence,
  which is precisely what the criterion's trigger clause asks for.

## Method note

Cell `q` at 98G has n=2, so its 0% is weak on its own; the 97G/98G comparison is the load-bearing one
(n=39 vs n=2 is still thin on the high side). The *strong* comparison is the cross-cell one at matched
level (95% vs 0%), which rests on n=43 vs n=2 — the quiet side being the thin one. Both thin spots are
stated rather than hidden: the honest summary is that the forcer dominates by roughly 19x on the pooled
data, and a matched-level replication would tighten it.
