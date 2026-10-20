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

# The discriminator is `gtt_start`: 5.1% at 97G vs 30.6% at 98G (6x)

Extracted from **every recorded arm on the box**, not the evidence digests: all `315*-console.log` under
`~/wt` were pulled (1,675 lines) and every line matching `arm [qelc]: (OK|FAULT|OOR|INVALID)` parsed for
its cell, `gtt_start` and `gtt_now`. This is a census of the corpus rather than a sample.

## The result

    gtt_start=97G:  OK=150  FAULT=8    rate = 5.1%
    gtt_start=98G:  OK=145  FAULT=64   rate = 30.6%

A **six-fold** difference between adjacent 1 GiB bands, on 313 arms at those two values.

Full tally at the in-regime values, by cell:

    cell l  @98  OK=100 FAULT=33        cell l  @97  OK=94  FAULT=6
    cell c  @98  OK=43  FAULT=18        cell c  @97  OK=45  FAULT=1
    cell e  @98  FAULT=13 OK=1          cell q  @97  OK=11  FAULT=1
    cell q  @98  OK=1  FAULT=0

## What it explains

1. **The 0/70 campaign of record.** It ran at `gtt_start=97G` — the 5.1% band. 70 arms at 5.1% give ~3.6
   expected faults; observing 0 is unsurprising (P(0) ~ 3%). So the 0/70 result is a genuine
   *low-band* measurement, and it was never evidence of absence — consistent with
   `315-0of70-has-no-eviction-cell.md`, reached independently.
2. **Cell `e`'s near-total fault rate** (13/14 at 98G). The forcer pushes arms into/through the high
   band immediately, which is why it faults so reliably.
3. **The cell-`c` and quiet-arm faults.** These occur at 98G starts (or by drifting there: `gtt_now=102G`,
   `106G`), i.e. in the high band, regardless of their nominal cell.
4. **The falsified H2' eviction-*count* correlation.** A count of evictions is not the band indicator;
   `gtt_start`/`gtt_now` is. The correlation was measuring the wrong quantity
   (`315-faults-track-gtt-drift.md`).

## Why this is the mechanism-level finding

The declared regime gate is `[95, 110]` GiB. The census shows that window contains **at least two
populations with a 6x difference in fault rate**, and that the boundary sits between 97 and 98 GiB. A
campaign that treats `[95,110]` as one regime is therefore mixing a ~5% population with a ~31% one, and
its arm count says nothing unless the split is preserved.

This also fits the physical story the mechanism statement already had: the higher the GTT occupancy the
tighter the memory pressure and the more likely KFD is to evict in-flight work. The band boundary is
where that becomes likely rather than rare.

## What this does NOT establish

- **It does not name the cause site.** It sharpens the *condition*. Which handoff reads stale data after
  an eviction remains open, and the read path is now eliminated by the A/B.
- **It is not a controlled experiment.** These arms come from many runs with different instruments
  (forcers, watchers, CPU workers, quiet). The 97/98 split is a strong signal *conditional on the
  corpus*, and the clean test is a single pre-registered run holding the cell fixed while stepping the
  start value across 97/98/99.
- **`gtt_now` is partly an effect, not a condition.** Arms that fault show `gtt_now≈80G` because the
  fault tears down the allocation. `gtt_start` is the pre-arm value and is the one to use as the
  covariate; `gtt_now` should be read only for drift.

## Consequence for this goal's criteria

**Criterion 3 as specified cannot be satisfied by arms in the low band.** ">=60 arms, every arm inside
the declared regime gate, zero faults" is satisfiable by 60 arms at 97G while the defect is fully
present at 98G. Any campaign claiming the defect is gone must show zero faults **in the high band**
(>=98G), or the gate must be redefined as the population that actually faults.

That is recorded as a finding about the criterion, not a claim that the criterion is met.

---

# Confound check (same turn): the effect is NOT driven by cell `e`, and holds INSIDE each cell

The obvious confound is that cell `e` (forced eviction) both sits at 98G and faults reliably, making the
pooled 98G rate circular. Re-ran the census with cell `e` removed and, separately, within each cell.

## Excluding cell `e` entirely

    excludng e:  gtt_start=97G  OK=150 FAULT=8   rate = 5.1%   (n=158)
                 gtt_start=98G  OK=144 FAULT=51  rate = 26.2%  (n=195)

Still a 5x difference. The effect does not depend on the forcer cell.

## Within each cell, independently

    cell l @97G:  OK=94  FAULT=6   rate =  6.0%      cell l @98G:  OK=100 FAULT=33  rate = 24.8%
    cell c @97G:  OK=45  FAULT=1   rate =  2.2%      cell c @98G:  OK=43  FAULT=18  rate = 29.5%
    cell q @97G:  OK=11  FAULT=1   rate =  8.3%      cell q @98G:  OK=1   FAULT=0   rate =  0.0%

**Every cell with adequate n shows the same direction and a similar magnitude** (4x for `l`, 13x for
`c`). Cell `q` at 98G has n=1 and carries no weight; cell `q` at 97G is 8.3% on n=12, consistent with
the others at the low band but with a wide interval.

This matters because it rules out the two readings that would have made the pooled number an artifact:

- not "98G is really the eviction cell in disguise" — excluding `e` keeps the effect;
- not "98G is a different mix of cells" — the effect is present *within* each cell, so it is the start
  value doing the work, not the cell composition.

## What is still owed

This is a census of heterogeneous runs, so the honest statement is: **the 97/98 boundary separates two
fault-rate populations in every cell measured, and no cell's own label explains the split.** The clean
confirmation is a single pre-registered run with the cell fixed and the start value stepped (97/98/99),
which would also give the boundary's sharpness. That run is the obvious next experiment and is cheap —
it needs no forcer, no watcher and no build change, only a controlled set of in-regime arms.

---

# IMPORTANT CORRECTION (same day) — the pooled 97/98 split does NOT survive within-console. Confounded.

The census above pools heterogeneous runs, so the right control is to compare 97G against 98G **inside a
single console** (one session, one instrumentation, one condition). Doing that breaks the finding:

    315ctl1-console.log    97G: 10 OK / 1 FAULT =  9.1%     98G: 15 OK / 1 FAULT =  6.2%
    315disc2-console.log   97G: 12 OK / 0 FAULT =  0.0%     98G:  1 OK / 5 FAULT = 83.3%

`315ctl1` — which ran **without the eviction forcer** — shows **no level effect at all** (9.1% vs 6.2%,
slightly inverted, on n=11 and n=16). The whole pooled 6x effect is carried by `315disc2`, which ran
**with** the forcer.

So the pooled table is confounded exactly as it looked dangerous: the 98G population is dominated by
**forcer runs**, and the forcer — not the level — is what produced those faults. The within-cell
breakdown I offered as the confound check does not rescue it either, because the cells are not
independent of the instrument: cell `e` is *defined* by having the forcer, and the `c`-cell 2.2%->29.5%
split comes from runs where the forcer was also present in the session.

## What is actually supported

- The **forcer** causes faults (7/10 vs 0/22 pooled; arm A 4/4; arm B faulting with the fix on). That
  stands and is a controlled intervention.
- A **single arm** drifted into the band and faulted while doing so (arm A quiet r2, 97->102G), and arm
  B's cell-`c` fault drifted 97->106G. These are individual observations, not rates.
- `gtt_start=98G` is **associated** with faults in the pooled corpus, but association here is
  confounded by instrument and does **not** establish the level as the cause.

## What is NOT supported, and must not be cited

- "Fault probability is governed by `gtt_start`" as a rate statement.
- "5.1% at 97G vs 30.6% at 98G" as a property of the level.
- "The 0/70 campaign is explained by running in the 5% band" — with the level effect unestablished, the
  0/70 run may equally be explained by having had no forcer at all, which its own file already said.

## What this changes

The pre-registered `315gttstep` experiment (`315-gttstep-preregistration.md`) becomes **more** necessary,
not less: it is now the only way to test the level question, because the corpus cannot separate level
from instrument. Its design is already correct for this (cell fixed to `q`, no forcer), and its branch
"rates equal -> refuted at this design" now looks like the likelier outcome on `ctl1`'s evidence.

The KNOWN-ISSUES update that cites the census as the governing condition must be corrected to carry this
caveat, or it will record a confounded rate as a mechanism.
