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

engine#315 pre-registration. Written BEFORE the run so the reading cannot be chosen post hoc.
-->

# Pre-registration: does `gtt_start` alone separate the fault populations? (`315gttstep`)

Motivated by `315-gtt-start-is-the-discriminator.md`, whose census found 5.1% faults at
`gtt_start=97G` against 30.6% at `gtt_start=98G` — holding *within* every cell (`l` 6.0%->24.8%,
`c` 2.2%->29.5%). That census is heterogeneous runs; this is the controlled confirmation.

## Hypothesis

**H-start.** Fault probability is governed by the arm's `gtt_start`, not by its cell label. Specifically
there is a boundary between 97 and 98 GiB below which the fault is rare (~5%) and above which it is
common (~30%).

## Design — no forcer, no watcher, no build change

One binary (the engine pin's, unmodified), one cell, three start levels. Cell `q` (quiet: no side load)
is used precisely because it is *not* the forcer — the question is whether pressure level alone moves
the rate. Start level is set by the neighbour composition (the same `regime_neighbours` machinery the
driver already uses), targeting `gtt_start` of **97, 98, 99 GiB**.

    driver : 315-disc.sh style, cell fixed to q, EXTRA_ENV = the standard detector env only
    levels : 97 / 98 / 99 GiB, interleaved across rounds so drift cannot line up with a level
    rounds : enough to reach >=20 scored arms per level if the budget allows; 12 per level minimum

Note this deliberately does **not** use `evictonly`. If H-start holds, no forcer is needed to reach the
high band, and that is the point — the earlier work's forcer may have been a fast way to reach a band
that ordinary occupancy can also enter.

## Reading, fixed BEFORE the run

    rate(97) << rate(98) <= rate(99), each level's CI excluding the others' point estimate
        -> H-start SUPPORTED. The regime gate must be redefined to the high band, and criterion 3's
           ">=60 arms in-regime, zero faults" is only meaningful at >=98G.
    rate(97) ~ rate(98) (overlapping CIs, same order)
        -> H-start REFUTED at this design. The census split was a composition artifact after all, and the
           discriminator reverts to the cell/instrument.
    rate decreasing with start, or non-monotone
        -> report as observed; do not force a story.

## Explicit stopping rule

Stop when each level has >=12 scored arms or the box-time ceiling binds, whichever comes first. Arms that
die for harness reasons (INVALID/OOR) are recorded as invalid, never as passes. No arm pruning, no
post-hoc level merging.

## What this does and does not settle

- **Settles**: whether the level alone is the discriminator, under control.
- **Does not settle**: the cause site. Even a clean H-start confirmation leaves "which handoff reads
  stale data after eviction" open — the read path was eliminated by the A/B
  (`docs/315-campaign/evidence/315-ab1-arm-B-interim.md`, `315-ab1-fix-path-is-coherent.md`).
- **Does not touch criterion 2.** No fix is proposed or tested here.

## Budget and coordination

~2-3 h of box time, taken under the box lock. This must not run concurrently with any other 315 driver
(see `~/wt/315-COORDINATION.md`): one tag, one driver. The other Pi session (`01a12f94`) is currently
idle; this run must not start until arm B (`315ab1b2`) has finished and released the lock.

---

# SUPERSEDED (2026-10-13) — do not run this as written; its hypothesis was retracted

This pre-registration was written on the **pooled 97G-vs-98G census** (5.1% vs 30.6%), which was
subsequently **retracted** after a within-console control broke it (`315-gtt-start-is-the-discriminator.md`,
correction section: `315ctl1` with no forcer showed 9.1% at 97G against 6.2% at 98G — no level effect).

Later and stronger: the **cell-separated** tally across every recorded console shows the quiet cell is
**0% at 98G** against 5% at 97G, so `gtt_start` does not govern the fault at all
(`315-quiet-vs-forcer-by-level.md`). Against that, this file's H-start — "fault probability is governed by
the arm's `gtt_start`" — is **falsified by the corpus**, and its three-level design would spend GPU time
re-testing a hypothesis already refuted.

## What replaces it

The corrected gate is not a band and not a start level; it is the **intervention cell**. The fault lives in
cell `e` (~95% under forced KFD eviction) against ~5% ambient, a 19x contrast on the same workload at the
same level. Any future campaign claiming the defect is gone must therefore:

1. run the **forced-eviction cell**, and report the ambient (quiet-cell) rate alongside it, so a clean result
   cannot be produced by simply not applying the trigger;
2. state its arm count against the **actual** rate it is testing — at ~95% a handful of arms discriminates,
   whereas the old ">=60 arms" bar was sized for a 5% population and is satisfiable by quiet arms.

## Status of this file

Kept rather than deleted so the retraction is traceable: the design was reasonable given what was believed
at the time, and the reason it no longer applies is recorded above and in the two files it cites. **Nothing
in this file should be run or cited as a live pre-registration.**
