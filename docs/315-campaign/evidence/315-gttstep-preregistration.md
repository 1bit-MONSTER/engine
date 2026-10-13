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
