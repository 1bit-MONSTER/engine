<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

engine#315 evidence: pre-registration for the eviction-intervention replication, written and
committed BEFORE the run so the reading cannot be chosen post hoc.
-->

# Pre-registration: replication of the eviction-intervention contrast (`315disc2`)

Written 2026-10-13 ~09:15, committed before the run.

## Why this run exists

`315disc1` (2026-10-11) is the only experiment in this issue's history that **forces** the candidate
mechanism rather than observing it. Its `315-disc.sh` design alternates three cells inside one box
composition:

| cell | side load |
|---|---|
| `q` | quiet — no side load (control) |
| `e` | `~/wt/evictonly 500` — a process that launches **no kernels** and only forces KFD evict/restore via `hipHostRegister` + `mprotect` churn |
| `c` | CPU pressure only, pinned to CCD1 (no GPU) |

Its recorded result:

    cell q (quiet)        5 arms   0 FAULT
    cell e (EVICTION)     4 arms   2 FAULT   (nan=1 transport=1, req=6 and req=4)
    cell c (CPU pressure) 5 arms   0 FAULT
    Fisher exact, e (2/4) vs q+c pooled (0/10): p = 0.066

Both faults are **Face A** (`nan=1`), both abort early (`req=4`, `req=6` of 66), both at
`gtt_start=98G`. The two control cells are clean across 10 arms.

That is the strongest causal contrast in the issue, and it is the rate-raising design the
26 %-vs-7 % rate correction identified as missing. It has never been replicated.

## Hypothesis (falsifiable)

**H-e:** forcing KFD eviction/restore from a separate process that runs no kernels increases the
fault rate, relative to both a quiet box and CPU-only pressure, at the same composition.

## Design

    driver : ~/wt/315-disc.sh <ROUNDS> 315disc2      (the SAME frozen design as disc1)
    cells  : q (quiet) / e (evictonly 500) / c (CPU pressure)  — rotating order per round
    build  : the arm binary the arms have always used (315instr3-build, 10613 / e44c9d01a)
    gate   : gtt_start >= 95 GiB (the frozen regime gate); OOR arms reported, NOT counted
    rounds : 6  -> ~12 e + ~6 q + ~6 c arms, ~43 min at the measured 2.4 min/arm

The rotating cell order is the design's own protection against session drift aligning with one cell,
and it is why this contrast is interpretable where a start-to-finish A/B would not be.

## Prediction

If H-e holds: cell `e` accumulates faults at a rate clearly above cells `q` and `c`, whose arms stay
clean. Combining with disc1, ~6/10 eviction arms faulting against ~0/16 control arms would put the
contrast beyond the p = 0.066 seen at disc1's size.

## Refutation and the branches, fixed in advance

    e faults clear of q and c          -> H-e REPLICATED. Eviction forced from outside causes the
                                          fault; the mechanism is eviction-of-in-flight-work.
    e ~= q ~= c (all clean, or all alike) -> H-e NOT REPLICATED. disc1's contrast was a small-sample
                                          artefact or a session-specific condition, and the eviction
                                          intervention is not a mechanism.
    eviction cell clean but q or c faults -> H-e CONTRADICTED in direction; report as such.
    all cells clean                       -> UNINFORMATIVE for the contrast. Must NOT be read as
                                          evidence that eviction is irrelevant: at the measured ~5-7 %
                                          ambient rate, ~16 arms would be expected to be clean most of
                                          the time, which is exactly the under-powering trap the
                                          26 %-vs-7 % correction documented.

**The uninformative branch is pre-declared as uninformative**, because the ambient rate is low enough
that a clean replication proves nothing on its own; only a POSITIVE (eviction faults, controls do not)
carries information at this size.

## What this run does NOT do

 - It does not land a fix (criterion 2) - it tests a mechanism (criterion 1).
 - It is not the >=60-arm campaign (criterion 3, already MET by `315cwsr70`).
 - It cannot by itself close criterion 1: even a clean replication names the TRIGGER (forced
   eviction) rather than the CAUSE (which handoff reads stale data). The cause would still need the
   static leg, which is already recorded as inconclusive for the #140 class in GLM's path.

## Cost and budget

~43 min at 2.4 min/arm, against ~114 min remaining of the objective's 12 h box-time ceiling. It fits.
