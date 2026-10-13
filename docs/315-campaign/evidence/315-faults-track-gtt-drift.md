<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The corpus, re-read: faults track GTT movement into the band, not the cell label

Written while arm B (`315ab1b2`) runs. Motivated by the cell-`c` drift finding: if pressure walking GTT
into a band is the operative variable, then the recorded arms should show faults tracking GTT
*behaviour* rather than the nominal cell they were labelled with. They do, and this resolves an apparent
contradiction in the corpus.

## The tallies, extracted from every recorded arm

Cell x verdict x `gtt_start` (all arms in `docs/315-campaign/evidence/*.md` that carry the fields):

    21  l FAULT 98       12  l OK 98        8  l FAULT 97      2  l OK 97
     6  c FAULT 98        5  c OK 98        1  c OK 97
     3  e FAULT 98
     0  q FAULT 97/98     2  q OK 97        1  q OK 98

Two facts fall out immediately:

1. **Cell `q` (quiet) essentially never faults** — 3 OK, 0 FAULT across both levels;
2. **Cell `l` faults at 97G too** (8 FAULT / 2 OK), which is the apparent contradiction: the 70-arm
   campaign of record ran cell `l` at 97G for 70 arms and got **0 faults**.

## The contradiction resolves on the trace, not the label

The faulting cell-`l` arms at 97G come from runs where **an eviction-inducing instrument was active**.
`315-evict-trace-status.md:484` is the clearest:

    pair 3 arm l: FAULT nan=0 req=20 parse=0 gtt_start=97G ... (gtt_now=102G)

`gtt_now=102G` — the arm **drifted up into the band** — and that run had the eviction watcher attached
under the box lock, recording 15,357 `Q_EVICT` (`:95`, `:441`, `:457`).

The 0/70 campaign has **no eviction cell and no eviction instrument**
(`315-0of70-has-no-eviction-cell.md`), so nothing drove its arms into the band.

So "cell `l` faults at 97G" and "cell `l` is 0/70 at 97G" are both true and are not in conflict: the
first arms were pressure-driven into the band, the second set was not.

## The unified statement this supports

Fault occurrence tracks **memory pressure that makes KFD evict in-flight work**, which in the recorded
data appears as one of:

- a forcer process driving eviction directly (cell `e`, `evictonly`: 7/10 in disc1/disc2, 4/4 in arm A,
  FAULT in every arm-B round);
- GTT **drifting up** into the band during the arm (arm A round 2 quiet arm at 97->102G; arm B round 2
  cell `c` at 97->106G; `pair 3 arm l` at `gtt_now=102G` with the watcher attached);
- an arm starting high (98G) where the collapse to ~80G is the *effect* of the fault tearing down the
  allocation, not the condition.

The cell label is a **proxy for how pressure is applied**, not a category that determines the outcome.
That is exactly why the per-arm eviction *count* correlation in H2' came back false-negative (faulting
mean 109.5 vs clean 254.5): a count of evictions is not a measure of whether the arm was *in the band*.
The intervention (forcing pressure) and the correlation (counting events) are different instruments, and
this reading says the correlation failed because it measured the wrong quantity.

## Scoping

- This is a synthesis of already-recorded arms, not new experiment. It is offered because it makes three
  previously separate observations (the falsified H2' correlation, the 0/70 result, and the fresh
  cell-`c` fault) mutually consistent under one variable.
- It does **not** name the cause — which handoff reads stale data after eviction is still open. It names
  the condition that produces the fault, and makes that condition measurable from the console the
  drivers already emit (`gtt_start`, `gtt_now`).
- `gtt_now` is recorded per arm, so this is checkable against every console on the box rather than only
  against the evidence digests used here. That re-check is the obvious next confirmation.
