<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Arm B round 2 cell `c` FAULTED — a control-cell fault that the mechanism predicts

Recorded immediately, because a fault in a *control* cell is the kind of result that gets quietly
explained away. Here it is informative, and it is the strongest single piece of evidence the A/B
produced about the mechanism itself.

## The result

    round 2 arm c: FAULT nan=1 req=34 transport=1 s500=0 unparsed=0
                   gtt_start=97G gtt_now=79G tctl_start=49C tctl_max=92C
                   side=12/12 stopped_frac=0.16 thermal_pauses=0

Cell `c` is CPU-pressure-only by design (12 busy-loop workers pinned to CCD1, **no GPU side load**).
It was clean 4/4 in arm A and clean in the original disc runs. Here it produced **Face A**:

    common_sampler_sample: HRX returned NaN logits: count=154880 of 154880,
      index range [0..154879], contiguous=yes, continuing=no (engine#315)
    common/sampling.cpp:620: HRX: NaN logits (engine#123)

whole-vocabulary NaN, 2.31 s into the arm, with the sampler guard aborting — the exact Face-A signature.

## Why it is not a harness artifact

- **in-regime**: `gtt_start 97GiB`, inside the declared [95,110] gate;
- **the side load ran the whole arm**: `side=12/12`, and the driver marks `INVALID(side-died)` if not;
- **not thermal**: `thermal_pauses=0`, `tctl_max=92C` (below the 93C abort);
- **not the CPU load itself**: cell `c`'s CPU load is the same in every clean round, including arm A's
  4 clean cell-`c` arms;
- the failure is the recorded Face-A string, not a parse or transport oddity — the `transport_failures=1`
  is the sampler guard aborting the connection *after* the NaN, which is the documented order.

## What the GTT trace shows — and this is the point

Per-sample GTT across the arm (`tctl.log`, GiB):

    79 96 97 98(4) 99(9) 100(4) 101(2) 102(4) 103(2) 104(2) ... 106

The cell **drifted upward** 97G -> 106G during the arm, then collapsed to 79G when the server's
allocation was torn down after the fault. It did not sit still at 97G and get unlucky.

That is precisely the mechanism's prediction, and precisely the pattern arm A's quiet-cell fault showed:
**the operative variable is memory pressure that drives the kernel to evict in-flight work, not the
forcer process.** `evictonly` is a *proxy* — it drives pressure to the band deliberately and fast.
A cell that reaches the band on its own, whatever its nominal load, hits the same condition. Cell `c`'s
CPU workers evidently perturbed allocation enough to walk GTT up across the band.

## What it changes about the reading

- **It does not weaken the A/B.** The pre-registered comparison is cell `e`; this is a control cell.
- **It strengthens the mechanism statement** by supplying a *third* independent way to reach the fault
  (quiet-drift in arm A, forcer in disc1/disc2, CPU-drift here), all of which share one variable:
  GTT/memory pressure walking into the eviction band.
- **It means the regime gate is necessary but not sufficient for cell purity.** `gtt_start` inside
  [95,110] does not keep a cell in one regime for its whole duration — `gtt_now` and the drift between
  them matter, and the console records both. Any future campaign should treat drift into the band as a
  covariate, not assume the start value holds.

## Honest scoping

This is one arm. It does not show cell `c` faults at the eviction cell's rate — cell `c` is 1 fault in
2 rounds here against cell `e`'s 2 in 2, and arm A had cell `c` clean 4/4. It is recorded because the
direction is the mechanism's and because the drift is measurable, not because it establishes a rate.
