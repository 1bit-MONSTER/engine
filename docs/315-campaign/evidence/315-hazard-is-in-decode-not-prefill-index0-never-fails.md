<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The fault is a per-decode-step hazard of ~1e-3, ~6× higher in the first seven decode steps, and ZERO in the prefill's own output

Follow-up to `315-fault-is-a-per-step-hazard-after-prefill.md`, with two corrections to that document's framing.
No GPU. 2,259 evaluated runs, 60 prompts, 54 arms.

## Corrected hazard table (per decode index)

    index  events  at-risk   hazard
      0      0      2259     0.00000   <- NEVER
      1      3      2233     0.00134
      3      5      2228     0.00224
      5      2      2202     0.00091
      7      1      2149     0.00047
     11      1      1978     0.00051
     14      2      1926     0.00104
     16      1      1925     0.00052
     31      1      1445     0.00069
     33      1      1282     0.00078
     49      1       719     0.00139

**All 18 events: `nan = 0`, and the prefix bit-matches the consensus.**

## The three things this establishes

1. **Index 0 never diverges — 0 of 2,259.** Index 0 is the logits the prefill itself produces. **So the prefill's
   output is always correct**, and every one of the 18 faults happens in a *decode* step. This **excludes** the
   leading hypothesis from the previous document — that the hazard is at the KV handoff between prefill and
   decode. If stale or unhidden prefill KV were the cause, index 0 would be the *most* exposed step and would
   fail most often; it fails never.
2. **The hazard is constant within Poisson noise, but the early excess is real in aggregate.** The individual
   per-index hazards span 0.00047–0.00224 with 1–5 events each, which Poisson noise alone explains. But
   aggregating: **steps 1–7 hold 17,734 of 84,933 at-risk steps (20.9 %) yet carry 11 of the 18 events.** Under
   a constant hazard the expectation is 3.8; observing 11 is p ≈ 0.002. **So the hazard is ~6× higher in the
   first seven decode steps (~6.2e-4) than afterwards (~1.0e-4).**
3. **A first draft of this analysis was wrong and is retracted before it was published.** I normalised index to
   an absolute KV position (`prompt_len + index`) expecting a boundary to appear. It does not: the prompt
   lengths on this record span 25…810, so the at-risk denominator at any absolute position mixes requests of
   different lengths, and the "hazard" computed that way is not a hazard. **Per-index is the only clean space**,
   because every request shares the same index axis. Recorded here rather than quietly dropped.

## What this means for the mechanism

- The fault is **in the decode loop**, not in the prefill and not in prefill→decode handoff.
- It is **per-step probabilistic at ~1e-3**, i.e. a race or an intermittently-wrong read that each decode step
  can lose, not an accumulating corruption and not a fixed-size region.
- It is **elevated ~6× for the first seven decode steps**, then settles to a lower floor. Whatever is special
  about the first few steps is not the KV *length* (an early guess, now excluded: events occur at decode index 3
  for prompts of length 61, 135, 159 and 657, i.e. at KV lengths 64, 138, 162 and 660 — all far past any
  32- or 64-entry boundary). The likelier explanation is **batch composition**: these requests are served by a
  continuous-batching server, and the first steps of a request are exactly when the running batch is changing as
  other requests start and finish, which changes the graph and the buffers it binds. That is a hypothesis, not a
  measurement.
- It is **finite**: no NaN in any of the 18, and the generation continues to full length.

## What it does not explain

Face A. Face A arms carry `HSA_STATUS_ERROR`, produce `empty` output, and show **no** logits divergence of this
kind. The two faces remain disjoint, and this document is about Face B only.

## The prediction this leaves, and the instrument that tests it

For any of the 18 events, the record now identifies the **exact arm, request and decode step** at which the run
leaves the consensus — for free, with no GPU. The next arm takes that step and compares the tensors it depends
on read two ways:

- if the plain-memcpy read and the `hrx_synchronous_d2h` read of the same buffer **disagree** at that step, the
  fault is an ordering/visibility defect and the fix is the visibility operation;
- if they **agree**, the wrong values are genuine device state and the race is on the device side.

That is `patch-input-f32-sync-twin.py` (verified to apply and compile), pointed at a step the archive already
knows. **No mechanism is named. But the fault is now located to a decode step, bounded to a rate, shown finite,
and the prefill excluded — which is a much smaller place to look than where this investigation started.**
