<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Face B: an invariant signature across 15 arms, present in every cell — and what the fields actually count

Extracted from the arm verdict lines in every `315*-console.log`. No GPU time.

## The signature is identical in all 15 arms

    nan=0 req=<5..27> parse=0 utf8=0 http=1 transport=0 s500=1 unparsed=2 hsa=0

Every Face-B fault on the record reads **exactly** `nan=0 http=1 s500=1 unparsed=2` with `parse=0`, `utf8=0`,
`transport=0`, `hsa=0`. There are no near-misses and no partial variants: Face B is one failure mode with an
invariant fingerprint, which is a stronger starting point than Face A had.

## It occurs in every cell, including the control cells

| cell | arms |
|---|---|
| `e` (eviction forcer) | rounds 1–4, `side=1/1` |
| `q` (**quiet**, `side=0/0`) | rounds 2 — three separate runs |
| `c` (CPU load, `side=12/12`) | round 1 |

So Face B — like Face A — is **not eviction-specific**. The quiet cell, with no forcer running at all, produced
the same signature in at least three runs (09:50:50, 18:34:37, and the `OOR` one at 09:34:57), and the CPU cell
produced it too. Whatever the trigger is, **the eviction forcer is not necessary for either face**, which is the
same conclusion the record already reached for Face A from the control-cell faults and which Face B now
independently reproduces.

One Face-B arm is even `OOR` (`gtt_start=71G`, below the 95G gate) yet still carries the full signature — so the
face does not require the declared regime.

## What the fields count — three subtleties that matter before these numbers are reused

Read from `315-disc.sh:139-148`:

- **`unparsed=2` is a line count, not an event count.** `unr=$(grep -cE "unparsed Content-only output|does not
  match the expected" …)` counts *log lines* matching that pattern. A constant 2 across fifteen arms means the
  one event is logged on two lines (detector and server), not that two responses were unparsed. This is the
  record's own lesson — count occurrences, not lines — appearing again.
- **`http=1 s500=1` is the same failure counted twice**: once by the detector's `http_failures` and once by
  grepping `"code":500` in the server log. So the Face-B event is **one** HTTP 500.
- **`FAULT` is also assigned to truncated arms.** Line 148 makes `req < 66` a fault condition on its own. Every
  Face-B arm has `req = 5…27`, so `FAULT` here mixes "the run failed" with "the run stopped early" — which is
  correct for the rig's purposes but means a `FAULT` verdict alone is not evidence of a *content* failure. The
  content evidence is `http=1 s500=1 unparsed=2`.

## A discrepancy with the objective's wording, recorded not resolved

The objective describes Face B as *"invalid-UTF-8 output surfacing as HTTP 500 parse failures"*. In every
Face-B arm the detector's own field is **`utf8=0`** and `parse=0`: the recorded failure is the **HTTP-500 /
unparsed** path, and neither the UTF-8 check nor the parse-failure counter fired. Either the invalid bytes are
caught downstream as an unparsed response before the UTF-8 check sees them, or the field names do not mean what
the objective assumes. **This is stated as an observation about the detector's semantics, not as a refutation of
the objective** — resolving it needs the detector's source, which is `~/wt/315-detector.py` on the box.

## Why this matters for the goal

Face B now has, for the first time, a **uniform, box-free characterization** to test an instrument against: any
run claiming to localise Face B must produce `http=1 s500=1 unparsed=2` in at least one arm, and its
finite-hash divergence must be sought in *that* arm. It also confirms the structural point from
`315-faceb-has-no-capable-instrument.md` from a second direction: because the arms with these markers carry
**zero** fingerprint lines (the NaN gate), Face B's uniformity has never been matched to a tensor-level
measurement.

## Caveats

- These are arm-level fields from the rig's own classifier; none is a tensor measurement.
- The arms span several weeks and two rigs, and the later ones ran on a thermally saturated box.
- No rate is given: the count of Face-B arms on this record is 15, with no denominator extracted here.
