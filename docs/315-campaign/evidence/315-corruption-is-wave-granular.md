<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The corruption is WAVE-GRANULAR: it starts on a wave64 boundary in 96.6% of cases (chance 1.6%)

Measured from the archive, controlled for the artifact that would have faked it. This is the sharpest
quantitative result in the #315 work, and it **discriminates between two candidates** that were both open.

## The measurement

Every hidden-state (`ne0 = 2048`) fingerprint reading that carries the extended `first=`/`row=` fields, split by
whether it is saturated (because a saturated reading is uninformative here — see the control below):

| subset | readings |
|---|---|
| partial (informative) | **266** |
| saturated (`nan == extent`, excluded) | 450 |

On the **266 partial readings**:

| quantity | observed | chance | enrichment |
|---|---|---|---|
| `first` (start of corruption) divisible by **64 elements** | **257 of 266 = 96.6%** | 1.6% | **~60×** |
| `nan` (extent) divisible by 64 elements | 157 of 266 = 59.0% | 1.6% | ~37× |

`first mod 64` histogram: **257 at 0**, then 4 at 40, 3 at 32, 1 at 37. There is essentially one residue.

## The control that made this trustworthy

The first cut over all 716 readings gave 98.7% and 84.8% — **and both numbers were inflated by saturation**, which
is the trap: for a fully-NaN tensor the extent is `2048 × T`, always divisible by 64, and `first = 0`, always
divisible by 64. So 450 of the 716 readings were structurally guaranteed to satisfy both tests. Excluding them
costs almost nothing — 96.6% and 59.0% — which is why the result stands. The uncontrolled version is reported
here rather than dropped, because the difference between them is the whole reason to trust the second.

## Why the unit matters, and what it discriminates

**64 *elements* of f32 = 256 bytes.** That is **one wave64** — 64 lanes of 32 bits each. It is **not** the 64-byte
cache line, which would be only **16 elements**.

That distinction decides between the two candidates that were open:

| candidate | its natural granularity | prediction for `first mod 64 == 0` |
|---|---|---|
| **host-staging coherence** (mine, `315-candidate-host-staging-coherence.md`) | **cache line = 64 bytes = 16 elements** | if corruption were only cache-line aligned, `first` would be a multiple of 16 and residue-0 mod 64 with probability ~**25%** |
| **wave64 lane masking** | **one wave = 64 elements = 256 bytes** | **~100%**, the observed 96.6% |

The observation is 96.6% at residue 0 — an order of magnitude above what cache-line alignment predicts.
**So the data favours the wave64 candidate and disfavours the coherence candidate** as the *granularity* of the
corruption. (It does not by itself kill the coherence candidate, which could still be a *cause* whose effect
happens to be wave-quantised — but the wave shape is no longer explained by it, and the burden shifts.)

## The mechanism this points at

A write that operates on **whole waves** — an entire wave64 written, or left unwritten, atomically — at the
**front** of a residual row (`first` also lands at row 0 or 1), with the rest of the row correct. That is the
signature of **lane masking**: a wave whose lanes are not all written, or whose lanes were not drained, rather
than of a short byte-range write (which would corrupt a prefix and leave a clean tail — the opposite of what is
seen, since it is the *rest* of the row that is clean and a *hole* that appears).

The record already carries this hardware class: the pin's `hrx-system` includes *"the GFX11 wave64 lane-mask
drain backport (**#313 NaN**)"*. **That backport is already in the pin**, so this is not #313 itself — it is the
same class of defect surviving it.

## What it does NOT establish, and how to test further

- **Not a cause.** It says the corruption is applied per wave, not what applies it. No instrument on the record
  observes the write (the producer probe is the closest and sees only host-visible tensors).
- **Predictions it makes, all checkable:**
  1. corrupted elements should fall in **whole waves within a row**, with a clean wave beside them — testable by
     dumping the row element-by-element rather than counting;
  2. counts should be `2048 − 64k` for small `k` when exactly one row is hit — the observed `k = 1` case;
  3. a **lane-mask drain** should be visible when the wave64 drain path is instrumented.
- **Sampling caveat:** 266 partial readings from a handful of arms and one configuration (`tail:16`, whose onset
  site this session showed to be instrument-relative). The *granularity* result does not depend on where the
  onset lands, but it should be re-measured under a name-list configuration before being treated as universal.
- **No rate:** arms were selected for having faulted.
