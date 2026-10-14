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

---

# CORRECTION to the interpretation above: the measured shape is a contiguous 256-byte-aligned PREFIX

The granularity result above (96.6% of corruptions begin on a 64-element boundary) is sound and stands. But I
took it further than it goes by reading it as "whole waves with a clean neighbour", and the fuller shape
measurement corrects that. Both are now recorded, the measurement first.

## What the extents actually show (266 partial readings)

| property | measurement |
|---|---|
| NaN is a **single contiguous run** from `first` | **266 of 266** — `first + nan ≤ extent` in every reading |
| the run **starts at element 0** (`first = 0`) | **227 of 266 = 85.4%** |
| the run **covers the end of the buffer** | 30 of 266 |
| the run is **interior** (neither) | **9 of 266** |
| `first` is 256-byte (64-element) aligned | 257 of 266 = 96.6% |
| the run length is 256-byte aligned | 157 of 266 = 59.0% |

Representative shapes:

    first=      0 nan=  2048 extent= 79872   rows [0.00:  1.00] of  39.00   <- exactly one row, the first
    first=      0 nan= 55296 extent= 79872   rows [0.00: 27.00] of  39.00   <- first 27 rows
    first=      0 nan=  1856 extent=  2048   waves[0:29] of 32              <- 29 of one row's 32 waves
    first=      0 nan=1572928 extent=2441216 rows [0.00:768.03] of 1192.00   <- large prefix
    first=  59904 nan= 19968 extent= 79872   rows [29.25: 39.00] of 39.00   <- the 30 tail cases
    first= 264192 nan= 88064 extent=352256   rows[129.00:172.00] of 172.00   <- tail from row 129

So the picture is **a contiguous run of wrong data starting at the front of the buffer in 85% of cases**, not a
scattered set of waves. The `1984` case is "the first 31 waves of a 32-wave row are wrong and the **last** wave
is clean" — i.e. a prefix of the row, not a hole in the middle of it.

## What this does and does not settle

**Settled by measurement:** the corruption is contiguous, and its boundaries fall on **256-byte** (64-element)
lines far more often than chance — 96.6% versus 1.6% for the start. That is a real granularity result and it is
much coarser than a 64-byte cache line (16 elements), which is what the staging/coherence candidate's mechanism
would act on.

**NOT settled — and I over-read it before:** *what* applies 256-byte-aligned contiguous prefixes. Two readings
remain live, and the data does not separate them:

| reading | what it would mean | what would separate it |
|---|---|---|
| **a copy/region boundary** | a `copy_buffer`/region operation whose extent is short, or is applied to a sub-range, and whose source held NaN — the runtime's own 256-byte alignment unit (the record's "256 the alignment the record gives for the MoE tables", and `7936 = 8192 − 256`) would then be the natural boundary | whether the *unwritten tail* keeps its previous content or the *prefix* is actively wrong: a short copy leaves the tail stale; a wrong prefix means the copy's source is bad |
| **wave64 lane masking** | a kernel writing whole waves, with the first waves' lanes wrong and the last wave's drain correct | a lane-mask drain visible with the wave64 drain path instrumented, and a per-element dump showing whole waves wrong with a clean neighbour inside a row |

**The discriminating observation is therefore the same for both, and it is cheap:** dump one corrupted row
element by element and see whether the wrong elements are (a) a prefix of whole waves with the tail clean
(region/copy), or (b) whole waves interleaved with clean waves (lane masking). Counting NaN cannot distinguish
them — which is exactly why the count-based reading went as far as it did and no further.

**Also unchanged:** this is not #313 (whose wave64 lane-mask drain backport is already in the pin), it does not
establish a cause, and no instrument on the record observes the write. Sampling remains 266 partial readings
from a handful of arms in one (`tail:16`) configuration, whose onset site this session showed to be
instrument-relative — though the *shape* result does not depend on where the onset lands.

---

# SECOND CORRECTION: contiguity was never measured — `nan` is a count and `first` is an index

The section above states that "the NaN is a single contiguous run from `first`" on the evidence that
`first + nan ≤ extent` in 266 of 266 readings. **That evidence does not support that claim and the claim is
withdrawn.**

`first + nan ≤ extent` is an arithmetic identity that any reading satisfies whenever the NaNs fit in the space
after the first one — contiguous or scattered. It is satisfied by a single run, by two runs, by every other
element, and by a run at the end plus one at the start. **The fingerprint reports exactly two numbers, `first`
(the index of the first NaN) and `nan` (how many there are); it never measures whether they are adjacent.** So
"one contiguous run" was an assumption dressed as a measurement, in a document whose whole point was to avoid
that — and it is the same failure mode as the raw row-index census earlier in this session (a count used to
support a positional claim).

## What is actually measured

| property | status |
|---|---|
| the **first** NaN is at element 0 (`first = 0`) | **measured**: 227 of 266 = 85.4% ⇒ the corruption *includes* the front of the buffer |
| the NaN **count** is 256-byte aligned | **measured**: 157 of 266 = 59.0% |
| the first NaN is at a 256-byte boundary | **measured**: 257 of 266 = 96.6% |
| the NaNs form one contiguous run | **NOT measured — withdrawn** |
| a "clean tail" or a "clean wave beside a wrong one" | **NOT measured** — both presuppose contiguity |

So the two shapes the last section offered as alternatives — *prefix of whole waves with a clean tail* versus
*waves interleaved with clean waves* — **cannot be separated from this data at all**, and neither can be
asserted. What survives is narrower and still useful: **the corruption always reaches down to the front of the
buffer (85%), and its boundaries/counts fall on 256-byte lines far more often than chance.**

## The consequence for the next instrument

The per-element dump was named as the discriminator between the two readings. It is now not merely *a* way to
choose between them — it is **the only way**, because the count-based instrument is structurally blind to the
distinction. Any run that wants to settle whether the corruption is a region or a lane pattern must dump a row
**element by element** (or report per-wave NaN presence), not a total.

That also means: **not one of the 266 readings on the record can distinguish a region from a lane pattern**, and
no amount of further counting will. The instrument to add is a per-wave (32-bit mask across the row's 32 waves)
or per-element dump — small, diagnostic-only, and in scope.
