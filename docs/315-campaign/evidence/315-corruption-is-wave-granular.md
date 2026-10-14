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

---

# PRE-REGISTRATION: what the wave mask must read for each shape, fixed before the run

The mask instrument is built and rides in both arms of the corrected `315ab` campaign. Its output is a 32-bit
hex word; these are the values the two candidate shapes must produce, fixed **before** any mask line exists, so
the result is scored rather than read.

Bit *w* is set iff wave *w* — elements `[64w, 64w+64)` — contains at least one NaN, for the first 32 waves
(elements 0…2047 = one `n_embd` row). `0x7FFFFFFF` = `0b0111…1111`; `0xFFFFFFFF` = all 32 set.

| shape | predicted mask | why |
|---|---|---|
| **one corrupted region at the front of the row** | a **contiguous run of set bits from bit 0**, e.g. `0x7FFFFFFF` for a 31-wave region | a region is contiguous by definition |
| **one clean wave inside an otherwise-wrong row** (the `1984` = 31-of-32 case) | `0x7FFFFFFF` if the *last* wave is clean; `0xFFFFFFFE` if the *first* is | it pins which wave the hole is in, which the count alone cannot |
| **whole-row corruption** | `0xFFFFFFFF` | every wave poisoned |
| **a lane/wave-interleaved pattern** | bits set with **gaps**, e.g. `0x55555555`, `0xAAAAAAAA`, or any non-contiguous run | this is the one shape a *count* can never show |
| **no mask line printed, though a fault occurred** | the instrument did not arm in that arm | INVALID for the mask question, and it must not be read as "no waves" |

**The decisive comparison, stated in advance:** if every faulting arm's mask is a contiguous run, the corruption
is a **region**, and the wave64-lane reading loses; if any arm shows a **non-contiguous** mask, it is a
**lane/wave pattern**, and the region reading loses. A count of 1984 is consistent with both, which is exactly
why this instrument exists.

**Also pre-committed, because it is easy to get wrong after the fact:** the mask covers only the **first 32
waves (2048 elements)**. For a tensor whose `ne0 != 2048` the bit indices no longer correspond to one row's
waves, and such a reading must be **excluded** rather than reinterpreted. Every hidden-state tensor on this
record has `ne0 = 2048`, so this excludes nothing observed so far — but it is stated now, not after a
surprising mask appears.

**What a mask result does not do:** it does not name the writer, and it does not by itself establish a
mechanism. It decides one binary question — region or lane pattern — that every previous instrument was blind
to.

---

# LIMITATION: the front/tail split is dominated by TENSOR FAMILY, so "85% front" must not be pooled

Checked because a position result that aggregates across tensor families can hide a family effect, and it does.

| region shape | tensors it is made of |
|---|---|
| **FRONT** (`first = 0`), 227 readings | `ffn_moe_down` **89**, `ffn_moe_weighted` **63**, `ffn_shexp` 21, `norm` 19, `ffn_out` 7, `node_3019` 4 |
| **TAIL** (covers the end), 30 readings | **`l_out` 13**, `ffn_out` 10, `ffn_moe_weighted` 4, `result_norm` 2, `node_3305` 1 |
| **INTERIOR**, 9 readings | scattered, no family above 2 |

So the two shapes are **not** the same phenomenon seen at different places — they are **largely different tensors**.
FRONT is overwhelmingly the **MoE path** (`ffn_moe_down`, `ffn_moe_weighted`); TAIL is the **residual output**
(`l_out`, `ffn_out`). Per arm the mix varies (e.g. `315fp2-p2l` is 156 front / 0 tail, while `315row-p1c` is
1 front / 3 tail), so the pooled 85.4% is a weighted average over families that behave differently.

**Consequences, all of which narrow the earlier result:**

- **"The corruption starts at the front" is a statement about the MoE-path tensors** that dominate the pool, not
  a law about the corruption. The `l_out` family — the one the size invariant was measured on — is
  **over-represented in the TAIL** group.
- **The family split is a live alternative to any positional reading**: the two shapes could be one mechanism
  seen through two different tensors, or two different mechanisms, and **this data cannot tell which**.
- **A pooled percentage is not the right statistic here.** Any future statement of the position result must be
  **per tensor family**, with the family's own denominator — which this table now supplies.

**A read-faithfulness caveat that applies to the FRONT group specifically.** `ffn_moe_down` and
`ffn_moe_weighted` are *intermediate* MoE tensors, and the record already establishes that intermediate reads
can be unfaithful (arena slot reuse; the aliased `l_out-42 ≡ l_out-44` case). So the FRONT group leans on
exactly the reads the record distrusts most, while the TAIL group leans on `l_out`, the family with the
best-validated readings. That is an argument for trusting the TAIL observations more, not less — the reverse of
what the pooled count suggests.

**Unchanged by this:** the 96.6% 256-byte alignment of `first`, which is computed over the same partial readings
and is not obviously family-specific — but it should be re-cut per family before being quoted as a headline.

---

# The alignment result SURVIVES the per-family re-cut — it is not a family artifact

The limitation above says a pooled percentage is the wrong statistic and the alignment result must be re-cut
per family. Done, and it holds in every family:

| family | n | first on a 256-byte boundary | % |
|---|---|---|---|
| `ffn_moe_down` | 91 | 91 | **100.0%** |
| `ffn_moe_weighted` | 67 | 63 | 94.0% |
| `ffn_shexp` | 22 | 22 | **100.0%** |
| `norm` | 21 | 21 | **100.0%** |
| `ffn_out` | 19 | 17 | 89.5% |
| `l_out` | 16 | 15 | 93.8% |
| `node_2948`, `node_3019`, `node_3090`, `node_3161`, `node_3232` | 4 each | 4 each | **100.0%** |
| `ffn_moe_out` | 3 | 3 | **100.0%** |

**Pooled: 252 of 259 = 97.3%** across families with n ≥ 3 (and 257 of 266 = 96.6% over all partial readings),
against a **1.6%** chance rate. **Eight of the twelve families are at exactly 100%**, and the lowest is 89.5%.

So the alignment is **family-independent**. That matters in both directions:

- It **narrows** the position claim (the front/tail split really is a family effect, as the limitation says), and
- it **strengthens** the alignment claim, because it is no longer an average that a single dominant family could
  carry. `ffn_moe_down` alone is 91 of 259 readings; even excluding it entirely the remaining families are
  ~96% aligned.
- It also **weakens the read-faithfulness objection to the alignment** specifically: the caveat says the FRONT
  group leans on intermediate MoE tensors whose reads the record distrusts, but the alignment holds just as
  firmly on `l_out` (15/16) and `node_2948` (4/4) — the residual-stream tensors with the **best-validated**
  readings. An artifact of unreliable intermediate reads would not reproduce on the families the record trusts.

**Standing result, stated at the strength the data supports:**

> The corruption **begins** on a 256-byte (64-element) boundary in **96.6%** of partial readings overall and in
> **at least 89.5%** within every individual tensor family tested — including `l_out` and `node_2948`, whose
> readings the record treats as faithful — against a 1.6% chance rate. It is therefore a **256-byte-granular**
> corruption, not a byte- or element-granular one.

**What remains open, unchanged:** *what* applies 256-byte-aligned damage (a copy/region boundary versus wave64
lane masking), whether the NaNs are contiguous at all, and the writer. The mask instrument decides the second
of those and rides in the running campaign.
