<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# `1984` reconciles the two populations: one 256-byte block is the whole difference

Read from the archived logs while `315cfg` runs. No GPU time.

## The two counts, and where each appears

Across every arm on the record, `nan=1984` occurs in exactly **seven** readings:

| site | arm | extent | T |
|---|---|---|---|
| `l_out-0` | call 29547 | 200,704 | 98 |
| `l_out-0` | call 40377 | 276,480 | 135 |
| `l_out-0` | call 40377 | 276,480 | 135 |
| `l_out-0` | call 75717 | 503,808 | 246 |
| `ffn_moe_out-41` | call 51764 | 352,256 | 172 |
| `ffn_out-41` | call 51764 | 352,256 | 172 |
| `l_out-41` | call 51764 | 352,256 | 172 |

and `nan=1790` once (`l_out-0`, 20,480, T=10).

The layer-41 readings are the *same* chunk as the `nan=2048` readings this session has been working from
(`call=51764` is the layer-41 chunk, where `node_2948` read `2048`). So **within one chunk** the counts 2048 and
1984 both appear, and across arms the layer-0 site reads 1984 and 1790.

## The arithmetic that ties them together

    1984 floats × 4 = 7936 bytes
    2048 floats × 4 = 8192 bytes
    2048 − 1984    = 64 floats = 256 bytes

**7936 is exactly the number the earlier row-size session reported** as "one row's worth of bytes, minus one
256-byte block", and 256 bytes is the alignment unit already on the record for the MoE tables. So the earlier
session's layer-0 `7936` and this session's layer-41 `8192` are **not two different sizes** — they are the same
one-row region differing by **one 256-byte alignment block**.

That reframes three things that had been separate findings:

- The "fixed size" is not `8192` or `7936`; it is **one row, read at an alignment that costs it 0 or 64
  floats**.
- The two onset populations (layer 0 and layer 41) may be the **same corrupted region**, reaching the host read
  through buffers whose 256-byte alignment differs — which is a property of the allocator and the arena, not of
  the layer.
- `1790` at `l_out-0` (extent 20,480, T=10) is the outlier to explain: `2048 − 1790 = 258 ≈ 256 + 2`, i.e. one
  alignment block plus two floats, so it is consistent with the same picture only if two further floats were
  corrupted or the read straddled slightly differently.

## The hypothesis this gives, stated so it can be wrong

> The corrupted region is a fixed **byte** extent of about 8 KiB in a host-visible arena slot; whether a given
> tensor's read reports `2048`, `1984` or `1790` NaNs depends on that tensor's **byte offset mod 256** relative
> to the region, so the NaN *count* is an alignment artifact of a fixed-size corruption.

**Falsified by:** the same tensor at the same offset reporting both 2048 and 1984 across arms; or NaN regions
found at byte offsets that are not congruent modulo 256; or a case where `finite + nan` does not equal the
extent (which would mean the read was partial after all — it did equal it in all four arms where it was
checked).

**What it would buy:** it would make the "which layer" question secondary — the layer varies, the *region size
and its alignment relationship* do not — and it would point the fix at the arena's allocation alignment rather
than at any particular op or layer, which is consistent with the standing conclusion that no host-side
mid-graph read can name the site.

## What this does NOT establish

- It does not show the corrupted region is at a fixed byte offset; only that the *counts* are consistent with
  one.
- It does not touch the `nan=2048` measurements' validity — the 18-of-19 invariance and the `finite + nan =
  extent` identity stand as they are.
- It gives no rate, and the arms compared were run under different configurations and, for `315size`, a
  thermally saturated box.
- It is a synthesis of counts, not a mechanism: nothing here says what writes the region.

## Reproducing

    grep -ahE "hrx-tensor-fp" ~/wt/rt-det/*/server.log | grep -aE "nan=(1984|1790) of"

### Self-test of the hypothesis, done immediately: it does not hold as stated

The hypothesis says the count is a function of the tensor's **byte offset mod 256** relative to a fixed region.
If that were the whole story, the *same tensor* would give the *same* count. It does not: `l_out-0` reads
**1984** in three arms (extents 200,704 / 276,480 / 503,808) and **1790** in a fourth (20,480).

Two readings survive that, and they are worth separating because only one of them is testable without more
data:

- **The offsets may differ between arms.** The four `l_out-0` readings come from arms with different sequence
  lengths, so their buffers are different sizes and the allocator may place them at different alignments. Then
  the hypothesis is not falsified — but it is also **not tested**, because no offset was recorded. Testing it
  needs the arena's allocation offsets per arm, which the logs do not carry.
- **Or the count is not an alignment function at all**, and `1984` and `1790` are two genuine sizes,
  `1984 − 1790 = 194`, which is not a multiple of 64.

So the honest status of the hypothesis is **untested, not supported**. What the 1984/7936 arithmetic does
establish is narrower and still useful: the earlier session's `7936` and this session's `8192` are **one row
differing by exactly one 256-byte block** — a fact about the two reported figures, independent of whatever
produces them. The mechanism reading of it is a conjecture that this note has just failed to confirm.

The test that would settle it is concrete and cheap: the same tensor read in the same arm at two known arena
offsets, or the arena offsets printed alongside the counts. Until then the claim stays at the arithmetic.
