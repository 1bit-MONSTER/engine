<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The NaN is born in layer 0 — the first faithful localization in this campaign

## The run

`315lout4` round 1 arm **e** (eviction cell): Face A, `nan=1` at `req=6`, **`gtt_start=98G` (in regime)**.
Guard: `count=154880 of 154880 ... sentinel=0` — the sentinel was armed and the logits were **written** with
NaNs, exactly as `315-sentinel-preregistration.md` established. Selector: `l_out-0, l_out-3, l_out-6, …, l_out-45,
46, 47` — the reduced 16-layer sample, on `315fp-build`. `l_out-<layer>` is the **last node of its layer's chunk**,
so each reading is the least displacement-exposed tensor available: the faithful class.

## The readout, complete and verbatim (all 20 lines)

    call=29547 name=l_out-0  nan=1984   of 200704  hash=8275da278cc381c6
    call=29549 name=l_out-1  nan=69568  of 200704  hash=3a2840df7a01a06e
    call=29551 name=l_out-2  nan=73664  of 200704  hash=aa14fd6c3f5d05b8
    call=29553 name=l_out-3  nan=174752 of 200704  hash=3d64dcc47ff90063
    call=29555 name=l_out-4  nan=200608 of 200704  hash=aeb7827c8d1130fa
    call=29559 name=l_out-6  nan=195584 of 200704  hash=536c9356d586c492
    call=29565 name=l_out-9  nan=200672 of 200704  hash=b99e0aff27eecda3
    call=29571 name=l_out-12 nan=200672 of 200704  hash=b99e0aff27eecda3
    call=29577 name=l_out-15 nan=200704 of 200704  hash=ba52bebcb7a7d383
    ... l_out-18 … l_out-45:  200704 of 200704, every one hash=ba52bebcb7a7d383
    call=29639 name=l_out-46 nan=2048   of 2048     hash=89ade87e7a93fb83

**Two things fall out immediately.**

### 1. The corruption is smallest at the START, not the end

    layer  0:   1,984 / 200,704  =  0.99 %     <- one token's row, almost exactly
    layer  1:  69,568 / 200,704  = 34.7 %
    layer  2:  73,664 / 200,704  = 36.7 %
    layer  3: 174,752 / 200,704  = 87.1 %
    layer  4: 200,608 / 200,704  = 99.95 %
    layer  9: 200,672 / 200,704  = 100 %
    layer 15+: 200,704 / 200,704 = 100 %

**The residual stream is essentially clean at layer 0 — 0.99 % — and is fully NaN by layer 15.** The NaN is not
introduced late and amplified; it is introduced **at or before the output of layer 0** and spreads.

`200704 = 2048 × 98`, so the layer-0 count is **about one token's row** (2048 wide), and it is
`1984 = 2048 − 64` — a **structured** deficit of exactly one 64-wide block, which is worth noting precisely
because it is structured. This also **reproduces the "exactly one token's row" invariant** the record had flagged
(section 47) and could not reproduce in the by-name arms — it appears here at **layer 0**.

### 2. Everything from layer 15 to layer 45 is the same tensor

From `l_out-15` to `l_out-45` the count is the full tensor and **the hash is identical across all of them**
(`ba52bebcb7a7d383`). Once every element is NaN the residual stream carries no information, so every subsequent
layer's output is the same saturated tensor. **Nothing downstream of ~layer 15 can be informative, in either
direction** — and the record's per-node leads (`ffn_shexp-41`, `norm-42`, the head) are all downstream of it.

## What this changes

- **The search has been in the wrong half of the model.** The campaign has spent its per-node effort on layer 41
  and the head; both are downstream of a stream that is *already fully NaN by layer 15* and 35 % NaN by layer 1.
- **The origin is layer 0's own computation, or the embedding that feeds it.** That is the one region the record
  has never covered: the `strstr` defect limited the named arms to layers 4 and 41, and `l_out-0` had never been
  read at all before this arm.
- **The shape is "one row first", not "everything at once"** — 0.99 % at layer 0 growing to 100 % by layer 15.
  That is the signature of a **localized** corruption that propagates through attention (which mixes tokens),
  not of a global numerical blow-up.

## Caveats, stated at the strength the evidence supports

- **One arm.** The fault is intermittent; this is a single in-regime Face-A arm. The gradient is smooth and
  monotone, which is not what noise looks like, but it is n = 1.
- **Faithfulness is relative.** `l_out-<layer>` is last in its chunk, so it is the least displaced reading
  available — but not proven absolutely faithful.
- **NaN counts grow for a trivial reason too.** Once a token's row is NaN, attention mixes it into others, so
  growth is *expected* from any single-row origin and does not by itself prove the origin is layer 0 rather than
  earlier. **What it does prove is that the origin is at or before layer 0's output** — the pre-layer-0 inputs
  (embedding) remain candidates and have not been read.

## The next measurement, which is now cheap and specific

`l_out-0` is the layer-0 chunk's **last** node, and a `tail:N` window covers a chunk's **last N** nodes — the
faithful class. So a `tail:16` run with `NANONLY=1` (the configuration that produced 16 Face-A faults in 17 arms
in `315nan`) would print **which of layer 0's final 16 nodes first carries NaN**, localizing within layer 0
without any new instrument. Everything before that window — the embedding and layer 0's attention — is
displacement-exposed by name, and would need the read-at-production instrument.
