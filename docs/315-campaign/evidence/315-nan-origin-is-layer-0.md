<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
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

---

# Replicated three times — and `l_out-0`'s NaN count is a constant, not a proportion

The reduced-selector session produced **four arms with fingerprints**: one clean and **three Face A**. All three
faulting arms show the layer-0 signature:

| arm | cell | tokens | total values | **`l_out-0`** | `l_out-1` | `l_out-2` | `l_out-3` | `l_out-4` |
|---|---|---|---|---|---|---|---|---|
| `r1e` | eviction | 98 | 200,704 | **1,984** | 69,568 | 73,664 | 174,752 | 200,608 |
| `r2e` | eviction | 135 | 276,480 | **1,984** | 14,272 | 16,320 | 254,472 | 276,448 |
| `r1c` | CPU | 10 | 20,480 | 46 | 245 | 20,448 | 20,448 | 20,448 |
| `r1q` | quiet | — | — | — (clean, 66/66) | | | | |

**The two eviction arms report `nan=1984` at `l_out-0` — the same number, to the digit, at 98 tokens and at 135
tokens, with different tensor sizes (200,704 and 276,480 values) and different hashes** (`8275da278cc381c6` vs
`ec5cc25ebe3aa882`). A proportion would have scaled with the token count; a per-token row would be 2048. **1984 is
a fixed count, and `1984 = 2048 − 64 = 31 × 64`.**

That is a structured number twice over — an exact 64-element deficit against one token's width, in arms whose
token counts differ by 37 — and it is the strongest clue this campaign has about the *shape* of the corruption
rather than just its location:

- **It is not "one token's row"**, which is 2048: it is that row **minus exactly one 64-wide block**.
- **It is not dilution**: the count is not the token count, and not proportional to it.
- The CPU-pressure arm does **not** share it (46 at `l_out-0`, growing to saturation by layer 2), so **the eviction
  path and the CPU path reach different layer-0 signatures** — the objective's trigger is eviction, and 1984 is the
  eviction signature.

## What the instrument cannot yet say

This build's fingerprint line prints `bytes / nan / of / hash` — **not** `ne0`, `first`, or `row`, which the
`315row` build did print. So **where** inside `l_out-0` the 1984 NaN sit is not recoverable from these arms: a
contiguous prefix of 1984, a suffix of 1984, and 31 scattered 64-blocks are all consistent with the count alone.

`315row-build` has the fields (`first=`, `row=`) and implements the fingerprint, so **a targeted `l_out-0` read on
that build would give the position directly** — and `first=` alone would distinguish "NaN starts at element 64 of a
row" from "NaN starts at 0", which is the difference between "one block survived" and "the first block is the
casualty".

## Status of the finding

Four arms now, three of them Face A, all in regime (`gtt_start=98G`), one clean quiet arm in the same session.
The layer-0 origin is no longer n = 1 — and the constant 1984 says the corruption has a **fixed shape** at layer 0,
not a diffuse one.
