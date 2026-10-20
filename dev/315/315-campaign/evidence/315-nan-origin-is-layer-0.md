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

---

# This contradicts a recorded conclusion, and resolves a tension the record had already flagged

The finding above does not sit quietly beside the record. It contradicts one conclusion and settles another.

## The contradiction: §40 says the corruption is born at block 41

`dev/315/315-mechanism.md` §40, on the `tail:16` window:

> its window covers **block 41's FFN and block 42's attention** ... **Every earlier call's window was clean, so the
> corruption is *born* at block 41, not inherited from block 0.**

That is a direct conflict with the gradient measured here: `l_out-0` carries NaN (1,984), `l_out-1` carries 34.7 %
of a 98-token tensor, and the stream is saturated by layer 15. **If the corruption were born at block 41, layers
0–40 would be clean.** They are not.

## The tension the record had already flagged, and how this resolves it

The same record is not of one mind, and it says so. `315-row-p1c-inregime-mechanism.md`:

> It does **not** reproduce section 47's "exactly one token's row" invariant, and does not refute it either ...
> The two observations are in tension, and until the same instrument is run over the same arms, "one row" must
> not be quoted as the fault's shape.

That arm had `l_out-4` at **20480 of 20480 — the whole tensor** — while §40's window, over an apparently different
episode, was clean before block 41. Two arms, two conclusions, explicitly left unresolved.

**The gradient here resolves it, and in the direction the `l_out-4` reading pointed**: the stream is already
essentially saturated at layer 4 (99.95 %), so an arm whose earliest instrumented layer is 4 can only report
"everything is NaN from layer 4 on" — which reads as *born nearby* if one does not look earlier. The `strstr`
defect meant no arm ever looked earlier than layer 4. **This is the first arm that reads layer 0.**

## Why the reading is probably not a displaced read — the check that matters

The obvious way to explain a too-early NaN is arena displacement: read `l_out-0`'s slot after later chunks have
reused it, and report a later layer's NaN as layer 0's. **The readings argue against it:**

| arm | `l_out-0` | `l_out-1` | `l_out-2` |
|---|---|---|---|
| `r1e` | **1984** | 69,568 | 73,664 |
| `r2e` | **1984** | 14,272 | 16,320 |

If `l_out-0`'s slot had been overwritten by `l_out-1`'s value, `l_out-0` would read whatever `l_out-1` reads —
and `l_out-1` differs between the two arms (69,568 vs 14,272) while `l_out-0` is **identical** (1984). Displacement
would have to have reproduced exactly 1984 twice in two arms of different sizes. The gradient is also monotone and
smooth, and the **clean quiet arm in the same session shows zero NaN at every layer** — so these counts are
tracking the fault, not noise.

## The arbiter the record already specified

§40 writes down, and explicitly does not run, the probe that settles causality for block 41:

    GGML_HRX_TENSOR_FP=l_out-40,ffn_inp-41,ffn_norm-41,ffn_shexp-41,ffn_swiglu-41,ffn_moe_out-41,ffn_out-41,l_out-41,
                       attn_norm-42,ffn_inp-42,ffn_norm-42

> the chain probe above is written down but not yet run

**That probe is now more valuable than when it was written**: it should be run *together with* `l_out-0…l_out-4`, so
that one arm can show both the early-layer counts and block 41's chain. If `ffn_inp-41` and `ffn_norm-41` are clean
while `l_out-0` is not, §40's conclusion is refuted and the early origin stands; if the early layers are clean in
that arm, then this gradient and §40 describe two different episodes and only the arm-in-common comparison can
separate them.

## What must not be claimed yet

- **One session, four arms, three Face A.** The gradient is consistent across the three and the constant 1984 is
  striking, but this is not yet a rate and not yet an independent replication outside this configuration.
- **This is the heavily instrumented configuration** (name list + `COPY_PROBE` + `SENTINEL` + `LOGITS_FP`). The
  face an arm takes is dose-dependent (see the selector comparison), so "which layer the NaN is born in" may also
  be dose-dependent. **That has not been tested**, and it is the strongest reason to hold the conclusion loosely.
- **§40 was not wrong about what it saw.** Its window really was clean before block 41 *in that arm*. The claim
  being contested is the generalisation from one window to "born at block 41".

---

# Pre-registration: where inside `l_out-0` do the 1,984 NaN sit?

Written before the run. Answers the one thing `315fp-build` cannot: this build prints `bytes / nan / of / hash` and
**omits `ne0`, `first` and `row`** — `315row-build` prints them (`[hrx-tensor-fp] ... ne0=%lld first=...`). Without
them, "contiguous prefix of 1984", "suffix of 1984", and "31 scattered 64-blocks" are indistinguishable.

## The run

    BIN=~/wt/315row-build/bin
    EXTRA_ENV="GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 \
               GGML_HRX_TENSOR_FP=l_out-0,l_out-1,l_out-2,l_out-3,l_out-4 \
               GGML_HRX_TENSOR_FP_NANONLY=1"

Narrow selector (5 names — a *lower* dose than the 16-name runs, which is the direction that favours Face A), the
configuration `315row` itself used for that build, plus `SENTINEL`. `LOGITS_FP` is absent from this build and is
therefore omitted rather than silently ignored.

## Readout, fixed in advance

`first` is a **byte** offset (established earlier in this campaign: `ffn_moe_weights-4` printed `first=32` for a
tensor whose first NaN is at element 8). So, for `l_out-0` with `nan=1984`:

| `first` | meaning |
|---|---|
| **0** | the NaN are a **prefix**: elements 0…1983, i.e. **the last 64 elements of the row survive** |
| **256** | elements 64…2047, i.e. **the first 64 elements — exactly one 256-byte block — survive** |
| other | report the offset; a value that is a multiple of 256 (the alignment unit the record gives for the MoE tables) is noteworthy, one that is not, more so |
| `nan` ≠ 1984 | the constant does not reproduce under this build; recorded, and no shape claim is made from it |

`row=` is read alongside: which row the first NaN falls in distinguishes "the corruption is one row's tail" from
"the corruption is a fixed byte range spanning rows".

## Why this is worth a run rather than an inference

`1984 = 2048 − 64` and `64 f32 = 256 bytes`, and **256 is the alignment the record gives for the MoE tables**
(`dispatch-mul-mat-id-common.h:311-317`, `alignment 256`). If `first=256`, the shape is "one 256-byte-aligned block
at the head of a row survived, the remaining 1,984 elements did not" — a partial-write-shaped corruption, and a far
more specific thing to look for than "layer 0". If `first=0`, it is a tail loss and that reading dies.

## Confounds

- **Different build and pin.** `315row-build` is from the `315row` session's pin, not the current one, so the shape
  measured is that build's. A shape that reproduces across builds is stronger; a shape seen in only one is a lead,
  not a fact.
- **Lower dose than the runs whose count it is checking** (5 names vs 16). If the count differs, that is itself a
  dose-dependence datum, and no shape claim follows.
- **Instrument induces**; no rate, no acceptance arm.
