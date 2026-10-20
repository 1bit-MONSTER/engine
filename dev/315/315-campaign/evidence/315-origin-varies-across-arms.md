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

# The origin is not in the same place in different arms — and that is itself the finding

## The pre-check that produced it

Before running the §40 chain probe, its node names were checked against every fingerprint line this campaign has
ever recorded:

    ffn_inp-41     0        ffn_shexp-41   30        ffn_out-41      7
    ffn_norm-41    0        ffn_swiglu-41  30        l_out-41        9
                           ffn_moe_out-41  7        attn_norm-42    2
    ffn_inp-42     0        ffn_norm-42     0

Four of the chain's names have **never once appeared**. That is not the names being wrong: **every one of those runs
used `GGML_HRX_TENSOR_FP_NANONLY=1`, which prints nothing for a tensor with zero NaN.** A name that never appears is
a name whose tensor was always clean when read — or was never inside the window at all. Both readings are
unavailable from the logs, which is the same ambiguity as defect #14.

**What this does to §40's evidence.** §40 concluded *"every earlier call's window was clean, so the corruption is
born at block 41"*. With `NANONLY`, a clean window and an unmeasured window look identical in the log, and
`ffn_inp-41`/`ffn_norm-41` were never in any name list — the §40 probe that would have covered them is the one it
says is *"written down but not yet run"*. So §40's positive claim rests on the **absence of lines for windows that
were in `tail:16`**, which is sound for those windows, and says nothing about nodes the window never reached.

## The conflict, stated at full strength

| source | arm | earliest layer read | reading |
|---|---|---|---|
| §40 | (tail:16 arm) | 41 (window) | clean before block 41 → *born at 41* |
| this session | `315lout4` r1e, r2e, r1c | **0** | `l_out-0` carries NaN; saturated by layer 15 |

`l_out-<layer>` **is** inside a `tail:16` window (it is last in its chunk), so in §40's arm `l_out-0`, `l_out-1`, …
would have printed had they carried NaN. They did not. **In this session's arms they do.** The two accounts cannot
both be general.

## The reconciliation, and why it is a finding rather than a fudge

The two arms are not the same episode. The fault is intermittent and its **trigger position varies** across arms —
recorded `req` values include 6, 8, 10, 34, 54, 59 — and the origin now measured varies with it:

| arm | cell | `l_out-0` | signature |
|---|---|---|---|
| `r1e` | eviction | 1,984 | grows to saturation by layer 15 |
| `r2e` | eviction | **1,984** (identical) | same shape, different tokens |
| `r1c` | CPU | **46** | saturates by layer 2 — a *different* shape |

**So the fault is not anchored to one node or one layer.** In one arm the earliest NaN sits at layer 0; in another
episode the first corrupted window is block 41. That is exactly what a **position-independent** mechanism looks
like — one in which the corruption follows *which memory region* was affected, not which op ran.

That reading is a better fit to the trigger than a fixed site has ever been:

- the intervention is **KFD eviction**, which acts on **memory regions**, not on ops;
- the corrupted magnitude is a **fixed count** (`1984 = 2048 − 64` in two arms of different token counts — a region,
  not a row, and not a proportion);
- the whole tensor saturates quickly once any of it is NaN, so the observable is the **position of the first
  corruption**, which is free to move with the region;
- and the two faces never co-occur (142 arms, 0 with both), consistent with one region being hit per episode.

**The hypothesis, stated falsifiably:** the fault is a corruption of a *transient memory region* whose contents at
the moment of the event decide where the first NaN appears — not a defect located in a particular layer, op, or
kernel. Its prediction is that the origin layer **varies across Face-A arms** while the trigger stays fixed.

**How it would be falsified:** run ≥10 in-regime Face-A arms with the early-layer selector and show the first
NaN-bearing layer is the **same** in every one. If it is fixed, a site-specific mechanism is back on and §40's
reading is the right one; if it moves, the region-following account is supported and the search should move to
*what the regions have in common* rather than to any layer.

**That is a cheap experiment** — the selector already exists and the dose is already known to give Face A with a
clean quiet cell — and unlike every previous framing it makes a prediction that distinguishes the two accounts
without needing a new instrument.

---

# Correction: the variation I cited is between CELLS, not between episodes — and the prediction is not testable as written

Reviewing the section above against its own evidence, the case for "the origin moves" is weaker than it states, in
three specific ways.

**1. The arms I cite do not differ in the way the hypothesis needs.** `r1e` and `r2e` are both **eviction** arms and
both give `l_out-0 = 1984` — *identical*. The one arm that differs (`r1c`, `l_out-0 = 46`) is the **CPU-pressure**
cell. So the variation on the table separates *cells*, not episodes of the same trigger. The objective's trigger is
eviction, and under it the layer-0 signature was **constant across two arms of 98 and 135 tokens**. That is evidence
for a *fixed* region, not a position-independent one.

**2. Layer 0 is already NaN in every arm I measured, so "which layer carries the first NaN" cannot vary below what
I can see.** All three Face-A arms put NaN at `l_out-0`; the earliest selector this session read **is** layer 0. A
hypothesis predicting movement in the first affected layer is therefore **not testable with this selector** — the
movement would have to be in the embedding or layer 0's internals, which are outside the window. I stated a
falsification ("run ≥10 arms and show the first NaN layer is the same") that could come out "same" purely because
the instrument cannot see earlier. **That is the defect #14 ambiguity again, written into a pre-registered-looking
prediction.**

**3. §40's contrary reading rests on an absence, which I have already argued is ambiguous.** I used that to weaken
§40 — correctly — but the same standard applies to my own comparison against it. A "clean layer 0" inferred from a
missing `NANONLY` line is not the same as a measured clean layer 0.

## What actually survives from the section above

- **`ffn_inp-41`, `ffn_norm-41`, `ffn_inp-42`, `ffn_norm-42` have never appeared in any recorded fingerprint**, while
  `ffn_shexp-41` has 30 times. §40's chain probe has genuinely never been run, and §40's conclusion rests on
  windows, not on those nodes. **That stands, and it is why the combined probe is worth running.**
- **Layer 0 carries NaN in all three Face-A arms measured** (1984, 1984, 46) — new, replicated within-session, and
  in direct tension with §40's generalisation.
- **Under the eviction trigger the layer-0 count was constant (1984) across two arms of different size.**

**What does not survive:** the claim that the origin moves across arms, and the prediction that tests it. The
region-following mechanism remains a *hypothesis* — it is not yet supported by any variation between episodes of the
same trigger, because every eviction arm measured gives the same number.

**What would test it:** arms that read **earlier than layer 0** (the embedding and layer 0's own internals), where
movement could actually be seen — and that is a displacement-exposed read, i.e. the read-at-production instrument
again. The honest position is the one this campaign keeps arriving at: the measurement that would separate the
remaining hypotheses is the one the host-side instrument cannot make.
