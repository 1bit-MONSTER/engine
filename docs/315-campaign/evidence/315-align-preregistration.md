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

engine#315 pre-registration. Written BEFORE the run so the reading cannot be chosen post hoc.
NOT YET RUN — see the "Why not run yet" section.
-->

# Pre-registration: is the fault the #140 FA layout/dispatch class? (`315align`)

The one discriminator the record names as cheapest-and-never-applied. §57 of the mechanism doc says it
directly:

> "The cheapest remaining discriminator is still the one the project's own #123 work used and #315 has
> never applied: `GGML_HRX_FA_PARTIAL_ALIGN` … If the fault rate rises when the partial alignment is
> lowered, the fault IS the #140 layout-sensitive class and its surviving instance is at a boundary the
> fences do not cover; if the rate is unchanged, the #140 class is not #315's mechanism and the search
> should leave the FA path entirely."

That is already a pre-registered form. This file fixes the design, the arms and the reading **before**
running it, and records what the prior work already found so the new run does not repeat it.

## What the knob is (read at source, not assumed)

`ggml/src/ggml-hrx/dispatch_registration/common/dispatch-flash-attention.cpp:23`:

    // Test knob for the decode-split partial transients' alignment. Production is 4096;
    // the engine#123/#140 rig oracle used 256, a layout that exposes the divergence.
    // Defaults to 4096 so production behaviour is unchanged unless explicitly requested.

So `GGML_HRX_FA_PARTIAL_ALIGN=256` restores the layout the #140 oracle used. It is **not** a pure
alignment change: per the mechanism doc, at `align=256` the dispatcher substitutes
`common.flash_attention_f32_f16_wmma` (the **prefill-shaped** kernel) for the decode-split kernel, and
`GGML_HRX_LOG_DISPATCH=1` shows the substitution. So the intervention has two co-varying effects —
alignment AND kernel selection — and any result must be read with that in mind.

## What the prior work already established (do not re-derive)

From `docs/315-mechanism.md`, the `align=256` + compaction recipe on **Qwen3-0.6B**:

    default align, quiet            1 run    clean 180/180
    default align + compaction      2 runs   clean 180/180 both
    align=256, quiet                1 run    clean 180/180
    align=256 + compaction          5 runs   1 faulty (95 distinct of 180), 4 clean (180/180)

and the honest correction recorded there: five runs do **not** support calling that a reproducer — it
fired once, and the one faulty run coincided with foreign box load (8-12) while the four clean repeats ran
quieter (4-7). So: **a preemption-sensitive wrong-output event is possible at the pin at align=256 under
contention, and it is not yet controllable.** All Face B (`nan=0`); nothing in that recipe produced
Face A.

Also established: `GGML_HRX_DISABLE_DISPATCH=kquant.mul_mat_add.decode_f32` did not stop it (68 distinct
of 180), so the matmul-add fusion is not the carrier; what is left is the FA substitution.

## Design — deliberately different from the prior recipe, to make it controllable

The prior recipe's weakness was that contention was **borrowed** from a foreign session. This design
creates it with the goal's own rig, which is what makes the rate measurable rather than incidental.

    rig     : 315-disc.sh style, but with the GLM eviction rig's *contention* and the small model
    model   : Qwen3-0.6B (378 MB — no regime needed, per §57)
    cells   : e (evictonly forcer) vs q (quiet), the same forcer the GLM arms use
    arms    : A = align default (4096), B = align 256; each in both cells
    detector: the cheap probe's identity check (N identical greedy requests in one process) PLUS the
              NaN-logits guard, so both faces are counted

The point of using the **same forcer** as the GLM arms is that it makes the contention deliberate and
equal across cells, removing the load-confounded reading the prior attempt was stuck with.

## Reading, fixed BEFORE the run

    align=256 faults at a materially higher rate than align=4096 in cell e, and align=256 is (near) clean in cell q
        -> the #140 layout/dispatch class IS implicated, and the surviving instance is at a boundary the
           fences do not cover. Follow the FA substitution (prefill-shaped kernel) as the carrier.
    align=256 ~= align=4096 (both fault, or both clean, in cell e)
        -> the #140 class is NOT #315's mechanism. Per §57's own instruction, **leave the FA path
           entirely** and do not spend further budget there.
    align=256 clean in cell e while 4096 faults
        -> contradicted in direction; report as such (it would suggest the layout knob is protective,
           which would be informative about the substitution).

## Why this is worth doing at all, given four families are already closed

It is the **only** discriminator on the record that is cheap (a 378 MB model, no regime), has never been
run against the fault with a controlled forcer, and can retire a whole model family — the FA/flash-attention
path — either way. A null here is as valuable as a hit, because §57 pre-commits to leaving the FA path if
the rate does not move.

## Why NOT run yet (recorded rather than silently deferred)

`~/npu`/GPU access is currently held by a **peer agent's campaign** (`315text1`, a full eviction-rig run
with 3 neighbours at GTT 79G, still in flight). GTT is at ~100 of 120 GiB, so:

- the probe would add GTT into roughly 20 GiB of headroom, risking an allocation failure for the peer's
  regime;
- it would perturb the very memory pressure the peer is measuring;
- the goal's constraints require taking the box lock and **not disturbing other workloads on the shared
  box**, and the peer's driver does not take the lock, so the lock cannot protect them.

So this is deferred until the peer's campaign ends, exactly as arm B of the graph-reorder A/B was. The
pre-registration exists now so the run is ready and the reading is fixed in advance.

## Cost

~10-20 minutes of GPU (one 378 MB model, two cells, both alignments, ~2 x N requests each). This fits the
remaining budget where a full GLM campaign would not.
