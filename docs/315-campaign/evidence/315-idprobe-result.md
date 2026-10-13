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

# The id probe is INVALID, and my first reading of it was wrong

The probe ran for the first time on 2026-10-13 ~19:12 (4 pairs, `ffn_moe_topk/argsort/weights/probs` anchored
per layer 0..63). It produced a genuine faulting arm:

    pair 1 arm c: OK     nan=0  req=66   gtt_start=98G
    pair 1 arm l: FAULT  nan=1  req=14   gtt_start=98G     <- Face A, in-regime
    pair 2 arm l: OK     nan=0  req=66   gtt_start=98G
    pair 3 arm l: OK     nan=0  req=66   gtt_start=98G

**The readout does not answer the question, and the first version of this file read it backwards.**

## What the instrument actually saw

Both arms, over every line of the two I32 tensors the instrument reads:

| arm | argsort lines | `oor = 0` | `oor = 64` | `oor > 0` |
|---|---|---|---|---|
| p1l (**FAULT**) | 36,708 | 16,614 | 19,728 | 20,094 |
| p1c (clean) | 126,822 | 16,637 | **106,446** | 110,185 |

The clean arm carries **more** all-out-of-range lines than the faulting arm. And the decisive number:

    lines with 0 < max < ne0      ->  0        (both arms, every tensor)

**The instrument never once observed a plausible rank value.** Every line is either all-zeros
(`min=0 max=0`, `first8=[0 0 0 0 0 0 0 0]`, 16,614 and 16,637 lines) or **f32 bit patterns** re-read as i32 —
`max=1074410205`, `min=-1138111520`, `first8=[1040594397 -1122889188 ...]`, which decode to small floats in
the ±0.001..0.25 range, i.e. **probability values**.

So `oor` was never measuring route ids. `oor > 0` is baseline behaviour of these reads in **both** arms
(superficially *more* so in the clean arm, though that is a per-request count effect), and `oor = 0` is the
all-zeros population, not a clean permutation.

## Why — at source

`src/llama-graph.cpp:1932-1935` (patched tree):

    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
    cb(selected_experts->src[0], "ffn_moe_argsort", il);   // labels the INPUT to the argsort
    cb(selected_experts,         "ffn_moe_topk",   il);    // labels its OUTPUT

The name `ffn_moe_argsort` is applied to `selected_experts->src[0]` — the argsort's **input** (`selection_probs`,
f32 routing probabilities), not to the argsort's output. That is exactly what the instrument found there:
probability-shaped floats. The name is a misnomer in the graph builder, and the probe had assumed it was the
i32 rank tensor.

And the read itself cannot be trusted for either name. The instrument in `ggml-hrx.cpp` (~line 616) reads with
`ggml_backend_tensor_get`, and its own comment states the case:

> Everything the fingerprint names on the MoE path lives in a host-visible, non-coherent arena ... and
> `ggml_backend_tensor_get` on a host-visible tensor is a plain memcpy - no iface call, no synchronisation -
> so without this the per-node lines are not a snapshot of one instant and the per-node attribution is invalid.

A plain memcpy out of a **non-coherent** arena returns the **stale host copy**, not the device's tensor
contents. Every value observed — zeros (fresh allocation) and f32 patterns (the previous occupant of a reused
buffer) — is consistent with that, in both arms.

## The corrected reading

- **Neither pre-registered branch fires.** The first branch (`oor > 0` or `dup_pairs > 0` in a faulting arm)
  does not discriminate, because `oor > 0` is the baseline state of these reads in clean arms too; the second
  branch ("ids clean while the matmuls are NaN") does not fire either, because the instrument never observed
  ids at all.
- **The id-producing / argsort family is NOT excluded.** It remains **untested**, because the instrument
  cannot measure it: it names the argsort's input by mistake and reads a non-coherent arena by memcpy.
- The probe's earlier framing as a discriminator is withdrawn.

## The error in this file's first version, and its shape

The first version claimed *"`oor` is ZERO in every line of every arm → the ids are clean → the argsort
candidate dies"*. It was produced with

    grep -oE "oor=[0-9]+ dup_pairs=[0-9]+" ... | sort | uniq -c | head -5

and `head -5` **cut off exactly the dominant `oor=64` lines**, leaving a list that began with `oor=0` entries
and looked like a full distribution. This is the **third** instance of the session's standing lesson: read the
distribution, never a truncated sorted head of it. (The first two: the `md5sum`-over-streaming parse that
briefly produced a false FA exclusion, and the line-counter on newline-free JSON.)

A second, smaller error accompanied it: the file read a *proportional* difference in `dup_pairs` between arms
as a signal, when the absolute counts of the non-zero values were **identical** in both arms (16,257 × `2016`,
156 × `50400`) — the faulting arm simply died earlier, so its totals were smaller.

## What would make this testable

The two named obstacles are both fixable and both in scope (engine-side instrumentation, not driver work):

1. **Name the right tensor.** Read the tensor the MoE kernel actually consumes as ids — the `ids` operand of
   the `MUL_MAT_ID` node — instead of a name that happens to label the argsort's input.
2. **Read it coherently.** Replace the plain-memcpy `ggml_backend_tensor_get` with a synchronising
   device→host read for these tensors, so the value reflects device truth rather than the stale host copy.

Until both are done, this family stays open and the probe must not be cited as evidence either way. No
criterion changes and no arm from it is eligible.
