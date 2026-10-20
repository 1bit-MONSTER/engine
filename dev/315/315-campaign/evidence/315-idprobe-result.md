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

---

# Addendum: the instrument was reading the CORRECT tensors — and the f32 side never ran

Two things were established after the retraction above, both from source and from the run's own logs. One of
them corrects a hypothesis I was about to record.

## 1. The two names are the right tensors (so "wrong tensor" is not the defect)

`ggml.c:5365`:

    struct ggml_tensor * ggml_argsort_top_k(ctx, a, k) {
        GGML_ASSERT(a->ne[0] >= k);
        struct ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);   // i32
        result = ggml_view_4d(ctx, result, k, ...);                                  // a VIEW of it
        return result;
    }

and `ggml.h:2393` documents it as *"similar to `ggml_top_k` but implemented as `argsort` + `view`"*. At
`src/llama-graph.cpp:1932`:

    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
    cb(selected_experts->src[0], "ffn_moe_argsort", il);   // the argsort result, i32, [64, n_tokens]
    cb(selected_experts,         "ffn_moe_topk",   il);    // the view of it,  i32, [4,  n_tokens]

So `ffn_moe_argsort` is the argsort **result**, `ffn_moe_topk` is a **view of the same buffer** — which is why
their bytes agree at the same call (confirmed in the log at call=94, layer 46: `ffn_moe_topk-46`'s four values
are exactly the first four of `ffn_moe_argsort-46`). The instrument took the i32 path for both, which means it
observed `node->type == GGML_TYPE_I32` for both — the declared types are right.

**Both were the correct tensors. The defect is what came back, not what was asked for:** an i32 tensor that by
construction must hold ranks in `[0, 64)` held float-shaped values (e.g. `1040594397` ≈ 0.131, `-1122889188`
≈ -0.0326 — logit/probability-shaped) or zeros, in a faulting and in clean arms alike, never once a rank.

## 2. The f32 side of the probe never ran at all — and my staleness hypothesis was wrong

I was about to record that the plain-memcpy read of a non-coherent arena (`ggml-hrx.cpp` `buffer_get`: when
`context->base != GGML_HRX_FAKE_PTR_BASE` it memcpys and **returns**, never reaching
`hrx_synchronous_d2h`) meant the fingerprint reads stale memory — which would have been a large claim, since it
would implicate every host-side tensor read in this campaign.

**That claim is not supported.** The run's log contains **only** `[hrx-ids-fp]` lines — 73,416 of them — and
**no f32 line at all**, in the faulting arm *or* the clean arms. The reason is coverage, not staleness: the
campaign's earlier NaN localisation used **position-based** selection (`GGML_HRX_TENSOR_FP=tail:16`, which
bypasses the name filter entirely), so it never depended on the head tensors matching by name. The id probe
instead supplied names — `norm`, `result_norm`, `result_output`, `ffn_moe_weights-N`, `ffn_moe_probs-N` — and
**none of them matched a node**, so the f32 path emitted nothing.

That also means the probe's third pre-registered branch (*"`probs` re-read as NaN with a coherent read → NaN
upstream of argsort"*) has **no data**, and cannot be read either way. Note the anchored name matching added
earlier in this probe fixes `-41`-vs-`-4` over-matching but introduces the opposite failure: a name that
carries a suffix no longer matches at all. Exactness cuts both ways.

## Consequence for the logits NaN, and what the arm actually was

The faulting arm's own log is unambiguous about the scale:

    common_sampler_sample: HRX returned NaN logits: count=154880 of 154880, index range [0..154879],
                           contiguous=yes, continuing=no sentinel=0 (engine#315)

**Every one of the 154,880 logits was NaN** — not a small corruption, the whole tensor. Under the earlier
recorded reading of the head's behaviour (*"1..4096 NaNs at the head's input means small upstream corruption
amplified by the head; zero there means the head itself manufactures the NaN"*), this is the *manufactured*
case — and the probe cannot confirm or refute that, because the head tensors were never read.

## Standing state of the family

**The id-producing / argsort family remains UNTESTED.** It is not excluded, and it is not implicated. Both of
its instruments failed for different, now-understood, and separately fixable reasons:

1. the i32 read returns float-shaped values where ranks belong — correct tensor, wrong contents, unexplained;
2. the f32 read matched no node, so the probs/weights half produced nothing.

Neither pre-registered branch fires. No criterion changes and no arm from this probe is eligible.
