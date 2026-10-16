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

# The faithfulness rule is LIFETIME, not position — and it buys a read inside layer 0

## The correction

`315-which-readings-are-faithful.md` states the rule positionally: "late in the graph is faithful; early or middle
is exposed". That is a good approximation but not the criterion. The criterion the arena actually imposes is
**lifetime**: a slot can be reassigned only once its tensor's last consumer has run, so

> **a tensor whose last consumer is the graph's final node is live until the end, and its read cannot be displaced.**

Position is only a proxy for that. The two agree for `l_out-<layer>` (produced by the chunk's last node) — which is
why the sweep worked — but they disagree for a tensor produced **early** and consumed **late**, and that disagreement
is where the useful reads are.

## What it buys: a faithful read inside layer 0

From the pin's own builder chain (`src/models/deepseek.cpp:130-176`, quoted in `315-mechanism.md` §40):

    ffn_inp-<il>  = ADD(attn_out, inpSA)      <- produced after the block's attention
    ffn_norm-<il> = RMS_NORM(ffn_inp)
    ffn_moe_out-<il> = MoE(build_moe_ffn(ffn_norm))
    ffn_shexp-<il>   = shared_expert(ffn_norm)
    ffn_out-<il>  = ADD(ffn_moe_out, ffn_shexp)
    l_out-<il>    = ADD(ffn_out, ffn_inp)     <- ffn_inp is consumed HERE, the last node

`ffn_inp-<il>` is **produced early and consumed by the chunk's final node**. Under the lifetime rule it is
**faithful** — and it is the residual stream *after the block's attention and before its FFN*.

So `ffn_inp-0` splits layer 0 exactly where it matters:

| `ffn_inp-0` | `l_out-0` | meaning |
|---|---|---|
| NaN | NaN | the corruption is present **after layer 0's attention** — i.e. in the attention branch, or already in the embedding that feeds it |
| clean | NaN | layer 0's **FFN branch introduced it** |
| clean | clean | the corruption is later than layer 0 (would contradict this session's gradient) |

`ffn_norm-<il>` is **not** in the same class: it feeds the FFN subgraph and its last consumer runs before the adds,
so it dies mid-chunk and **is** displacement-exposed. This distinction is exactly what the positional rule blurred,
and it is why §40's chain — which lists `ffn_inp-41` and `ffn_norm-41` as if they were equivalent — should be read
with `ffn_inp-41` as the trustworthy half.

## Consequence for the run already on the box

The combined §40-chain + early-layers probe is **already correctly designed for this** without having intended it:
its chain includes `ffn_inp-41` (faithful) and `ffn_norm-41` and `ffn_inp-42`/`ffn_norm-42` (not). So it will settle
§40's causality claim on the half that can be trusted, which is more than §40's own reading could do.

## The next selector

Add the layer-0 and layer-1 residual inputs to the early-layer list:

    GGML_HRX_TENSOR_FP=l_out-0,ffn_inp-0,l_out-1,ffn_inp-1,l_out-2,ffn_inp-2,l_out-3,ffn_inp-3,l_out-4,ffn_inp-4

Ten names — inside the dose that produced Face A with a clean quiet cell — of which **five are faithful by lifetime**
(`l_out-N`) **and five by lifetime-plus-position** (`ffn_inp-N`). If `ffn_inp-0` carries NaN while the embedding is
unmeasurable either way, the honest conclusion becomes: **the corruption is already present in layer 0's post-attention
residual**, and the remaining candidate set is the attention branch and the embedding — with no host-side read able
to separate those two, because the embedding dies before the chunk's end.
