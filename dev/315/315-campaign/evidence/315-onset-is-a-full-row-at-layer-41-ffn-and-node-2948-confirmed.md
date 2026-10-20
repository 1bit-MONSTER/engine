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

# The onset is a FULL ROW at layer 41's FFN and its residual ADD (`node_2948`) — reproduced in three arms, and it independently confirms the graph arithmetic

Found by reading `[hrx-finite-fp]`, which turns out to be a **NaN-site instrument**: across all 1,687 records in
the archive, the number of tensor names that never show NaN is **zero**. It fires only on tensors that have
NaN, and it carries `call=`, so it can be sorted into a propagation order. No GPU was used.

## The onset, in three independent arms

| arm | first NaN `call` | tensors at that call | count |
|---|---|---|---|
| `315fbar-p2c` | 51764 | `ffn_shexp-41`, `ffn_swiglu-41`, **`node_2948`**, `norm-42`, `q-42` (×3) | **2048** each |
| `315fbar-p2l` | 112279 | same set | **2048** each |
| `315fbar-p4c` | 92614 | same set | **2048** each |

**`2048` is exactly one row** — `n_embd` elements, one token's hidden vector. So the fault's **onset is a whole
row going NaN**, not a granule:

    layer 41 FFN  ──► ffn_swiglu-41 / ffn_shexp-41        (2048 NaN)
                  ──► node_2948  = 71×41 + 37             (2048 NaN)  ← the post-attention residual ADD
    layer 42      ──► norm-42 → q-42 → Kcur-42 (view) → Vcur-42 → k_pe-42
    layer 43..46  ──► the FLASH_ATTN node: node_3056, 3127, 3198, 3269   (+71 each — same node, next layer)

with the wide carriers following (`ffn_moe_gate-*` at 20+ layers, `ffn_moe_weighted-*`, `node_2985`), reaching
`nan` in the millions.

**`node_2948` is the record's own derived site** — `315-layer41-site-by-graph-arithmetic.md` computed it as
layer 41's unnamed post-attention residual ADD purely from the graph's structure (`71 nodes/layer`, one unnamed
ADD at `+37`). **The `[hrx-finite-fp]` propagation order reproduces it from measurement**, by a completely
different route. That derivation is now confirmed rather than merely derived.

## The correction this forces on the 256-byte result

The 256-byte granularity (96.6 % of partial readings begin on a 64-element boundary; ≥89.5 % in every family)
was measured on **partial** readings — `1984 of 2048`, the "31 of 32 waves" observation. **But the onset here
is a full `2048` NaN, and `1984`-style readings are what the fault looks like *after* it has propagated.**
So the granularity is measured on **downstream** tensors, and may be a property of **propagation**, not of the
fault.

That would explain why all three readings of the granularity (wave64, the record's MoE-table block, a platform
cache granule) stayed unpinned: none of them has to be the fault's unit, because the measurement was taken
where the unit has already been reshaped. **The onset's unit is one row (2048 = 8 KB = 32×256 B), and that is a
different quantity from the 256-byte boundary that was scored.**

## The instrument gap that stops the causal step

The onset is **layer 41's FFN**, but the archive cannot say whether layer 41's FFN *input* was clean, because
**both** the tensor fingerprint and the finite fingerprint are gated to print only NaN-carrying tensors. In the
`fbar` arms: `tensor-fp: 113 records, 0 with nan==0, 113 with nan>0`. This is the same
`if (fp_nan_only && nan == 0) continue;` gate already recorded in
`315-faceb-has-no-capable-instrument.md` — it makes a clean-input observation impossible.

**So the one measurement that decides the causal step is missing, and its shape is known:**

> Run the same arm with `GGML_HRX_TENSOR_FP_NANONLY` **off**, so the layer-41 FFN's *inputs*
> (`ffn_norm-41`, the gate/up projections, the routed-expert outputs) are fingerprinted alongside the NaN
> outputs. If the inputs are clean and `ffn_swiglu-41` is NaN, the fault is **inside layer 41's FFN
> computation**. If the inputs already carry NaN, it is upstream, and the search moves back.

That is one arm, and the instrument already exists — only its gate needs inverting for that run.

## What is now established, and what is not

**Established (measured, archive-only, reproducible):**

- the onset is a **full 2048-element row**;
- it appears in **layer 41's FFN and its residual ADD** in three arms with the identical tensor set;
- **`node_2948`'s arithmetic derivation is independently confirmed**;
- the propagation map to layers 42–46 (`norm`, `q`, `Kcur`, `Vcur`, `k_pe`, then FLASH_ATTN, then `ffn_moe_gate`
  widely) — so the NaN spreads along the normal residual/attention path, as NaN does;
- the 256-byte granularity was scored **downstream** of a full-row onset, which is why none of its three
  readings could be pinned to a mechanism;
- `hrx-finite-fp` is a NaN-site instrument and had not been read this way before.

**Not established:** what makes that row NaN. Whether layer 41's FFN input is clean is the single missing bit,
and it needs one run with the NaN-only gate inverted. **No mechanism is claimed.**
