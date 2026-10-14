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

# The layer-41 site, identified by graph arithmetic: an unnamed post-attention residual ADD, one row NaN in 18 of 19 arms

**Status:** read entirely out of the archived `315nan` / `315fbar` server logs while the pre-registered `315row`
replication runs. **No GPU time, no new arm.** This consolidates and **corrects** `315-first-row-site-is-fixed-layer-41.md`
and the op-identification in `315-node-2948-is-an-unnamed-moe-accumulation.md`.

## 1. The unnamed node is a graph index, and it is stable across T

    ggml.c:7202-7203
        if (strlen(node->name) == 0) { ggml_format_name(node, "node_%d", cgraph->n_nodes); }

`node_N` = the node's position in the **full** forward graph, minted at build time for a node llama.cpp never
named. The graph's *node count* is per-operation, not per-token, so the index is **stable across sequence
length** — demonstrated, not assumed: `node_2948` is the first corrupted node at `T = 39, 87, 98, 135, 140,
172, 306, 342, 348`.

## 2. The graph is 71 nodes per layer, with exactly two unnamed nodes per layer

Every `node_N ` seen anywhere in the archive reduces to one of two residues modulo 71:

| residue mod 71 | op | count | role |
|---|---|---|---|
| **37** | `ADD` | 47 | one per layer |
| **3** | `FLASH_ATTN_EXT` | 47 | one per layer |

47 layers, one unnamed attention op and one unnamed residual add each. This is a **pure arithmetic**
derivation from the name set, and it independently reproduces "the model has 47 layers" without reading the
model. Per-layer block `L` therefore starts at `71·L`, with the attention op at `71L+3` and the add at
`71L+37`.

**Consequently:**

    node_2948 = 71×41 + 37   -> LAYER 41's unnamed ADD
    node_2914 = 71×41 +  3   -> LAYER 41's unnamed FLASH_ATTN_EXT

The layer-41 site is now fixed by **two independent methods**: the layer-`N` names in the same window
(`ffn_shexp-41`, `l_out-41`, …) and this arithmetic. **Correction:** the earlier document guessed this ADD was
a MoE expert-accumulation add (`llama-graph.cpp:2152`). That guess is **withdrawn** — the per-layer op pair is
attention + residual add, and 47 of each accounts for the model exactly. The accumulation adds, if they are
unnamed at all, are not the residue-37 family.

## 3. The measurement: one 2048-wide row, at layer 41's add, in 18 of 19 arms

`node_2948` across every arm that has a line for it:

| arm | `nan` / extent | T = extent/2048 |
|---|---|---|
| `315nan-p2l`,`p3l`,`p4l`,`p7l`,`p8l` | 2048 / 178176 | 87 |
| `315nan-p10l`,`p1l`,`p6l`,`p9l` | 2048 / 352256 | 172 |
| `315nan-p11l` | 2048 / 626688 | 306 |
| `315nan-p17l` | 2048 / 700416 | 342 |
| `315nan-p5l` | 2048 / 276480 | 135 |
| `315nan-p13l` | 2048 / 286720 | 140 |
| `315nan-p15l` | 2048 / 200704 | 98 |
| `315fbar-p1c`,`p4c` | 2048 / 79872 | 39 |
| `315fbar-p2c` | 2048 / 352256 | 172 |
| `315fbar-p2l` | 2048 / 712704 | 348 |
| **`315nan-p12l`** | **1048576 / 1048576** and **720832 / 729088** | 512 / 356 |

**18 arms, nine different sequence lengths, `nan = 2048` every single time** — one hidden-state row exactly,
against extents from 79,872 to 712,704 floats. This is the strongest quantitative result in the campaign and it
makes the "one row" claim a measured invariant, not a reading.

**`315nan-p12l` is the one exception**, and it is the exception that proves the site: it is the arm whose NaN
is already present at layer 1 (`node_74`, `node_108`, `node_145`, `node_179` … — layers 1, 2, 3 …, and its first
line is `norm-0 nan=55 of 20480`). There the whole tensor is NaN because the corruption predates layer 41. So
**the two populations are distinguishable by their first-corrupted node's layer**, and only one of them is a
layer-41 onset.

## 4. The attention op is clean in the arms where the add is NaN

Layer 41's attention op `node_2914` carries NaN in **exactly one** arm — `315nan-p12l`, the layer-1 onset —
where it is `nan=4780672 of 5242880` (91 %). In the 18 one-row arms **it does not appear**, and with
`GGML_HRX_TENSOR_FP_NANONLY=1` absence means no NaN in a node the window covered. Layer **42**'s attention op
`node_2985` *is* NaN in 15 of those arms, which is what propagation from a corrupt layer-41 residual predicts.

**Hypothesis this supports, stated so it can be wrong:** in the one-row population the corruption is introduced
**at or immediately before layer 41's post-attention residual add**, and is **not** produced by layer 41's
attention op; the attention op is a *victim* from layer 42 onward. Falsified by finding `node_2914` NaN-bearing
in a one-row arm, or by finding `node_2948` clean in an arm whose layer-42 attention op is NaN.

## 5. What this does NOT establish

- **Not the buffer.** The add's two inputs are the layer-41 input residual and the attention output. Both being
  finite and the sum NaN is impossible, so *something* is wrong with a read or a buffer — but which one is not
  determined here.
- **Not that the add produced it.** The displacement caveat stands: these are host-side reads of a
  non-coherent arena, and the `nan=1984` vs `nan=2048` inconsistency already on the record (a NaN count
  cannot *fall* along a chain) shows individual lines are not all faithful. The **invariant** is the 2048
  across arms; individual counts within one arm are not.
- **Not a rate.** 19 faulting arms selected for having faulted. Nothing here is a denominator.
- **Not why 41.** Nothing explains the choice of layer.
- **Not the MoE hypothesis**, which is withdrawn rather than adjusted: the residue-37 family is attention
  residual, and 47+47 accounts for the model with nothing left over.

## Reproducing

    # the two residues, and the op attached to each
    grep -ahoE "name=node_[0-9]+ op=[A-Z_]+" ~/wt/rt-det/*/server.log | sort -u | awk '{print $2}' | sort | uniq -c
    # the one-row invariant
    for d in ~/wt/rt-det/*/; do echo "[$d]"; grep -ahE "name=node_2948 " "$d/server.log" | grep -aoE "nan=[0-9]+ of [0-9]+"; done

Pin `e44c9d01a`; naming rule at `third_party/llama.cpp/ggml/src/ggml.c:7202-7203`.
