<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# What `node_2948` actually is: an auto-generated index for an unnamed node, inside `build_moe_ffn`

**Status:** source read at the arm's own pin. This resolves the open item in
`315-first-row-site-is-fixed-layer-41.md` ("`node_2948` is not yet source-addressable") without the GPU, and
it changes what a fix can be aimed at.

## The name is not a name

    ggml.c:7201-7203
        if (strlen(node->name) == 0) {
            ggml_format_name(node, "node_%d", cgraph->n_nodes);
        }

`node_2948` is therefore **the 2,948th node of the forward graph**, assigned to a node that llama.cpp
never gave a name. It is a **positional index**, not a semantic identifier — but see the correction below: it
is **stable across sequence length**, and the reason is that a graph's *node count* is per-operation, not
per-token (`T` is a tensor dimension, not a node).

**Correction (this document originally overstated this).** The first version of this file claimed the index
"shifts with `n_tokens`, batch shape and model depth". The first half of that is **wrong**, and the gathered
evidence says so: the same name `node_2948` is the first corrupted node at `T = 39, 172 and 348`
(`315fbar-p4c`, `315nan-p10l`, `315fbar-p2l` — all with `ne0=2048`, so `T = of/2048`). A name that moved
with `T` could not appear at three different `T`. What the index *does* depend on is the **graph shape**:
prefill vs decode build different graphs, and so would `-fa` changes or a different ubatch split. So the
correct statement is: **stable for a fixed graph shape, not name-stable across graph shapes.**
What remains true and matters for a fix is that it is **not a semantic name** — nothing in the source says
"this is the layer-41 MoE accumulation"; the index has to be re-derived if the graph shape changes.

## Which unnamed ADD it is

The fingerprint line is `op=ADD`, `bytes=1409024` = `352256 × 4` = `2048 × 172` = `n_embd × n_tokens`. The
only unnamed ADDs of exactly that shape in the GLM MoE path are the **per-expert accumulation adds**:

    llama-graph.cpp:2149   ggml_tensor * moe_out = cur_experts[0];
    llama-graph.cpp:2152       moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);   // unnamed
    llama-graph.cpp:2162   cb(moe_out, "ffn_moe_out", il);                          // only the LAST is named

with the per-expert inputs being unnamed views:

    llama-graph.cpp:2140       cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

So `node_2948` is **consistent with** the intermediate `moe_out` of the layer-41 `build_moe_ffn` expert
accumulation — the tensors the loop creates before `cb(moe_out, "ffn_moe_out", 41)` names the final one.

## What is proven and what is only consistent

**Proven:** the naming rule (`ggml.c:7202-7203`); that the first corrupted node is unnamed and is an `ADD` of
shape `n_embd × n_tokens` at layer 41; that `build_moe_ffn` contains exactly such unnamed adds at
`llama-graph.cpp:2152`.

**NOT proven:** that `node_2948` is precisely one of those adds. Node index 2948 depends on the whole
graph-visit order, which was **not** reconstructed here; other unnamed `ADD`s of the same shape may exist on
the attention residual path. Naming it exactly needs either a graph dump at the pin or a print of the
tensor's `src[0]/src[1]`.

**Also NOT proven, and this is the important one:** that the `ADD` *produced* the NaN. In `call=51764` the
`ADD` is the first NaN-bearing line **that the `tail:16` window contains** — and its inputs
(`cur_experts[i]`, `op=VIEW`, unnamed) are **not in that window**, so their absence from the print is not
evidence that they were clean. This is the same "no host-side read of a mid-graph tensor is trustworthy"
gap the campaign already recorded, now with a concrete instance: the first-corrupted node has **unnamed
inputs that the instrument does not cover**.

## What this buys a fix

- The site is now bounded to *a region of `build_moe_ffn`*, not "somewhere in layer 41".
- The node is **unnamed**, and every named stage in the layer-41 chain (`ffn_moe_out-41`, `ffn_swiglu-41`,
  `ffn_shexp-41`, `ffn_out-41`, `l_out-41`) *does* have an explicit name — so the first corrupted tensor is an
  intermediate llama.cpp never names.
- `node_2948` is stable across `T` for a fixed graph shape (see the correction above), so it is usable as a
  handle **within** a run series that shares a graph shape — but it is not semantic, so a fix must still key
  on **op + shape + layer** (or the buffer/allocator) rather than on the string.

## The selector, read at source (why `tail:16` shows this node)

The instrument walks the **subgraph** it is computing, not the full model graph:

    ggml-hrx.cpp:661-667
        const int fp_first = tail_n > 0 && graph->n_nodes > tail_n ? graph->n_nodes - tail_n : 0;
        for (int i = fp_first; i < graph->n_nodes; ++i) {
            ggml_tensor * node = graph->nodes[i];
            ...
            if (tail_n == 0 && !hrx_fp_name_listed(want, node->name)) { continue; }

The `node_%d` name, however, was **baked at full-graph build time** (`ggml.c:7202-7203`, `cgraph->n_nodes`),
so a name minted in the full graph rides into whichever HRX subgraph computes that tensor. That is how a
`tail:16` window over a *per-layer* subgraph can print a name whose index (2,948) is far larger than the
subgraph's own node count — and it means the index is a handle into the **full** graph, not into the window.

Note also `hrx_fp_name_listed` uses an **anchored, length-checked** compare (`ggml-hrx.cpp:604`, compare at
`:613`), not
`strstr` — the comment states the earlier `strstr` filter was itself a probe defect that matched
`ffn_moe_weights-41` against `ffn_moe_weights-4`.

## Reproducing

    # the naming rule
    sed -n '7201,7203p' third_party/llama.cpp/ggml/src/ggml.c
    # the unnamed accumulation adds and the one named result
    grep -n 'moe_out = ggml_add\|cb(moe_out, "ffn_moe_out"\|cur_experts\[i\] = ggml_view_2d' \
        third_party/llama.cpp/src/llama-graph.cpp

Pin: `third_party/llama.cpp = e44c9d01a4d5110cecaf45472a717f46cd1abbf9` (matches the arms' `version: 10613
(e44c9d01a)`). Read on strixhalo at `~/wt/pin-fork-hrx/third_party/llama.cpp/` during the `315row` run; it
uses no GPU and disturbs no measurement.

---

## WITHDRAWN (same day): the MoE-accumulation identification is wrong

The op-count query settles it against this document. Across the whole archive the unnamed `node_N` nodes
comprise **exactly two ops, 47 of each**: `ADD` (residue 37 mod 71) and `FLASH_ATTN_EXT` (residue 3 mod 71) —
one of each per layer for 47 layers. `build_moe_ffn`'s accumulation adds cannot be the residue-37 family, and
if they were unnamed at all there would be more than one per layer.

So `node_2948` is **layer 41's post-attention residual add**, not a MoE accumulation add. The `op=ADD` shape
argument in this file was necessary but not sufficient, and the "consistent with, not proven" hedge was the
right place to leave it — but the identification itself is now superseded. See
`315-layer41-site-by-graph-arithmetic.md`, which also adds the one-row invariant (18 of 19 arms, `nan=2048`
exactly) and the clean-attention-op evidence.
