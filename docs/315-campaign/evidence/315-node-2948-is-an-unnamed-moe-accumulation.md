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

`node_2948` is therefore **the 2,948th node in graph-visit order**, assigned to a node that llama.cpp never
gave a name. It is a **positional index**, not a semantic identifier. Two consequences follow immediately:

- It **shifts with graph size**, i.e. with `n_tokens`, batch shape and model depth. The same physical tensor
  is `node_2948` at one sequence length and something else at another. **A fix cannot be aimed at it by
  name.**
- It tells us the node is *unnamed*, which is itself informative: every named stage in the layer-41 chain
  (`ffn_moe_out-41`, `ffn_swiglu-41`, `ffn_shexp-41`, `ffn_out-41`, `l_out-41`) has an explicit name, so the
  first corrupted tensor is an **intermediate that llama.cpp does not name**.

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
- It is **not name-addressable**, so any fix must key on **op + shape + layer** (or on the buffer/allocator),
  not on `node_…`. A patch that matched the string `node_2948` would silently stop matching when the
  sequence length changed.
- The natural next discriminator is cheap and named: **add the per-expert views to the watched set** so the
  `ADD`'s inputs are fingerprint-printed. If a view already carries NaN, the adds are downstream and the
  origin is earlier; if every view is clean and the `ADD` result is NaN, the add/its output buffer is
  implicated. That is a **watch-list change, not a new build** — the same instrument with a different
  `GGML_HRX_TENSOR_FP` value.

## Reproducing

    # the naming rule
    sed -n '7201,7203p' third_party/llama.cpp/ggml/src/ggml.c
    # the unnamed accumulation adds and the one named result
    grep -n 'moe_out = ggml_add\|cb(moe_out, "ffn_moe_out"\|cur_experts\[i\] = ggml_view_2d' \
        third_party/llama.cpp/src/llama-graph.cpp

Pin: `third_party/llama.cpp = e44c9d01a4d5110cecaf45472a717f46cd1abbf9` (matches the arms' `version: 10613
(e44c9d01a)`). Read on strixhalo at `~/wt/pin-fork-hrx/third_party/llama.cpp/` during the `315row` run; it
uses no GPU and disturbs no measurement.
