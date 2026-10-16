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

# The first corrupted row sits at a FIXED site (layer 41), read from the 315nan arms that already existed

**Status:** read out of the **raw server logs of the archived `315nan` arms** while the pre-registered
`315row` replication is still running. Nothing here was measured for this document; the logs are the same
campaign's `tail:16` + `SENTINEL` + `COPY_PROBE` configuration, so this is the **same instrument** the
running test uses, applied to data that has sat unread since 2026-10-10.

## Why this exists

`315-row-size-test-preregistration.md` fixed three verdict branches in advance. Its third branch is:

> `N` is the same in every arm → the moving-position half is **falsified**; the site is fixed and §40's
> reading is back.

The `315nan` logs can answer that **without any new box time**, because each faulting arm's fingerprint
stream begins at the first NaN-bearing node, and that node **is** the site. So the question "does the origin
move between arms?" was answerable the whole time.

## What the logs show

Across the **17 faulting arms** in `~/wt/rt-det/315nan-*` (guard line present), the **first NaN-bearing
`hrx-tensor-fp` line** is:

| first NaN-bearing node | arms |
|---|---|
| `node_2948 op=ADD` | **14** |
| `ffn_swiglu-41 op=GLU` | 2 |
| `norm-0 op=RMS_NORM` | 1 |

`node_2948` and `ffn_swiglu-41` are both in **layer 41's chunk** (`call=51764` in `p10l` holds
`node_2948`, `ffn_moe_out-41`, `ffn_swiglu-41`, `ffn_shexp-41`, `ffn_out-41`, `l_out-41`), so **16 of 17
arms put the origin in layer 41 or its immediately following node**; the single `norm-0` arm is the outlier.
The site does **not** move.

**And the row count is fixed too.** In all 14 `node_2948` arms the count is **`nan=2048` exactly**, against
tensor extents that differ between arms:

| arm | extent (`of`) | NaN | extent ÷ 2048 |
|---|---|---|---|
| `p10l` | 352,256 | 2,048 | 172 |
| `p11l` | 626,688 | 2,048 | 306 |
| `p15l` | 200,704 | 2,048 | 98 |
| `p17l` | 700,416 | 2,048 | 342 |
| `p2l`/`p3l`/`p4l`/`p7l`/`p8l` | 178,176 | 2,048 | 87 |

So `nan=2048` is **one hidden-state row at T = 87…342** — the row is the unit, exactly as
`315-corruption-is-one-row-sized.md` argued — but the row **is at layer 41 in every arm**, which is the
half that document left open.

## What this does to the live prediction

- It **supports** the fixed-size part: one row, `2048 × 4 = 8192` bytes, repeatedly, at varying sequence
  lengths.
- It **falsifies the moving-position part**: the *position* here is the **layer**, and the layer is 41 in
  16 of 17 arms. The earlier session's "position moves" reading came from the `tail:16` window's
  **displacement-exposed head** — the very weakness its own pre-registration declares — not from a faithful
  reading. The last node of the window (`l_out-46`) is the *most* faithful line available, and it is
  **`8192 bytes / 2048 of 2048` in 17 of 18** arms, i.e. a different (and non-diagnostic) quantity.

## What it does NOT establish

- **Not the row index.** These lines predate the extended fingerprint (`ne0=`/`first=`/`row=` are absent), so
  *which* token's row is corrupted is still unmeasured. "One row" is established; "row k" is not.
- **Not the buffer.** One row of the residual stream, one KV slot's row, and one row of some other
  row-oriented buffer are not distinguished. `node_2948` is an `ADD` with the same shape as the layer-41
  residual output, which is consistent with the residual stream and does not prove it.
- **Not a rate.** 17 faulting arms selected for having faulted; nothing here is a denominator.
- **Not why layer 41.** The model has 47 layers; why the corruption lands there rather than uniformly is
  unexplained and is the next question after the site is accepted as fixed.
- **Not a replacement for the running test.** `315row` remains the pre-registered replication and its
  verdicts stand; this document is an **additional branch** that its own third row already anticipated.

## Addendum: the ROW index, from the arms that DO carry the extended fields

The `315nan` lines above lack `ne0=`/`first=`/`row=`, so they fix the row *size* and the *site* but not
*which* row. Four archived arms in the same campaign **do** carry the fields — `315fbar-p2c`, `315fbar-p2l`,
`315fbar-p4c` (build `315fbar-build`) — and at the same node they read:

| arm | extent `of` | T = of/2048 | `nan` | `ne0` | `first` | `row` |
|---|---|---|---|---|---|---|
| `315fbar-p2c` | 352,256 | 172 | 2,048 | 2048 | **0** | **0** |
| `315fbar-p2l` | 712,704 | 348 | 2,048 | 2048 | **2048** | **1** |
| `315fbar-p4c` | 79,872 | 39 | 2,048 | 2048 | **0** | **0** |

So at the fixed site the row **size** is one row (`nan=2048`) and the row **index is 0 or 1 — never higher**,
while T is 39, 172 and 348. A per-last-token reading would sit at T−1 (tens to hundreds); a fixed-offset
reading would sit at one constant. `row ∈ {0,1}` is neither: it is a **front-of-buffer** corruption whose
extent is constant and whose index is small.

This is the second branch of `315-row-index-preregistration.md` read from existing data — **fixed site,
moving (small) row** — and it is the strongest pointer yet at the residual-stream buffer's first row(s) at
layer 41. It does **not** name the buffer (front-of-buffer is shared by the residual stream's first token
rows and by any row-oriented staging buffer), does not explain why layer 41, and rests on **three** arms in
one build; the pre-registered `315rowidx` run is what replicates it.

## Evidence

    for d in ~/wt/rt-det/315nan-*; do
      grep -ac "HRX returned NaN logits" "$d/server.log" || continue
      grep -a "hrx-tensor-fp" "$d/server.log" | head -1
    done

Raw logs: `~/wt/rt-det/315nan-p*l/server.log` on strixhalo (17 faulting arms; 16 of 17 first-NaN in layer 41).

    row-readout.py ~/wt/rt-det/315nan-p10l-GLM-4.7-Flash-Q4_K_M

prints the full per-node chain for one arm (`node_2948` first, then `ffn_moe_out-41` … `result_output` at
`nan=154880 of 154880`), which is the propagation path from one row to the whole vocabulary.

### Why only the three `315fbar` arms are used, and not the other four

Every faulting arm in the archive that carries the extended fields was read out, and the first NaN-bearing
node differs between them:

| arm | first NaN node | size | `row` | config |
|---|---|---|---|---|
| `315fbar-p2c` | `node_2948` ADD (layer 41) | 2,048 = one row | 0 | `tail:16` |
| `315fbar-p2l` | `node_2948` ADD (layer 41) | 2,048 = one row | 1 | `tail:16` |
| `315fbar-p4c` | `node_2948` ADD (layer 41) | 2,048 = one row | 0 | `tail:16` |
| `315fp2-p2l` | `ffn_out-0` MUL_MAT | 7,048 | 0 | `tail:16` |
| `315idp-p4l` | `ffn_moe_probs-1` UNARY | 640 of 640 | 0 | id probe |
| `315row-p1c` | `ffn_moe_weights-4` GET_ROWS | 8 of 40 | 32 | name list |
| `315row-p2l` | `ffn_moe_weights-4` GET_ROWS | 8 of 40 | 32 | name list |

Only the `315fbar` triple is used, because **the fingerprint only reports the nodes the arm was asked to
watch**: a run driven by a name list (`l_out-…`, `ffn_moe_weights-…`) cannot see `node_2948` at all, so its
"first NaN node" is an artifact of its name list, not of the corruption. The four other rows are therefore
**not evidence against the fixed site** and are **not pooled** with it — they are recorded here so the
comparison is not silently dropped, and so nobody later reads their differing node names as a contradiction.
`315rowidx` (pre-registered) removes this confound by using one config across all its arms.

## Open item: `node_2948` is not yet source-addressable

The engine tree here has **uninitialized submodules** (`git submodule status` shows a leading `-` on every
entry), so the `ggml-hrx` source that emits the `node_NNNN` names is not present locally and `node_2948`
could not be resolved to a graph-construction site from this worktree. The pin is
`third_party/llama.cpp = e44c9d01a4d5110cecaf45472a717f46cd1abbf9`, which matches the `version: 10613
(e44c9d01a)` the arms report, so the revision is fixed and the lookup is reproducible — it just needs the
box (or a `git submodule update --init`) rather than a guess.

Resolving it would say whether `node_2948` is a stable graph node (in which case a fix can be aimed at it
directly) or a per-compilation index (in which case the site must be re-derived each build). That
distinction is the next step after `315rowidx` confirms the site replicates, and it is recorded here rather
than guessed.
