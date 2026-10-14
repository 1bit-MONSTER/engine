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

## Evidence

    for d in ~/wt/rt-det/315nan-*; do
      grep -ac "HRX returned NaN logits" "$d/server.log" || continue
      grep -a "hrx-tensor-fp" "$d/server.log" | head -1
    done

Raw logs: `~/wt/rt-det/315nan-p*l/server.log` on strixhalo (17 faulting arms; 16 of 17 first-NaN in layer 41).

    row-readout.py ~/wt/rt-det/315nan-p10l-GLM-4.7-Flash-Q4_K_M

prints the full per-node chain for one arm (`node_2948` first, then `ffn_moe_out-41` … `result_output` at
`nan=154880 of 154880`), which is the propagation path from one row to the whole vocabulary.
