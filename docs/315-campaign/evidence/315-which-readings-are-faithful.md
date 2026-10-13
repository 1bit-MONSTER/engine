<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Which readings in this campaign's record are faithful, and which are not

The arena-reuse finding (`315-idprobe-readpath-refuted.md`, correction section) is not only a defect of the id
probe. It reclassifies **existing** recorded evidence, and it explains a caveat the record had already reached
without knowing its cause.

## The rule

The fingerprint reads each named tensor **after the whole graph has executed**. A tensor whose lifetime ends
mid-graph has its slot reassigned by the allocator, so that read may return **a later node's output**. A tensor
whose lifetime reaches the **end** of the graph is not displaced, so its read is faithful.

That gives a clean split of this campaign's per-node evidence:

| tensor | lifetime | reading |
|---|---|---|
| `norm` (RMS_NORM), `result_norm` (MUL), `result_output` (MUL_MAT) | to end of graph | **faithful** |
| `ffn_moe_weights`, `ffn_moe_gate`, `ffn_moe_up`, `ffn_moe_down`, `ffn_moe_weighted`, `l_out-<layer>`, `ffn_moe_probs-<layer>` | to their layer's consumer | **NOT faithful** |
| `ffn_moe_argsort`, `ffn_moe_topk` (the route ids) | to the MoE op | **NOT faithful** — measured directly this stretch |

## What that does to the `315row-p1c` lead

`315-row-p1c-inregime-mechanism.md` records the sharpest lead in the campaign:

    call=158945 ffn_moe_weights-4   GET_ROWS   bytes=160  nan=8 of 40  ne0=1  first=32  row=32
    call=159019 ffn_moe_probs-41    UNARY      bytes=2560 nan=640 of 640 ne0=64 first=0

and reasons: `ffn_moe_weights` is `ggml_get_rows(ffn_moe_probs, ffn_moe_topk)`, its source `ffn_moe_probs-4` is
finite, so *"a gather whose output is NaN while its source is finite is either reading outside its source (an
out-of-range id) or not writing its output"* — i.e. the route ids are the suspect.

**That inference does not survive the reuse rule.** `ffn_moe_weights-4` is an intermediate tensor whose slot is
reused once layer 4's consumer has run. The NaN counted there can be a **later node's** NaN that now occupies the
slot, not a NaN produced by the gather. The same applies to `gate`, `up`, `down`, `weighted`, `l_out` and
`probs`.

So the strongest id-family lead in the record is **consistent with a read artifact**, and it is not evidence
that the gather read out of range. It is also not evidence *against* that — the reading cannot distinguish the
two. The record had already refused to use these lines as attribution (its closing caveat: *"the reason those
per-node lines are not used as attribution"*); **this is why**, arrived at independently and now stated as a
mechanism rather than a caution.

## What is left standing, and it is not nothing

The three faithful tensors say something definite, and it is the whole of this campaign's trustworthy per-node
statement about a faulting graph:

    call=159029 norm            RMS_NORM  nan=2048 of 2048   first=0
    call=159029 result_norm     MUL       nan=2016 of 2048   first=32
    call=159029 result_output   MUL_MAT   nan=143745 of 154880 first=32

`norm` is **entirely NaN**: the residual stream entering the final norm is fully NaN by the time the graph
reaches the head. The final norm's input is an end-of-graph-adjacent tensor, so this one is trustworthy.

**That is the boundary of what the existing instrument can honestly say: the residual stream is all-NaN at the
head, and which mid-graph node first produced the NaN is not attributable from these files.**

## Why this closes the lane rather than merely delaying it

Every route to a faithful mid-graph read has a cost that contaminates the measurement:

1. **read at production** (hook the node's own execution) — the fix the reuse finding implies; it introduces
   per-node synchronisation, and this campaign has already recorded that the fingerprint instrument is itself an
   **inducer** that raises the fault rate (§39, and `315-row-p1c`'s 2-of-5 arm faults). More synchronisation moves
   the system being measured.
2. **`GGML_HRX_DIAGNOSTIC_FRESH_TRANSIENT_ARENA`** — does **not** help: `transient-arena.cpp` shows it only
   allocates a *new* buffer when capacity must grow (retiring the old one into `diagnostic_retired_buffers_`).
   It does nothing about the allocator reusing slots **within** one arena, which is the actual mechanism.
3. **forcing a keep-alive edge** so the ids' slot cannot be reused — changes the graph, adds an op, and moves
   memory layout: the same class of perturbation as (1).
4. **device-side validation** (count out-of-range ids inside the kernel, read a counter afterwards) — the only
   low-perturbation faithful option, and it is a new instrument in the HRX op/command-program lane, not a
   continuation of this campaign.

(3) and (4) are the honest next steps if the id family is to be settled. Neither is a re-run, and (4) is the one
that would not disturb the timing.

## Standing state

- The **trigger** remains named, replicated and quantified (eviction raises the rate ~5% → ~95%).
- The **id / argsort family** remains neither excluded nor implicated, with its strongest recorded lead now
  downgraded to "indistinguishable from a read artifact".
- Every other in-scope family (eight) remains as recorded.
- The only remaining **faithful** statement about a faulting graph: **the residual stream is entirely NaN at the
  head**.
