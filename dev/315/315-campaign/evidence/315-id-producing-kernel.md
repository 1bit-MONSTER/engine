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
<!-- Copyright 2026 bong-water-water-bong
     SPDX-License-Identifier: Apache-2.0

     engine#315 evidence: the kernel that produces the routed-FFN's expert ids, read against the
     in-regime arm's fingerprint (evidence/315-row-p1c-inregime-mechanism.md). Static reading plus one
     measured arm; no new arm was run for this file. -->

# The id-producing kernel: an ordered-comparison rank sort, a clamping gather, and an unchecked `assume`

`315row-p1c` left the fault sitting on the route **ids**: the route-weight gather
`ffn_moe_weights = ggml_get_rows(ffn_moe_probs, ffn_moe_topk)` was NaN for two whole tokens while
`ffn_moe_probs` was not, and the route index tensor itself was not instrumented. This file reads the three
kernels that touch those ids, in the order the graph uses them.

## 1. `ffn_moe_topk` is a *view* over a full argsort, and the argsort is a rank computation

`src/llama-graph.cpp:1930-1935`:

```c
selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
cb(selected_experts->src[0], "ffn_moe_argsort", il);
cb(selected_experts, "ffn_moe_topk", il);
```

The ids are the first `n_expert_used` columns of a full `[n_expert, n_tokens]` argsort, read through a view
whose row stride is `n_expert` - which is why the matcher derives `route_stride` from `route_ids->nb[1]`
(`dispatch-mul-mat-id-common.h:229-238`) and why `ggml_get_rows_small_f32` takes an explicit `id_stride`.

The producer is `ggml_argsort_rows_f32`, defined in
`kernel-corpus/kernels/loom-libs/ops/small_rows_f32.loom:111`. It is one thread per element, and each thread
computes its **rank** by scanning its row:

```text
%mine    = load(input_view[row, element])
%rank    = scf.for %other = [0 to %cols] (%count = 0) {
             %value   = load(input_view[row, %other])
             %greater = cmpf ogt, %value, %mine        // ORDERED comparisons
             %less    = cmpf olt, %value, %mine
             %equal   = cmpf oeq, %value, %mine
             %earlier = index.cmp ult, %other, %element
             %tie_before = andi(%equal, %earlier)      // ties broken by index -> a total order
             %strict  = select(%want_descending, %greater, %less)
             %before  = ori(%strict, %tie_before)
             yield %count + select(%before, 1, 0)
           }
scf.if %valid { view.store cast(%element, i32), %output_view[%row, %rank] }
```

For finite inputs this is a **bijection**: ranks 0…cols-1 are assigned exactly once per row, so every output
slot is written and the top-k view reads the true top-k. The store is guarded by `%valid`, so the padded tail of
the last wave cannot write.

## 2. The one input for which it is *not* a bijection: NaN

`cmpf ogt`, `olt` and `oeq` are all **ordered** comparisons, so for `%mine` or `%value` equal to NaN all three
are false: `%before` is false, no increment happens, and the tie-break cannot fire either — because the
tie-break requires `%equal`, which is also false.

Therefore, if a row contains **two or more NaNs**, those elements receive the *same* rank. Two consequences,
both silent:

* two elements store into the **same** output slot (last writer wins), and
* one slot in `[0, cols)` is **never written**, so it keeps whatever the buffer held before this graph. Every
  other candidate in this issue leaves a *value* wrong; this one leaves an *index* stale.

That is exactly the shape the ids path needs: a stale `i32` in the top-k view is an arbitrary expert index.

## 3. The two consumers disagree about how much they trust that index

* **The gather clamps.** `ggml_get_rows_small_f32` (`small_rows_f32.loom:157` onwards) reads the id, maps
  negatives to 0, and tests it against the source row count:

  ```text
  %id_safe_i32 = select(%id_raw >= 0, %id_raw, 0)
  %in_range    = index.cmp ult, %id0, %s
  %id          = scf.select %in_range, %id0, %c0
  ```

  So a stale id **cannot** make the gather read out of bounds: `ffn_moe_weights` can only ever be a copy of
  *some* row of `ffn_moe_probs`. (This is why the arm's NaN in `ffn_moe_weights` needs its own explanation -
  see section 5.)

* **The table builders assert.** `ggml_moe_build_expert_table` / `..._partition_table` consume the same ids and
  the kernels that read their descriptors use `index.assume` with a *range* rather than a check
  (`mul_mat_id_swiglu_f32_f32_wmma.loom:47-53`: `%expert` in `[0,511]`, `%route_tile_base` in `[0,2016]`,
  `%partition_row_count` in `[1,32]`). A stale or non-index id is neither clamped nor reported there; it is
  undefined behaviour by construction, one tile at a time.

So the same corrupt id is *harmless* in the gather and *unbounded* in the matmul path. That asymmetry is
testable, and it predicts something specific: a fault whose gather output is finite-but-wrong while the expert
matmuls are NaN.

## 4. Where the NaN in the argsort's *input* could come from

The bijection breaks only if the argsort's input row already contains two NaNs. Two ways that happens:

1. The router softmax output genuinely contains NaN at that layer. In `315row-p1c`, `ffn_moe_probs-4` was
   *not* NaN when the fingerprint read it - but the fingerprint reads through
   `ggml_backend_tensor_get`, i.e. a host-visible, non-coherent read *after* the graph, and on this backend
   that read has no ordering guarantee of its own (the whole visibility family). A *stale* read of `probs`
   would show the pre-softmax logits, which are finite. **So "probs was finite" is not solid evidence** and is
   recorded here as a caveat against the arm, not as a fact.
2. The argsort's input binding does not point at the tensor the fingerprint read - an aliasing/binding defect
   in a reused plan, which is the same class as the identical `ffn_moe_weights` hashes at layers 4 and 41 in
   that arm.

## 5. What must be measured next, and what would falsify this

Instrument `ffn_moe_argsort` and `ffn_moe_topk` **as integers** (`i32`, not the f32 reinterpretation the current
probe uses), and print, per row: the first 8 ids, the number of ids outside `[0, n_expert)`, and whether the
row's ids contain a duplicate (a duplicate rank is what a NaN-induced collision produces). Anchoring the name
match is a prerequisite - the current `strstr(want, node->name)` filter also matches `-4` against `-41`, so
today's coverage is layers 4 and 41 only.

* duplicate or out-of-range ids in a faulting arm -> this kernel is the site, and the fix is to make the rank
  computation total (a NaN-total comparison, or a rank derived from an argsort of keys that cannot be NaN);
* ids always a clean permutation while the matmuls are NaN -> this candidate dies, and the fault is downstream
  in the table builders or in the matmul's descriptor use;
* `probs` re-read as NaN in the same arm with a coherent read -> the NaN is upstream of the argsort and *this*
  file is only about how it is amplified.

Nothing in this file is a rate or a mechanism claim. It is the reason the next probe is written the way it is.
