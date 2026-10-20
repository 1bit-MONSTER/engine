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

# Section 51's candidate, verified at both ends in the IR

Section 51 named a site and a falsifier. Both halves of that claim are re-read here at the pin
(`ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops/`), because the candidate is now the survivor
and its two halves should not rest on a single reading.

## The producer: the rank sort is not a bijection on NaN

`small_rows_f32.loom`, `ggml_argsort_rows_f32` — rank(i) is computed by scanning the row:

```loom
%mine  = view.load %input_view[%row, %element] : ... -> f32
%rank  = scf.for %other = [%c0 to %cols step %c1](%count = %c0 : index) -> (index) {
    %value   = view.load %input_view[%row, %other] : ... -> f32
    %greater = scalar.cmpf ogt, %value, %mine : f32
    %less    = scalar.cmpf olt, %value, %mine : f32
    %equal   = scalar.cmpf oeq, %value, %mine : f32
    %earlier = index.cmp ult, %other, %element : index
    %tie_before = scalar.andi %equal, %earlier : i1
    %strict  = scf.select %want_descending, %greater, %less : i1
    %before  = scalar.ori %strict, %tie_before : i1
    %step    = scf.select %before, %c1, %c0 : index
    %next    = index.add %count, %step : index
}
scf.if %valid {
  %element_i32 = index.cast %element : index to i32
  view.store %element_i32, %output_view[%row, %rank] : i32, view<[%rows]x[%cols]xi32>
}
```

All three comparisons are **ordered** (`ogt`/`olt`/`oeq`). With `%mine = NaN` every one of them is
false, so `%before` is false for every `%other` and the element contributes nothing to any count. Two
NaN elements in one row therefore receive the **same** rank: `view.store ... [%row, %rank]` writes that
slot twice and leaves some slot in `[0, cols)` never written, holding whatever the buffer held before.
The rank map is a bijection only while the row is NaN-free. **Confirmed as recorded.**

## The consumer: an assumption, not a check

`moe_routing_tables.loom:64` consumes a route's expert index with an `index.assume` range, not a clamp:

```loom
%route_expert = index.assume %route_expert0 [range(%route_expert0, -1, 511)] : index
```

and `:45` bounds the expert the same way (`[lt(%expert0, %bounded_expert_count)]`). By contrast the
gather on the same ids **clamps** (`small_rows_f32.loom`, `ggml_get_rows_small_f32`: negatives to 0,
then `in_range` against the source row count). So the same stale index is harmless on one path and
unbounded on the other — the asymmetry section 51 records. **Confirmed as recorded.**

## What is still missing, and it is the only thing missing

The candidate's precondition is NaN in the argsort's **input**. Neither half above shows that, and
section 51's own caveat stands: the "probs was finite" reading came from a fingerprint
`ggml_backend_tensor_get`, a host-visible read with no ordering guarantee, so it cannot exclude NaN at
the argsort's input.

That is precisely what the recorded next probe is for: read `ffn_moe_argsort` / `ffn_moe_topk` **as i32**
(the current probe reads every tensor as f32, so an index is decoded as a denormal float and the signal
is destroyed), anchored names, and per row print the first 8 ids, the count outside `[0, n_expert)`,
and whether the row repeats. **One faulting arm with a non-permutation row names the mechanism; a clean
permutation kills this site.**
