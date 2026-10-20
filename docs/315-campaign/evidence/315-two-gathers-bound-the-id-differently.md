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

# The two gathers bound their id differently: one clamps, one only assumes

Section 50 reads its central observation as *"a gather that is NaN where its source is finite is either
reading out of range (a bad id) or not writing"*, and section 51 says the consumers "assert instead —
`index.assume` ranges, not checks". Both are true of **one** of the two gathers GLM dispatches, and not
of the other. The distinction decides whether section 50's first alternative is eliminated.

## `ggml_get_rows_small_f32` — clamps, so it cannot read out of range

`kernel-corpus/kernels/loom-libs/ops/small_rows_f32.loom`, body at line 162:

```loom
%id_raw          = view.load %ids_view[%batch, %slot] : ... -> i32
%id_nonnegative  = scalar.cmpi sge, %id_raw, %c0_i32 : i32
%id_safe_i32     = scf.select %id_nonnegative, %id_raw, %c0_i32 : i32   // negative -> 0
%id0             = index.cast %id_safe_i32 : i32 to index
%in_range        = index.cmp ult, %id0, %s : index                       // vs source_rows
%id              = scf.select %in_range, %id0, %c0 : index               // out of range -> 0, not s-1
%source_row      = index.add %source_base, %id : index
%value           = view.load %input_view[%source_row, %column] : ... -> f32
```

Every id is forced into `[0, source_rows)`, so the value read is always a row of the source. **A bad id
cannot make this gather read outside `probs`, and therefore cannot make it produce NaN from a finite
`probs`.** For this kernel section 50's "reading out of range (a bad id)" is eliminated.

## `ggml_get_rows_f32` — clamps negatives only; the upper bound is an *assumption*

`kernel-corpus/kernels/loom-libs/ops/get_rows_f32.loom`. The body (line 62) reads the id and clamps
**only** the sign:

```loom
%token_id_raw           = view.load %token_ids_view[%token] : ... -> i32
%token_id_nonnegative   = scalar.cmpi sge, %token_id_raw, %c0_i32 : i32
%safe_token_id_i32      = scf.select %token_id_nonnegative, %token_id_raw, %c0_i32 : i32
%token_id               = index.cast %safe_token_id_i32 : i32 to index
```

There is **no** `cmp ult` against `row_count` and no select — unlike the small gather. The bound is
enforced one level down, in the load template (line 38), and it is an assertion:

```loom
%token_id = index.assume %token [range(%token, 0, 262207)] : index
%bounded_token_id, %bounded_row_count =
    index.assume %token_id, %row_count [lt(%token_id, %row_count)] : index, index
%row_byte_base = index.scale %bounded_token_id, %row_bytes : index, offset -> offset
```

`index.assume` is a **proof obligation, not a runtime check** — the same construct whose violation the
target reports as `TARGET/059` when it can *prove* it statically. At runtime an id `>= row_count` is not
clamped: it addresses past the end of the weight row range and the read is unbounded.

## What this changes

* **Section 50's first alternative survives for this kernel.** "Reading out of range (a bad id)" is
  eliminated for whichever tensor the *clamping* kernel serves, and **not** eliminated for
  `ggml_get_rows_f32`. The record's sentence should not be read as eliminating it globally.
* **It connects section 51's producer defect to a concrete unchecked consumer.** Section 51 verified at
  source that the rank sort stops being a bijection when a row holds two NaNs, so an id can be stale or
  duplicated; section 51 then noted the consumers "assert instead". `ggml_get_rows_f32`'s
  `index.assume [lt(token_id, row_count)]` is a second such consumer, on the **weight** gather, and an
  unbounded weight read is a plausible route to Face A's `HSA_STATUS_ERROR_MEMORY_FAULT` — not only to a
  wrong value.

## What is NOT established here

**Which kernel serves `ffn_moe_weights = ggml_get_rows(ffn_moe_probs, ffn_moe_topk)` is not settled by
this reading.** Both are dispatched by GLM — the arm census has
`60 ggml_hrx dispatch: common.get_rows_small_f32 (single)` and
`60 ggml_hrx dispatch: common.get_rows.f32 (single)` — and the two registrations
(`dispatch_registration/common/dispatch-get-rows.cpp:298-320`) select on shape/type, which this note does
not resolve. `get_rows_f32` dequantizes a *quantized weight* row (`%weight_format`, the IQ4_NL/IQ3_S
tables), so it is the weight gather, while `probs` is f32 and its ids are the top-k — the shape that
would expect the clamping kernel. That expectation is an inference, not a measurement.

**Falsifier:** if `ffn_moe_weights` is served by `ggml_get_rows_small_f32`, then section 50's "bad id"
alternative is eliminated for the tensor it was raised about, and what remains of this note is the
general observation that the weight gather enforces its id bound by assumption rather than by check.

## Resolved: which kernel serves which, read from the two matchers

Both matchers are in `dispatch_registration/common/`, and the selection is **shape-dependent**, not
fixed. `match_get_rows_small` (`dispatch-small-rows.cpp:182`) states its own place in the order in its
header comment — *"Only rows the regular get_rows kernel does not take (narrower than 4 floats, or not a
multiple of 4)"* — and enforces it:

```cpp
if ((w >= 4 && w % 4 == 0) || w > 65536 || s > 65536 || b > 65536 || r > 65536 || ...) {
    return false;                       // the CLAMPING kernel refuses this shape
}
```

with the registration priorities `common.get_rows_small_f32` = 0 against `common.get_rows.f32` = 100 and
`common.get_rows.f32_next` = 150 (`dispatch-get-rows.cpp:298-320`). So the **clamping** kernel is the
exception — it takes only what the regular kernel refuses — and the **assuming** kernel is the rule.

Is a plain f32 source allowed to reach the assuming kernel? Yes. `match_get_rows_f32` requires only
`get_rows_format_for_type(weight->type, ...)`, and the supported list in
`dispatch-mul-mat-weight-format.h` **includes `case GGML_TYPE_F32`** (alongside F16, BF16 and the
quantised formats).

**That resolves this note's falsifier against the elimination.** `ffn_moe_weights =
ggml_get_rows(ffn_moe_probs, ffn_moe_topk)` has an f32 source, and for any row width the regular kernel
accepts it is served by `ggml_get_rows_f32` — the one whose bound is `index.assume [lt(token_id,
row_count)]` and which does **not** clamp an out-of-range id. Section 50's "reading out of range (a bad
id)" therefore survives for this tensor, and it chains to section 51's verified producer defect: a stale
or duplicated id out of the rank sort would be unbounded here, not clamped.

The converse also has to be stated, because it bounds the claim: for shapes the regular kernel refuses
(`w < 4`, or `w % 4 != 0`) the clamping kernel runs and the id **is** bounded. The protection is real but
shape-conditional, so a claim about this tensor must name the shape it was measured at.
