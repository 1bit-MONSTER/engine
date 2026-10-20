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

# The MoE-id instrument has never observed expert ids — third instrument-integrity defect, and it retires the fp/sync negative's subject

## What was being tested

`ffn_moe_argsort` / `ffn_moe_topk` are the MoE routing tensors. The objective's own non-regression control is the
**`MUL_MAT_ID` op-test**, and `MUL_MAT_ID` is the op that gathers expert weight rows *by* those ids. So a
corrupted or out-of-range expert id is a first-class mechanism candidate: an id outside `[0, n_expert)` makes
`MUL_MAT_ID` read the wrong weight row, which is exactly how finite-but-wrong or NaN MoE output arises.

The record carries a purpose-built instrument for it — `[hrx-ids-fp]` (unsynchronized host `memcpy`) and
`[hrx-ids-sync]` (`hrx_synchronous_d2h`), each reporting `min/max/oor/dup_pairs/first8`. `oor` is literally
"ids out of range". This should have been the decisive test.

## The instrument reads something that is not ids

Grouping all 3,412,028 `hrx-ids-fp` records by the width the probe checks against:

| `ne0` | records | names | sample `first8` |
|---|---|---|---|
| 64 | 1,706,014 | `ffn_moe_argsort-N` | `1040594397, -1122889188, -1131885677, 1021126885` |
| 4 | 1,706,014 | `ffn_moe_topk-N` | `1040594397, -1122889188, -1131885677, 1021126885` |

**Those are not indices — they are float bit patterns.** `1040594397 = 0x3E0635DD` is `f32 ≈ 0.131`;
`-1122889188` is `f32 ≈ -0.034`. An I32 tensor holding floats. And `min < 0` in **1,276,798** of the `ne0=4`
records, i.e. the majority of readings cannot be an expert index at all.

## And it is not the fault — it is deterministic and identical in clean and faulting arms

    fault class   arms   records      "insane" (not in [0, ne0))   arms with zero insane
    FACE_B           4      24,776            12,840  (51.8 %)            0
    clean           23   3,387,252         2,864,418  (84.6 %)            0

**The clean arms are *worse* than the Face-B arms.** And the values are byte-identical across fault classes:

    315fp2-p1c  (clean)  ffn_moe_argsort-46 min=-1162420187 max=1066077695 ['1040594397', …]
    315idsy-p1c (clean)  ffn_moe_argsort-46 min=-1162420187 max=1066077695 ['1040594397', …]
    315ids2-r1c (clean)  ffn_moe_argsort-1  min=-1229665280 max=1051938418 ['996680142', …]
    315ids2-r1e (FACE_B) ffn_moe_argsort-1  min=-1229665280 max=1051938418 ['996680142', …]

with a **constant "sane" count of exactly 33,274 in every full-length arm**. Deterministic, fault-independent,
and reproducible to the value.

**Conclusion: the probe is reading a buffer that holds float data at the moment it reads, in every arm.** The
most likely reading is that these I32 routing tensors are scratch space whose storage is shared with a float
tensor in the captured graph, so what the probe sees is the *other* tensor's contents — an arena/aliasing
artefact of the capture, identical on every replay because the graph is identical.

## What this retires

1. **`oor` and `dup_pairs`** — already withdrawn as artefacts of a wrong bound; now definitively meaningless,
   since the underlying values are not ids.
2. **"clean permutation" reasoning on these tensors** — the record has no observation of a permutation.
3. **The subject of the fp-vs-sync negative.** The 253,598 paired readings with zero divergence are still a
   valid statement *about the two read paths* (they agree, on this data), but they are **not** evidence about
   routing integrity, because the data is not routing data. The negative stands; what it is about changes.
4. **Expert-id validity has never been measured on this record.** That is the gap, and it is the one the
   `MUL_MAT_ID` control was presumably meant to cover.

## What survives, and what the record actually shows

The fault signature is untouched by all of this, and it is now the strongest thing on the record:

- the onset is in the **FFN/MoE path**, never an attention-only tensor first — `ffn_shexp`, `ffn_swiglu`,
  `ffn_out`, `ffn_moe_gate`, `ffn_moe_probs`, `ffn_moe_up`, `ffn_moe_down`, `ffn_moe_weighted`;
- in the `fbar` arms, **exactly one row of thirty-nine** — `node_2948` `nan=2048` of `79872`, `ffn_swiglu-41`
  `nan=2048` of `59904` — and it is **row 0** (`row==0` 92.0 %, `first==0` 86.9 %);
- the **router can collapse entirely**: `ffn_moe_probs-1` `nan=640` of `640`, `ffn_moe_weights-1` `nan=40`
  of `40`, `ffn_moe_up-4` `nan=61440` of `61440` — whole tensors;
- the onset layer is **not fixed** (41, 0, 1, 4 across arms), consistent with the read-order dependence already
  recorded;
- Face B carries finite logits and is about half benign prose; Face A is `HSA_STATUS_ERROR` in 12 arms.

**Three instrument-integrity defects have now been found in this record: the disjoint instrument trees, the
`fp_nan_only` gate that hides clean tensors, and now the id probe reading non-id data.** Each one silently
removed the observation that would have decided a question. **The next step is therefore not another archive
query — it is a run whose instruments are verified to read what they claim**, and the two that matter are (a)
`GGML_HRX_TENSOR_FP_NANONLY=off` to see layer 41's FFN *inputs*, and (b) an id check against the real
`n_expert`, not `ne0`.

**No mechanism is claimed. What is claimed is that the record's routing evidence is void, and that two of the
three questions it was meant to answer can still be answered by one arm each.**
