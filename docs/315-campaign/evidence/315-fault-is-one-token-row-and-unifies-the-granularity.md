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

# The fault is ONE TOKEN'S ROW — and that unifies every granularity result on the record

Measured over all 1,616 NaN-carrying `[hrx-tensor-fp]` readings in the archive. No GPU.

## The signature

    first == 0  (first NaN is at element index 0) : 1405 of 1616 = 86.9 %
    row   == 0  (first NaN is in row 0)           : 1487 of 1616 = 92.0 %

and the counts are a single row, not a fraction of the tensor:

| tensor | NaN | total | rows |
|---|---|---|---|
| `node_2948` (layer 41 post-attention residual ADD) | **2048** | 79872 | **39** |
| `ffn_swiglu-41` (layer 41 SwiGLU output) | **2048** | 59904 | 39 |

`79872 / 2048 = 39` exactly, and `2048` is `n_embd`. So **exactly one row of thirty-nine is NaN, and it is row 0.**

## This unifies every granularity finding on the record

The row is `2048` elements, and the record's own arithmetic was `2048 = 32 × 64` and `1984 = 31 × 64` with
`2048 − 1984 = 64`. Read against "the NaN row is row 0, starting at buffer offset 0":

- **"96.6 % of partial readings begin on a 64-element (256 B) boundary"** — the corruption lives inside **row 0**,
  which begins at offset 0, and 64 elements (256 B) is the granule its sub-blocks are cut on. The alignment was
  never a property of the fault; it is the alignment of **the row**, seen from inside.
- **"`2048 = 32×64` … 31 corrupted, one clean"** — that is **31 of the 32 granules of row 0**, not a property of
  a wave, a MoE table, or a cache sector.
- **"one row at the front of the buffer"** — the record's own earlier finding, now with the mechanism of its
  frontness: row 0 is the front, always.

**So the three competing readings of the 256-byte granularity (wave64 / the record's MoE-table block / a platform
cache granule) were all attempts to explain a unit that is not the fault's unit.** The fault's unit is **one
token's row**; 256 B is how that row is subdivided when a downstream tensor is partially NaN. That is why all
three stayed unpinned, and it is why the granularity survived every per-family re-cut — every family shows it
because every family's row 0 is cut the same way.

## Onset layer, and why the record saw two different ones

In the `fbar` arms the selector is `tail:16` (reads every node), and the first NaN call contains layer 41 and
layer 42 tensors — **not** anything earlier. With `nan`-only printing, a *clean* earlier tensor is invisible,
but a NaN one would have been recorded at an earlier call, so **layers 0–40 are clean in these arms and row 0
goes NaN at layer 41.** The record separately observed the onset at layer 0 under name-list selectors. Both are
consistent with one fault: the instrument reports **the first tensor in its read order that carries NaN**, and
that is a function of the read order, not of the fault — which is exactly the conclusion
`315-onset-is-set-by-read-pattern-not-probes.md` reached by a different route.

## What is now pinned, and the one question left

**Pinned by measurement, archive-only, reproducible:**

1. the fault is **one row** — `2048` of `79872`, `2048` of `59904`;
2. it is **row 0** (92.0 %) and begins at **element 0** (86.9 %) — the front of the buffer;
3. the onset tensor set is **layer 41's FFN and its post-attention residual ADD (`node_2948`)**, identical in
   three arms;
4. **`node_2948`'s arithmetic derivation is independently confirmed**;
5. it propagates along the residual/attention path (layer 42 `norm`/`q`/`Kcur`/`Vcur`/`k_pe`; layers 43–46
   FLASH_ATTN; then `ffn_moe_gate` widely), and grows from 1 row to millions;
6. **the 256-byte granularity is a downstream view of row 0**, not the fault's unit.

**The one question left is unchanged and sharpened:** does row 0 of layer 41's input arrive already NaN, or does
layer 41's FFN produce it? That is the `GGML_HRX_TENSOR_FP_NANONLY=off` arm — one arm, existing instrument,
gate inverted — and it is now the only measurement standing between this record and a named mechanism.

**Nothing here is a mechanism claim.** It is the strongest *shape* result on the record: one token's row, at the
front, at layer 41, and every apparent granularity finding is that row seen from downstream.
