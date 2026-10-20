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

# Pre-registration: the row INDEX at the fixed site (layer 41), via the extended fingerprint

Written **before** the run. Companion to `315-first-row-site-is-fixed-layer-41.md`, which established from
the archived `315nan` logs that the first corrupted row is fixed at **layer 41** and is **`nan=2048`
exactly** — one hidden-state row — but that the lines predate the extended fingerprint, so **which row** is
still unmeasured.

## The question

`315-first-row-site-is-fixed-layer-41.md` leaves three things unmeasured. This run attacks the first, which
is the one that can name a mechanism:

1. **Which row** (`first=`/`row=`)? If it is a *fixed* row index across arms, the corruption is a fixed
   offset in a fixed buffer — a width/offset bug. If it is a *moving* row index, it is per-token state —
   a KV slot, or the last token's row.
2. Which buffer (`node_2948` is an `ADD` shaped like the layer-41 residual output — consistent, unproven).
3. Why layer 41 of 47.

## The run

    BIN=~/wt/315fp2-build/bin          # the build whose arms carry ne0=/first=/row=
    EXTRA_ENV="GGML_HRX_SENTINEL=1 GGML_HRX_LOGITS_FP=100000 \
               GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1"
    ~/wt/315-disc.sh 8 315rowidx

`315fp2` is named because its logs (`315fp2-p2l`) are the ones on this record that contain the extended
fields; if that build does not carry them, the run is **aborted and re-pre-registered against the correct
build** rather than reinterpreted. The readout below is defined on the fields, not on the build.

## Readout, fixed in advance

For each in-regime Face-A arm, take the first NaN-bearing line with `row=` present and record
`(name, ne0, first, row, T = n/ne0)`.

| observation across ≥5 in-regime Face-A arms | verdict |
|---|---|
| the first corrupted node is layer 41 (or its `ffn_swiglu-41`) **and** `row` is the **same** in every arm | **fixed site, fixed row** — a fixed offset in a fixed buffer; the KV-slot and per-token readings are both out |
| layer 41 fixed but `row` **varies** between arms | **fixed site, moving row** — per-token state at a fixed layer; `row` vs the sequence length decides last-token vs slot-index |
| the first corrupted node's layer **varies** across arms | the fixed-site finding does not replicate in a fresh run — reported as a **contradiction** of `315-first-row-site-is-fixed-layer-41.md`, not as a null |
| `ne0/first/row` absent from every faulting arm's lines | **INVALID for this question** — the build does not carry the extension; reported, never as a null |

A second, free check on the same arms: print `row` for **every** NaN-bearing named node in the chain. If the
row is constant from the first corrupted node to `result_output`, the propagation carries one row and a
per-token reading is strongly implied; if it changes, the chain mixes rows and only the first is
interpretable.

## Confounds, declared

- **Instrument inflation.** `tail:16` raises the fault rate (~94 % in `315nan` against ~5 % ambient). No rate
  is quoted; the yield is used, not measured.
- **The window's head is displacement-exposed** — but this run's target, layer 41, is *not* the window head
  in the arms examined: `node_2948` is followed in the same chunk by five further named nodes, and
  `l_out-41` (the window's last line for that chunk) carried the same NaN. So the reading is not taken from
  the least faithful line, unlike the row-size test's.
- **`row` is only as good as `ne0`.** If `ne0` is reported as the *innermost* extent of a non-2D view, `row`
  may index something other than a token. The `T = n/ne0` column is printed so a nonsense `T` is visible
  rather than silently trusted.
- **8 rounds is 8 eviction-cell arms.** If fewer than 5 are in-regime Face-A, the run is INVALID for this
  question (same rule as the row-size test).

## What a result would and would not buy

- A **fixed row** at a fixed layer would point at an offset/width error and would be a *mechanism*, with an
  obvious discriminating experiment (compare the row index against the buffer's known width and the
  sequence length).
- A **moving row** at a fixed layer would point at per-token state and would make the KV-slot reading the
  prime candidate — the reading `315-corruption-is-one-row-sized.md` raised and could not decide.
- Neither result names the **buffer**, and neither explains **why layer 41**. Both remain, and are stated
  here so a positive result is not overread.

No criterion changes. This is a diagnostic run; no arm in it is an acceptance arm.
