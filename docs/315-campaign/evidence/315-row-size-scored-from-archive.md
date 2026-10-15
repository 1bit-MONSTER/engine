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

# Scoring the row-size prediction against the archive (and why the node read is the reliable one)

Read out of the archived `315nan` / `315fbar` logs while `315row` runs. **No GPU time.** This scores
`315-row-size-test-preregistration.md` against arms that already existed, using a **stronger readout** than
that document fixed — which is stated plainly rather than passed off as the pre-registered test.

## The measurement

`node_2948` = layer 41's unnamed post-attention residual ADD (derivation:
`315-layer41-site-by-graph-arithmetic.md`). Across **19 arms** it reads:

| T | arms | `node_2948` |
|---|---|---|
| 39 | `315fbar-p1c`, `p4c` | 2048 / 79872 |
| 87 | `315nan-p2l`, `p3l`, `p4l`, `p7l`, `p8l` | 2048 / 178176 |
| 98 | `315nan-p15l` | 2048 / 200704 |
| 135 | `315nan-p5l` | 2048 / 276480 |
| 140 | `315nan-p13l` | 2048 / 286720 |
| 172 | `315nan-p10l`, `p1l`, `p6l`, `p9l`, `315fbar-p2c` | 2048 / 352256 |
| 306 | `315nan-p11l` | 2048 / 626688 |
| 342 | `315nan-p17l` | 2048 / 700416 |
| 348 | `315fbar-p2l` | 2048 / 712704 |
| 512/356 | `315nan-p12l` (layer-1 onset) | 1048576 / 1048576 |

**18 of 19 arms: `nan = 2048`, exactly one `n_embd`-wide row, `2048 × 4 = 8192 bytes`** — against extents
spanning 79,872 to 712,704 floats and nine different sequence lengths. The pre-registered band was
7,936–8,448 B (±3 %). Every one of the 18 lands on **8,192 exactly**, at the band's centre.

## Fixed size: confirmed. Position: split, and the split is the finding

The pre-registration asked for "one row's worth of bytes (~8 KiB) at a position that varies between
episodes" and fixed three branches. The archive forces a fourth reading, because **"position" was
under-specified in that document** and the data separates two things it ran together:

| what varies | verdict | evidence |
|---|---|---|
| **size** | **fixed — confirmed** | 2048 × 4 B in 18 of 19 arms, nine sequence lengths |
| **layer** | **fixed — the moving-position half is falsified at layer granularity** | always layer 41 (`node_2948`) |
| **row within the tensor** | **varies** | same T (87), same count (2048), but **five different content hashes** across `p2l p3l p4l p7l p8l`; and the ext-fp arms read `row=0, 1, 0` |

So the honest statement of the prediction is **"one row, always at layer 41, in a row that moves"** — the
layer is as fixed as the size, and only the row moves. That is a *different* claim from the document's, and
it is the one the data supports; it is recorded as a correction to the prediction rather than as a
confirmation of it.

## Why the node read, and not `l_out-<N>`

The pre-registered readout was the earliest `l_out-<N>` line. That readout is **not reliable**, and this is
measurable rather than asserted. Grouping each arm's `l_out-N` lines by content hash:

| arm | coincident `l_out-N` reads |
|---|---|
| `315fbar-p1c` | `l_out-43 ≡ l_out-45`, `l_out-42 ≡ l_out-44` |
| `315fbar-p2c` | `l_out-43 ≡ l_out-45` |
| `315fbar-p4c` | `l_out-43 ≡ l_out-45`, `l_out-42 ≡ l_out-44` |
| `315nan-p10l` | `l_out-43 ≡ l_out-45` |
| `315nan-p2l` | `l_out-43 ≡ l_out-45` |

**Layers two apart return identical content, in every arm that shows it.** That is the arena slot reuse the
record already identified (`315-idprobe-readpath-refuted.md`: the fingerprint reads after the whole graph has
run, so a slot whose tensor's lifetime ended is reassigned) — now measured with a **2-layer stride**.

**Scope correction (this document first stated the stride too broadly).** The coincidence is only evidence of
aliasing when the reads are **not saturated**. Fully-NaN tensors hash identically whatever layer they are from,
so an all-NaN `l_out-42` and an all-NaN `l_out-45` matching proves nothing. The *sound* instances are the
`315fbar` cold-cell arms at **25 % NaN** (`nan=19968 of 79872`), where two different layers returning
byte-identical largely-finite 319 KB outputs cannot be genuine. In `315lout4-r1e` a stride-**3** family
(`l_out-15 l_out-18 l_out-21 l_out-24 l_out-27 l_out-30 l_out-33 l_out-36 l_out-39 l_out-42 l_out-45`) shares
one hash — but every member of it reads `nan=200704 of 200704`, i.e. **fully saturated**, so that family is
**explained by saturation and is not evidence of aliasing**. The safe statement is therefore narrower than the
first version of this file made it: **coincident hashes are aliasing only when the reads are unsaturated, and
the only sound instances found are the stride-2 pairs in the `315fbar` cold cell.** The stride may differ by
configuration; nothing here establishes one universal value.

The `node_NNNN` reads do **not** share this defect: across the same arms, the only coincident pair is
`node_3304 ≡ node_3305` (a distinct, adjacent family), and **`node_2948` never coincides with anything**. That
is why the finding above is stated on the node read. Note the asymmetry is partly structural rather than
lucky: an unnamed node is read at one position in its chunk's window, so it appears once per call, whereas the
`l_out` names in a `tail:16` window are the *same* arena region reached from different layers — which is
exactly the condition under which aliasing shows up. The practical rule either way: **two readings with the
same hash are one reading.**

## What this does and does not do

- **Does:** put the fixed-size half of the prediction on 18 in-regime Face-A arms instead of four, and
  correct the prediction's own wording on "position".
- **Does not** replace `315row`: that run is the formally pre-registered test with a pre-declared readout,
  and its verdicts stand independently. This document is a **readout of the archive**, and it says so.
- **Does not** establish why one row, why layer 41, which buffer, or any rate. Arms were selected for having
  faulted; there is no denominator here.
- **Does not** claim the differing hashes prove the row moved: the hash covers finite values too, which vary
  between arms for other reasons. The unaliased `row=0/1` reads from `315fbar` are the direct evidence; the
  hashes are consistent with it.

## Reproducing

    for d in ~/wt/rt-det/315nan-p*l*; do
      grep -ahE "name=node_2948 " "$d/server.log" | grep -aoE "nan=[0-9]+ of [0-9]+"
    done
    # aliasing check
    grep -ahE "^\[hrx-tensor-fp\].*name=l_out-[0-9]+ " "$d/server.log" \
      | sed -E 's/.*name=(l_out-[0-9]+).*hash=([0-9a-f]+).*/\2 \1/' | sort | uniq -c -w16
