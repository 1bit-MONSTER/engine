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
run, so a slot whose tensor's lifetime ended is reassigned) — now measured with a **2-layer stride**. A
per-layer NaN count taken from `l_out-N` is therefore **not** a per-layer measurement.

The `node_NNNN` reads do **not** share this defect: across the same arms, the only coincident pair is
`node_3304 ≡ node_3305` (a distinct, adjacent family), and **`node_2948` never coincides with anything**.
That is why the finding above is stated on the node read.

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
