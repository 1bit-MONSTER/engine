<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The onset layer is a property of the configuration, not of the defect — layer 41 with probes, layer 0 without

**This qualifies, and partly withdraws, the layer-41 claim** made in
`315-layer41-site-by-graph-arithmetic.md` and the status record. Both measurements are correct; the reading
that "the site is layer 41" is not, because a second working configuration puts the onset at layer 0.

## The two configurations, and their onsets

| config | instrumentation | first NaN-bearing tensor | onset |
|---|---|---|---|
| `315nan`, `315fbar` | `tail:16` + `COPY_PROBE` + `SENTINEL` + `LOGITS_FP`, `NANONLY` | `node_2948` = layer 41's unnamed post-attention ADD, `nan=2048` | **layer 41** |
| `315size` | 15-name `l_out` list, `NANONLY`, **no probes** | `l_out-0`, `nan=1984` (r1e/r2e) or `1790` (r2q) | **layer 0** |

`315size-r2e`, in call order (each `l_out` is the last node of its layer, so each has its own call):

    call=40377 name=l_out-0  op=ADD bytes=1105920 nan=1984  of 276480
    call=40379 name=l_out-1  op=ADD bytes=1105920 nan=14272 of 276480
    call=40381 name=l_out-2  op=ADD bytes=1105920 nan=14336 of 276480
    call=40383 name=l_out-3  op=ADD bytes=1105920 nan=20480 of 276480
    call=40385 name=l_out-4  op=ADD bytes=1105920 nan=276480 of 276480   <- saturated
    ...
    call=40469 name=l_out-46 op=ADD bytes=8192    nan=2048  of 2048

The same shape of curve appears in `315size-r1e` (`l_out-0 nan=1984`) and `315size-r2q`
(`l_out-0 nan=1790`). **Layer 0 already carries NaN in every Face-A arm of that configuration.**

## Why the layer-41 reading is not simply wrong

In `315nan-p2l` the **first** printed line is `call=152939`. `GGML_HRX_TENSOR_FP_CALLS` defaults to 1,000,000,
so every call is fingerprinted, and `NANONLY=1` suppresses only NaN-free lines. A first printed call at
152,939 therefore means **~152,938 earlier graphs produced no NaN line at all** — layers 0–40 were watched and
were **clean** in that arm. Layer 0 is watchable: `norm-0` printed in `315nan-p12l`.

So both statements are true of their own data: **with the probe-heavy instrumentation the first corruption is
at layer 41; with the name-list instrumentation it is at layer 0.**

## What this means, stated plainly

- **"Layer 41" is not a stable site.** It is where this defect surfaces *under the probe-heavy configuration*.
  A claim that the site is layer 41 is a claim about the instrument as much as about the fault.
- The cleanest reading consistent with both: the corruption is a **region-level event whose observable onset
  moves with memory traffic**, which is what the probe set changes — `SENTINEL` fills the 619,520-byte logits
  allocation every arm, `COPY_PROBE` doubles the host reads, `LOGITS_FP` reads the head. That is the same
  "instrumentation is an inducer, and the dose selects the presentation" note already on the record, now
  showing up as a **change of onset layer**, not merely of face.
- It also means the row-size invariant measured on `315nan` (exactly `nan=2048`, 18 of 19 arms) is an
  invariant **of that configuration**. The `315size` onset counts are *not* one row: `1984` and `1790` are not
  `2048`, though `1984 = 2048 − 64` is suspiciously close and appears in the other configuration too.

## Caveats on using `315size`'s arms

State them, because they weaken the comparison:

- `315size` ran on a **thermally saturated** box: `thermal_pauses` up to **29**, `tctl` 85–95 °C,
  `stopped_frac` up to **0.82** in the CPU cell, and gate waits of 10–50 minutes. Its arms are in-regime by
  `gtt_start` (97–98G) but the composition is far from the quiet baseline the gate assumes.
- `315size` **also faulted in the quiet cell** (round 2 arm q, `nan=1`, `side=0/0`), so its control cell is not
  clean and its cells are not a clean contrast.
- Its arms were **not** run by me; they are the peer session's, read here.

## What must change in the record

- The status record's criterion-1 row must not say the site is layer 41 without the configuration qualifier.
- `315-layer41-site-by-graph-arithmetic.md` stands as a description of that node and its arithmetic, but its
  claim to be *the* site is narrowed to *the site under the probe-heavy config*.
- The row-size result is an invariant of the probe-heavy config.
- **The discriminating experiment this implies:** run the *same* fingerprint window on both configurations in
  the *same* session, at the same regime, and find whether the onset layer moves with the probe set. That is
  a config-only test and it settles whether "layer 41" is instrument or defect.
