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

## This comparison is CONFOUNDED, and that is the main caveat

The two configurations differ in **two** ways, not one:

| config | selector | probes |
|---|---|---|
| `315nan`, `315fbar` | `tail:16` (positional window) | yes |
| `315size` | 15-name `l_out` list | no |

So the onset difference **cannot be attributed to the probes**. Either variable could produce it, and the
selector is the more worrying one: if a `tail:16` window does not in fact reach layer 0 in those arms, then
"the first NaN-bearing tensor is in layer 41" is a statement about **what the window covered**, not about the
fault. My earlier argument that layers 0–40 were watched and clean rests entirely on
`GGML_HRX_TENSOR_FP_CALLS` being effectively unlimited and `NANONLY` suppressing only NaN-free lines — which
is true of the *printing* — but it says nothing about whether a **layer-0 node was inside the 16-node window of
any call**. Those are different questions and I ran them together.

The evidence does not settle it either way. In favour of early layers being covered: `node_74`,
`node_108` (layer 1's ADD), `node_145`, `node_179`, `node_250`, `node_287` and `norm-0` **do** print, in
`315nan-p12l`, so early-layer nodes are reachable by the window in at least one arm. Against: they print in
**exactly one** of 19 arms, and that one is the arm whose onset is already early — which is also what you would
see if the window normally misses them.

**Consequently the honest statement is weaker than this document's title.** What is established is: *under
the probe-heavy `tail:16` config the first NaN-bearing tensor seen is in layer 41; under the name-list config
it is layer 0.* What is **not** established is that the probe set is the reason, or that layer 41 is anything
other than the window's reach.

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

## The controlled experiment this requires (and which `315cfg` is)

Both conditions must share the **selector** so that the same tensors are observed:

| condition | `EXTRA_ENV` | what it isolates |
|---|---|---|
| A | 15-name `l_out` list, `NANONLY` | baseline — known to fault today (`315size`, `315lout4`) |
| B | the same list, `NANONLY`, **plus** `COPY_PROBE` `SENTINEL` `LOGITS_FP` | the probes' effect on the onset layer, selector held fixed |

**Readout, fixed in advance:** for each Face-A arm in each condition, the **smallest `N` with `nan(l_out-N)>0`**.

| observation | verdict |
|---|---|
| A's onset is layer 0 and B's is layer 41 | the **probes** move the onset — the layer-41 site is an instrument artifact |
| both onsets are layer 0 | the layer-41 reading was a **window artifact** of `tail:16`; this document's title is wrong and is withdrawn |
| both onsets are layer 41 | the name-list config's layer-0 reading was an artifact (e.g. an aliased `l_out-0` read) |
| either condition yields no Face-A arm | **INVALID** for the comparison; reported as such |

The third condition's caveat matters and is checked the same way as before: an `l_out-0` reading is only
evidence if its **hash is unique** in that arm. In `315size-r2e`, `l_out-0`'s hash (`3fb58e998569ad4d`) is
unique among the printed lines, so that arm's reading is not an obvious alias.

## What must change in the record

- The status record's criterion-1 row must not say the site is layer 41 without the configuration qualifier.
- `315-layer41-site-by-graph-arithmetic.md` stands as a description of that node and its arithmetic, but its
  claim to be *the* site is narrowed to *the site under the probe-heavy config*.
- The row-size result is an invariant of the probe-heavy config.
- **The discriminating experiment this implies:** run the *same* fingerprint window on both configurations in
  the *same* session, at the same regime, and find whether the onset layer moves with the probe set. That is
  a config-only test and it settles whether "layer 41" is instrument or defect.

### The `l_out-0` reading is not an alias (checked on all three Face-A arms)

The remaining way the comparison could fail is if `l_out-0`'s layer-0 reading were itself an aliased read.
It is not. Grouping each `315size` Face-A arm's `l_out` readings by content hash:

| arm | coincident readings |
|---|---|
| `315size-r1e` | `l_out-10 ≡ 15 ≡ 20 ≡ 25 ≡ 30 ≡ 35 ≡ 40 ≡ 45` |
| `315size-r2e` | `l_out-10 ≡ 15 ≡ 20 ≡ 25 ≡ 40 ≡ 45`, `l_out-4 ≡ 30 ≡ 35` |
| `315size-r2q` | `l_out-1 ≡ 2 ≡ 3 ≡ 4 ≡ 10 ≡ … ≡ 45` |

**`l_out-0` is in none of them.** Every coincident group on this record consists of layers that are
**saturated** (`nan` equal to the extent), where identical hashes are expected because all-NaN content hashes
the same — the scope rule established earlier. `l_out-0` reads a *partial* count (`1984`, `1984`, `1790`)
with a **unique** hash in every arm, so it is a genuine distinct-buffer measurement.

That also means the layer-0 onset is not a saturation artifact: a partially-NaN reading cannot be produced by
everything being NaN. And it makes the contrast sharper — both sides of the comparison now rest on a
uniquely-hashed, partial reading:

| config | first NaN-bearing tensor | reading | hash |
|---|---|---|---|
| `315nan` / `315fbar` | `node_2948` (layer 41) | `nan=2048` exactly, 18 of 19 arms | unique |
| `315size` | `l_out-0` (layer 0) | `nan=1984`, `1984`, `1790` | unique |

What remains unresolved is only **which** of the two differing variables (selector, probes) produces it —
which is what `315cfg` is running now.
