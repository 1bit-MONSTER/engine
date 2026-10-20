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

---

## Second correction: it is an onset *distribution* that differs, not a fixed config property

Two further results, one of which withdraws the title's framing.

### 1. The `tail:16` window DOES reach layer 0 — so the layer-41 onset is not a coverage artifact

In `315nan-p12l`, **one call** (`158937`) prints:

    call=158937 name=norm-0       op=RMS_NORM nan=55  of 20480
    call=158937 name=ffn_norm-0   op=MUL      nan=57  of 20480
    call=158937 name=ffn_gate-0   op=MUL_MAT  nan=1312 of 102400
    call=158937 name=ffn_up-0     op=MUL_MAT  nan=166 of 102400
    call=158937 name=ffn_swiglu-0 op=GLU      nan=204 of 102400
    call=158937 name=q_nope-1     op=VIEW     nan=66  of 51136

A single window spanning layer 0's **MoE block and** layer 1's attention proves the window covers the
layer-0→1 boundary. `node_30 op=FLASH_ATTN_EXT` also prints in this arm, so early attention nodes are watched
too. **The window reaches layer 0.** Therefore the absence of layer-0 NaN in the other 18 arms means layer 0
was genuinely clean there, and the layer-41 onset in that configuration is **not** a window artifact. That
part of the earlier concern is resolved.

### 2. But the same configuration produces BOTH onsets, so "layer 41" is not a config property either

`315nan-p12l` is a **`tail:16` + probes** arm — the same configuration as the 18 late-onset arms — and its
onset is layer 0/1 with *tiny* counts (`norm-0 nan=55`, `ffn_norm-0 nan=57`). So within one configuration:

| onset | arms |
|---|---|
| layer 41, `nan=2048` exactly | 18 of 19 |
| layer 0/1, `nan=55`–`204` | 1 of 19 (`p12l`) |

and in the name-list config:

| onset | arms |
|---|---|
| layer 0, `nan=1790`–`1984` | 3 of 3 |

**The correct statement is therefore a difference of onset *distribution*, not a fixed property of either
configuration**: the probe-heavy config put 18 of 19 onsets at layer 41 and 1 at layer 0/1; the name-list
config put 3 of 3 at layer 0. Both configs can produce an early onset; they differ in how often.

This is the third time this document has been revised, and the sequence is worth stating plainly because it is
the honest record: first "the site is layer 41", then "the onset is config-dependent", now "the onset is a
distribution that differs between configs". Each step was forced by data the previous one had not used
(the `l_out-0` uniqueness check, then the single-call coverage proof, then `p12l`'s counts).

**What survives all three revisions, and is the solid result:**

- The size invariant `nan=2048` exactly, in 18 of 19 arms, across nine sequence lengths, at a uniquely-hashed
  node whose two counts partition the extent — i.e. a complete read of exactly one row.
- `node_2948` = layer 41's unnamed post-attention residual ADD (`71×41+37`; 71 nodes/layer, one unnamed ADD
  at +37 and one unnamed `FLASH_ATTN_EXT` at +3 per layer, 47 each).
- `1984 = 2048 − 64` recurs in **both** configurations (`l_out-41` in the probe arms, `l_out-0` in the
  name-list arms) — a shared constant across an otherwise different presentation, and the most promising
  thread left.

**What does not survive:** any claim that the site is layer 41 *for the defect*. It is the site *most often
observed* under the probe-heavy configuration.

`315cfg` (running) tests the probes as a cause with the selector held fixed, and its readout table already
includes the branch that withdraws this document's framing.

---

## Third refinement: the selector, not the probes, is the prime suspect for the dose difference

`hrx-producer` shows subgraphs of **`n_nodes=9`** (`315-subgraphs-are-9-nodes-so-tail-covers-all.md`), and
`tail:16` covers all of them. That makes the read **volume** of the two configurations very different, and it
is the *selector* that sets it — not the probes:

| config | tensors read per chunk | why |
|---|---|---|
| `tail:16` + probes | **~9** (every node) | 16-node window over a 9-node graph covers all of it |
| 15-name `l_out` list | **1–3** (only matching nodes) | the list is matched against each node's name |

Both configurations also pay the probe reads, but that is a constant added to both sides of *this* comparison
and cannot by itself explain a 9-vs-1 difference. So of the two variables that differ between the original
observations, **the read dose difference is dominated by the selector**, and the probe set is the secondary
candidate.

This matters for what `315cfg` can and cannot settle. Its two conditions vary the **probes** with the selector
held fixed, so it isolates the secondary candidate. The primary one — read dose via the selector — needs a
third condition:

| condition | selector | probes | reads/chunk |
|---|---|---|---|
| A (running) | 15-name `l_out` list | no | ~1–3 |
| B (running) | 15-name `l_out` list | yes | ~1–3 + probe reads |
| **C (not yet run)** | `tail:16` | no | **~9** |

**A vs C is the dose test**: same absence of probes, ~1–3 reads versus ~9. If the onset moves from layer 0 (A)
to layer 41 (C), the read dose is the variable, not the probes — and that is consistent with everything the
record says about the fingerprint being an inducer, with the "dose" now measured in tensors-read-per-chunk
rather than in name-list length.

**Stated as a hypothesis with its falsifier:** the observable onset layer is a monotone function of read dose,
so A (low) → layer 0, B (mid) → intermediate, C (high) → layer 41. Falsified if C's onset is layer 0 like A's,
or if A's is layer 41 like the probe-heavy config's. **Not yet run**; recorded here so the run is a
pre-registered test rather than a response to whatever A and B return.

**Caveat carried forward:** `n_nodes=9` is measured on **one** graph (a q-path chunk at layer 0) in three arms.
MoE chunks, where the layer-41 node lives, are not measured and may be larger — so "~9 reads/chunk" is an
estimate for that chunk type, and if MoE chunks have ≥17 nodes then even condition C's window is partial there.
