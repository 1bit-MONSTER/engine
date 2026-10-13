<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The origin is at or before layer 4 — and layers 0–3 were never instrumented

## The reading, from a run already on the record

`315row-p1c` is the campaign's in-regime Face-A arm with the named fingerprint (`gtt_start=98G`, guard:
`count=154880 of 154880`). Its complete set of `l_out-*` lines:

    name=l_out-4   op=ADD  bytes=81920  nan=20480 of 20480  ne0=2048  first=0  row=0  hash=79cb0b351c401383
    name=l_out-41  op=ADD  bytes=81920  nan=20480 of 20480  ne0=2048  first=0  row=0

and the layers the name list actually covered — which is the point — were:

    $ grep -oE "name=[a-z_]+-[0-9]+" server.log | sed -E 's/name=[a-z_]+-//' | sort -n -u
    4 41

**Only layers 4 and 41.** That is the recorded `strstr` defect: the filter was `std::strstr(want, node->name)`,
so a list entry `ffn_moe_weights-41` also matched a node named `ffn_moe_weights-4`, and the string list happened
to yield these two layers.

## Why this is a narrowing and not just a defect

`l_out-<layer>` is the **last node of that layer's chunk** (verified from the node order:
`ffn_moe_weights → gate → up → down → weighted → l_out`), so under the positional faithfulness rule it is the
**least displacement-exposed** tensor in the layer — the one class of reading this campaign can trust.

It reads **all-NaN**: `nan=20480 of 20480`, `first=0`, i.e. **every one of the 10 tokens' 2048-wide rows is NaN,
from element 0**. So:

> **The residual stream is entirely NaN immediately after layer 4. The corruption is at or before layer 4.**

Two consequences the campaign should act on:

1. **The search has been looking in the wrong place.** The record's sharpest per-node lead pointed at layer 41
   (`ffn_shexp-41`) and the head; but layer 41 is *downstream* of a stream that is **already fully NaN at layer 4**.
   Whatever layer 41 shows, the origin cannot be there. Layer 41's NaN is inherited.
2. **Layers 0–3 have never been instrumented.** They were excluded by the same `strstr` defect, and nothing in the
   record covers them. The origin is in that unmeasured prefix.

**And the breadth matters as much as the position.** It is not one row that is NaN at layer 4 — it is **all 10
rows**. A single corrupted value propagating would not do that within four layers of a 47-layer model; a NaN in
something **shared across tokens** — the embedding, a shared weight, a norm statistic, or the first block's
attention — would. That is a different shape of candidate from anything the campaign has been testing.

## What this does not say

- `first=0` and `nan = bytes/4` mean the whole tensor; the fingerprint reports no finite subset, so nothing here
  distinguishes "written as NaN" from "written from a NaN input" at layer 4 itself.
- The reading is faithful **relative** to the name-matched intermediates around it (it is last in its chunk), not
  absolutely: nodes after it in `graph->nodes` could in principle still displace it.
- One arm. The fault is intermittent; a second in-regime Face-A arm with the same coverage would be worth having,
  and the `315row` session did produce one (`315row-p2l`) — its `l_out` lines should be checked for the same
  reading before this is treated as fixed.

## The obvious next measurement

A **layer sweep of the prefix** — `l_out-0 … l_out-7` — with the same faithful selector, on a Face-A arm. That is
exactly what `315lout4` (running now, selector `l_out-0, l_out-3, …, l_out-45, 46, 47`) covers at coarse
resolution: it samples layers 0 and 3, which bracket the region this finding says the origin must be in.
