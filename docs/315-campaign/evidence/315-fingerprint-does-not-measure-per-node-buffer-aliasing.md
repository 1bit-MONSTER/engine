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

# The tensor fingerprint does not measure per-node state — buffer aliasing explains the "read-pattern-dependent onset", and no per-node localization exists on this record

Found from the `hash` field, which the record has always carried and never been read this way. No GPU.

## The evidence

**`node_2948` and `ffn_shexp-41` are the same buffer** (`315fbar-p1c`):

    node_2948      hash = 65f4b97ca74c   ne0=2048  rows=39
    ffn_shexp-41   hash = 65f4b97ca74c   ne0=2048  rows=39

Identical hash and identical shape. They are one allocation reported under two names.

**The residual stream is double-buffered, then single-buffered** (`315fbar-p1c`):

    l_out-42 = d69318656684      l_out-44 = d69318656684     <- same buffer
    l_out-43 = 4365d17f7e43      l_out-45 = 4365d17f7e43     <- same buffer

and in `315fp2-p2l`, **47 distinct `l_out-N` names collapse to 4 distinct hashes**, with

    l_out-1 = l_out-10 = l_out-11 = l_out-12 = l_out-13 = … = ffn_shexp-41 = 0b1627cc1a65

so essentially the whole layer stack reads back **one** shared residual allocation. Only `l_out-0`
(`f02981534fcf`) is distinct.

## What that means

The fingerprint reads a **buffer**, at a point where the graph has already finished writing it. So **every
tensor name bound to a shared allocation reports that one allocation's final content**, and the fields
`nan/of/first/row/hash` are identical for all of them.

That single fact explains the record's most stubborn anomaly. `315-onset-is-set-by-read-pattern-not-probes.md`
measured that the onset moves from layer 0 to layer 41 when the *selector* changes and does not move when
*probes* are added — and concluded that "the read pattern sets the onset". **The real explanation is simpler:
"onset" = the first name in the fingerprint's visit order whose (shared) buffer holds NaN.** It is a property of
the read order and of the aliasing map, and of nothing else. There was never a layer-level onset to find.

It also explains why **`l_out-1 … l_out-46` are all fully NaN** — which read literally would mean forty-six
independent layer failures, an absurdity. It is one buffer, holding NaN, read forty-six times.

## What is therefore retracted

- **"The onset is layer 41."** Not a measurement. The first NaN-carrying **buffer** is visited at that point;
  which layer produced the NaN is unrecoverable from this record.
- **"The onset is a full row at layer 41's FFN and its residual ADD (`node_2948`)"** — the row claim survives
  for `node_2948` itself (its own shape is 2048×39 and exactly 2048 are NaN), but the **layer attribution does
  not**, and `ffn_shexp-41` sharing `node_2948`'s buffer means they are one observation, not two.
- **"`node_2948`'s arithmetic derivation is confirmed independently."** It is *not* — the graph arithmetic and
  the fingerprint read order are not independent if the fingerprint's order is itself derived from the same
  graph layout. The arithmetic derivation stands on its own; the "confirmation" was circular.
- **"`row == 0` in 92 % of readings"** — mostly an artefact. Of the 45 readings where `nan == ne0`, **36 have
  `rows == 1`**, where "one row" *is* the whole tensor. Only ~9 are genuinely one-row-out-of-many.
- **The surviving candidate set for the fault's unit** returns to being unpinned. The 256-byte granularity is
  measured on tensors whose values are shared-buffer snapshots, so which granule belongs to which computation
  is not established.

## What genuinely survives

- **The endpoint result holds** and is not alias-contaminated for that tensor: `result_output` is a **1-row,
  uniquely-hashed** (154880-element) tensor, and in Face-A arms it is **entirely NaN** — so *NaN logits* is
  confirmed at the one place where the reading is unambiguous.
- **`ffn_swiglu-41` has a unique hash** and so is a real per-node datum — and its NaN count is **2048**, which is
  its **own** row is 1536. So the NaN in it is not aligned to its own shape but to a **2048-element** unit.
  That is consistent with the NaN having been produced by a residual-stream-shaped operation, and it is the one
  grain of per-node information the record actually contains.
- **Face B's reclassification stands** (`nan=0`, ~half benign prose, disjoint from Face A) — it rests on the
  logits path and the output text, not on per-node readings.
- **The three instrument-integrity defects already recorded stand**, and this is a fourth and more serious one:
  the earlier three each hid a question; this one **invalidates a whole class of answers that were already
  published as findings**, including the onset layer, the read-pattern result's interpretation, and the
  independence of the `node_2948` confirmation.

## Consequence for the plan

Per-node localization is **not available from this record at all** — not "not yet measured", but structurally
unavailable, because the instrument's read point is after the writes and the aliasing map merges the names.
Any future localization needs the fingerprint to read a node's buffer **at that node's completion**, or the
graph to be instrumented at the aliasing level rather than the name level.

That does not block the endpoint questions, and it sharpens them: `result_output` is unambiguous, `ffn_swiglu-41`
is unambiguous, and Face B is unambiguous. **Three measurements are unaffected by aliasing and all three are
still open** — which is why the next arm should instrument the endpoint and the uniquely-buffered tensors, not
the layer stack.

**No mechanism is claimed.** What is claimed is that a fourth instrument defect is now established, that it
retroactively invalidates five published findings in this directory, and that the valid observations are a
smaller and cleaner set than the record believed.
