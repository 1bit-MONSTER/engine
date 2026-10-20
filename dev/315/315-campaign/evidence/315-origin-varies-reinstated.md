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

# The origin varies across arms of the SAME trigger — reinstated, with the evidence that was missing

I retracted a "the origin moves across arms" claim earlier in this session because the only variation I had was
between **cells** (eviction vs CPU), which proves nothing about episode-to-episode movement. **`315inp2` round 3
arm e supplies the missing case**, and it moves the reading back.

## The arm

    round 3 arm e: FAULT nan=1 req=6 transport=1 ... gtt_start=98G      <- Face A, IN REGIME
    (rounds 1 and 2 arm e both faulted as Face B; rounds 1-3 arms q and c all OK, 66/66)

Selector `l_out-0,ffn_inp-0,l_out-1,ffn_inp-1,…,l_out-4,ffn_inp-4`, and its complete fingerprint:

    call=29549 name=l_out-1   nan=2048  of 200704
    call=29551 name=ffn_inp-2 nan=69632 of 200704
    call=29551 name=l_out-2   nan=69632 of 200704
    call=29553 name=ffn_inp-3 nan=4096  of 200704
    call=29553 name=l_out-3   nan=69632 of 200704
    call=29555 name=ffn_inp-4 nan=69632 of 200704
    call=29555 name=l_out-4   nan=69632 of 200704

**`l_out-0` and `ffn_inp-0` do not appear at all** — under `NANONLY` that means **zero NaN at layer 0**. The first
NaN in the model is **`l_out-1 = 2048 = exactly one `n_embd` row`**.

## Why this reinstates the claim

| arm | trigger | in regime | earliest layer carrying NaN |
|---|---|---|---|
| `315lout4` r1e | eviction | yes (98G) | **0** — `l_out-0` = 1,984 |
| `315lout4` r2e | eviction | yes (98G) | **0** — `l_out-0` = 1,984 |
| `315lout4` r1c | CPU | yes (98G) | **0** — `l_out-0` = 46 |
| **`315inp2` r3e** | **eviction** | **yes (98G)** | **1** — layers 0 and `ffn_inp-0` clean, `l_out-1` = 2,048 |

**Same cell, same regime gate, same trigger — different origin layer.** The earlier retraction was right about the
evidence it had; this is new evidence, and it supports what that retraction withdrew: **the corruption is not
anchored to a fixed layer.**

Two further observations sharpen it:

- **`l_out-1 = 2048` is exactly one row** — the invariant the record calls §47, appearing here as the *earliest*
  corruption rather than at the MoE aggregation.
- **The `1984` "constant" must now be withdrawn as a property.** It appeared in two arms of one run and does not
  recur here (this arm's first count is 2048, not 1984). Within-run agreement was coincidence, not structure.

## An inconsistency I cannot explain, recorded rather than smoothed

`ffn_inp-3` reads **4096** NaN while `l_out-2` reads **69632** — but layer 3's residual input *is* layer 2's output
(`ffn_inp-3 = ADD(attn_out-3, inpSA-3)`, `inpSA-3 = l_out-2`), and `NaN + x = NaN`, so `ffn_inp-3 ≥ l_out-2` must
hold. **It does not.** So at least one of those two readings is not the value the graph produced.

That matters because both are supposed to be **lifetime-faithful** (`l_out-2` is its chunk's last node; `ffn_inp-3`
is consumed by `l_out-3`, the chunk's last node). The likely explanation is the **temporal** hole in the faithfulness
rule that this session has not closed: the fingerprint reads *after* a chunk completes, and if the next chunk has
already been submitted and run, the arena can be re-leased and the slot overwritten **even for a tensor that was
live to the end of its own chunk**. `ggml_backend_synchronize` orders against the device queue, not against the
caller's next submission.

**A check that argues against simple displacement, and which stops this collapsing into "all reads are noise":**
if `l_out-0`'s slot had been overwritten by layer 1's value, `l_out-0` would read 2048 — it reads **zero**. So the
read is not uniformly lagging by one chunk. The inconsistency is therefore real but not explained, and it is flagged
as an open instrument defect rather than folded into the mechanism.

## What stands

- **The origin is not fixed**: layer 0 in three arms, layer 1 in a fourth, all in-regime under the same trigger.
- That is the **position-independent** signature the region-following account predicts — and the account is now
  supported by an episode-to-episode difference rather than by a cell difference.
- **Any per-layer attribution from this instrument carries the temporal caveat above**, so the layers are reliable
  as a *range* and not as an exact start.
