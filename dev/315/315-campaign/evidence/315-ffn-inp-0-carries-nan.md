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

# `ffn_inp-0` carries NaN — the pre-registered branch fired, on an arm that must be qualified

## The reading

`315inp1`, arm `r1q`, selector `l_out-0,ffn_inp-0,l_out-1,ffn_inp-1,…,l_out-4,ffn_inp-4`:

    name=ffn_inp-0 op=ADD bytes=4194304 nan=1009992 of 1048576
    name=l_out-0   op=ADD bytes=4194304 nan=998820  of 1048576
    name=ffn_inp-1 op=ADD bytes=4194304 nan=1048576 of 1048576
    name=l_out-1   op=ADD bytes=4194304 nan=1048576 of 1048576
    … ffn_inp-2…4 and l_out-2…4: 1048576 of 1048576

The pre-registered readout in `315-lifetime-is-the-faithfulness-rule.md` was:

> if `ffn_inp-0` carries NaN while the embedding is unmeasurable either way, the honest conclusion becomes: **the
> corruption is already present in layer 0's post-attention residual**, and the remaining candidate set is the
> attention branch and the embedding

**`ffn_inp-0` carries 1,009,992 NaN of 1,048,576.** So the branch fires: **the corruption is present after layer 0's
attention and before its FFN.** Layer 0's FFN branch is *not* where it starts — the FFN receives an already-corrupted
`ffn_norm-0`, so everything from `ffn_shexp-0` onward inherits it.

**The remaining candidate set is therefore: layer 0's attention branch, or the embedding that feeds it** — the
former within reach of a chain probe, the latter not (see below).

## The qualification that must travel with it

**This arm is `OOR` — `gtt_start=115G` against the gate `[95,110]`.** The cause is recorded: three of my chained
scripts, which I had abandoned but **never killed**, fired concurrently and ran three or four disc sessions at once.
The box was at **load 28-49** and **GTT 117 GiB** — a saturated machine, not a controlled one. Every arm in that
window is discarded, including this one.

So the *qualified* statement is: **the pre-registered branch fired, and the arm it fired on is out of regime and
contaminated.** The component-level localization is a **lead, not a result**, and `315inp2` (running now on a
verified-clean box, one session only) is the confirmation.

## The anomaly worth noting without interpreting it

`ffn_inp-0` reads **1009992** NaN while `l_out-0` reads **998820** — *fewer*, and `l_out-0 = ADD(ffn_out-0,
ffn_inp-0)` cannot have fewer NaN than `ffn_inp-0` unless something else moved. That is either a consequence of the
saturated box (both counts are mid-flight readings of a machine running three compositions), or a displacement
artifact, or a real difference in what the two tensors hold. **No explanation is adopted here.** It is flagged
because it is exactly the kind of small numerical inconsistency that this campaign has twice built conclusions on
before checking (the truncated sorted head, the mis-indexed `awk`), and because a clean arm is about to make it
checkable.

## Why an embedding-arm read is not available

`ffn_inp-0` is faithful **by lifetime** — it is consumed by `l_out-0`, the chunk's final node. The embedding is not:
it feeds layer 0 and dies there, so its slot is free long before the chunk ends and any post-execution read of it is
displacement-exposed. **The attention branch can be probed through lifetimes (its residual output is `ffn_inp-0`
itself, already read); the embedding cannot be**, and separating "attention corrupt" from "embedding corrupt" is the
next place the host-side instrument runs out.
