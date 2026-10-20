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

# RESULT (task 2): §51's precondition is NOT measurable with the by-name graph fingerprint — a bounded negative

`315-PREREG-argsort-precondition.md` ran. Its third reading applies, and it is reported as such rather than
read as "zero NaN".

## What ran

`315apre3`, `-ub 512`, in-regime, eviction forcer, `EXTRA_ENV="GGML_HRX_TENSOR_FP=ffn_moe_probs-0,
GGML_HRX_BUFFER_TRACE=1"` (one name — see `315-extra-env-comma-trap.md` for why one).

**The arm faulted, so a faulting arm was available:**

```
arm 1 verdict: gtt_start 98  nan=1  hsa=0  utf8=0
315HUNT_FAULT arm=1
ffn_moe_probs-0 records: 0     total twin records: 0
```

Zero records is **not** zero NaN: nothing matched, so nothing was read.

## Why it could not match — measured, not assumed

The `tail:16` census of the same configuration (`315apre2`, which produced 22,695 records across **946
distinct node names**) contains the MoE-adjacent tensors

```
ffn_moe_out-*     ffn_shexp-*
```

and **none** of

```
ffn_moe_probs-*   ffn_moe_topk-*   ffn_moe_argsort-*   ffn_moe_weights-*
```

while the layer families that *are* present read `norm-*`, `attn_norm-*`, `q-*`, `k_pe-*`, `Qcur-*`,
`Kcur-*`, `q_nope-*`, `l_out-*`, `kv_cmpr-*`, `ffn_gate-*`, `ffn_up-*`, `ffn_swiglu-*`, `ffn_out-*`.

So the tensor §51's chain depends on is **not a named non-leaf node in the graphs this instrument sees**.
Two candidate reasons, and this run does not separate them: it may be a **graph input (a leaf)**, which the
fingerprint loop — which iterates `graph->nodes` operations — does not walk; or it may be named differently
in this build. Distinguishing them needs a run that dumps **leaf** names too, which is a different
instrument.

## The bounded negative, stated as such

**§51's rank-sort precondition is unmeasured, and it cannot be measured by the by-name graph fingerprint.**
The producer half and the consumer half of §51 remain what they were — verified in the IR (`483a07a`) — and
the precondition that connects them stays open. This is a **negative about the instrument, not about the
site**: nothing here says the precondition does or does not hold.

It also bounds §50's central observation from a second direction. §50 read `ffn_moe_weights-4` as NaN while
`ffn_moe_probs-4` was finite and concluded "either reading out of range (a bad id) or not writing". Since
`ffn_moe_probs-*` is not among the 946 names this build's fingerprint walks, **how that earlier probe read
it is itself unexplained** — the earlier probe used anchored per-layer names, and this run shows the name is
not reachable by the same mechanism. That is recorded as an open discrepancy, not smoothed over.

## Cost, and what would settle it

One arm, no instrument edit, no rebuild. Settling it needs either (a) an instrument that enumerates leaves
as well as operations, or (b) the anchored-name mechanism the earlier probe used, if it can be reproduced.
Both are new instrument changes and both need an owner-approved run.
