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
<!-- Copyright 2026 bong-water-water-bong
     SPDX-License-Identifier: Apache-2.0

     engine#315 evidence: the fp2 run - the ordered fingerprint, its one in-regime fault, and the second reason a
     by-name locator cannot work. Verbatim log lines from ~/wt/rt-det/315fp2-p2l-GLM-4.7-Flash-Q4_K_M. -->

# fp2: the ordered fingerprint, one in-regime fault, and why it still cannot name the producer

`315fp2`, 2026-10-10, build `315fp2-build` = the pin + all diagnostic patches + **`ggml_backend_synchronize`
before the fingerprint's read loop** (`tools/patch-fp-sync.py`), anchored name matching, ids dump. Composition
`REGIME_LO=80`, so every arm starts at `gtt_start=98G`.

```text
pair 1 arm c: OK    nan=0 req=66 gtt_start=98G
pair 1 arm l: OK    nan=0 req=66 gtt_start=98G
pair 2 arm l: FAULT nan=1 req=19 gtt_start=98G      <- the one fault, in regime
pair 3 arm l: OK    nan=0 req=66 gtt_start=98G
pair 4 arm l: OK    nan=0 req=66 gtt_start=98G
```

**1 fault in 5 in-regime arms with the sync, against 2 faults in 5 in-regime arms without it** (the `315row`
session, same composition, same model, same harness, same day: Fisher exact p = 1.000). The sync-before-read
therefore neither inhibits nor enables the fault. Together with sections 55-56 - the consumer's own path ends in
`hsa_signal_wait_scacquire` and the waited timepoint covers the last submission - that closes the
visibility/ordering family for the *production* fault: it is not an artefact of when a reader looks.

## Why the fingerprint still cannot name the producer: the arena's last writer

The faulting arm's fingerprint is 977 ordered lines. The first two NaN-bearing tensors, in graph-node order, are:

```text
call=103457 ffn_out-0  ADD  bytes=4194304 nan=7048   of 1048576 ne0=2048 first=37     row=0
call=103457 l_out-0    ADD  bytes=4194304 nan=4963   of 1048576 ne0=2048 first=276639 row=135
```

`4194304` bytes is 512 x 2048 floats, i.e. a **T = 512 prefill**. And the two lines contradict the graph:
`l_out = ffn_out + attn_out`, so `l_out`'s NaN set must **contain** `ffn_out`'s - 4963 cannot be smaller than
7048. The same violation recurs at layer 1 (`ffn_moe_out-1 nan=393216` vs `l_out-1 nan=1048576 of 1048576`, the
whole residual stream).

The only explanation that fits is that these tensors are **transient arena slots that the graph itself has since
reused**: after the graph completes, the slot that held `ffn_out-0` may hold a later tensor's bytes, so reading it
"by name" returns the *last writer's* value, not the named tensor's. Ordering the reads (the fp2 patch) does not
help, because the problem is not when the reader looks but *which* bytes now live at that address. This is the
**second**, independent reason the by-name locator cannot work, and it explains the contradiction that motivated
the withdrawal in section 53 as well as this one.

What remains trustworthy in that arm is therefore only what is not a transient read: the guard's own line, and the
non-transient tail (`norm`, `result_norm`, `result_output`). From those, the fault's *coarse* shape in this arm:

* the corruption is already present at **layer 0** (`ffn_out-0`, `l_out-0`), i.e. upstream of every named MoE
  tensor, in the layer-0 FFN or its input - and layer 0's attention output is not in the instrumented set;
* it cascades fast: by layer 1 the whole residual stream is NaN (`l_out-1 nan=1048576 of 1048576`), and every
  downstream count is 30-100 %;
* it is whole-batch, not one row: 512 rows of 2048.

## The ids diagnostic in this build is mis-specified - recorded so it is not read as evidence

The run's arms report ~253 000 `hrx-ids-fp` lines each of which ~253 000 are flagged
`oor=[1-9]|dup_pairs=[1-9]`. That is not a finding: `oor` was computed against the tensor's own innermost extent
`ne0`, and for the top-k **view** of the argsort `ne0` is `k` (4), not the expert count (64) - so every legitimate
expert id in `[4, 64)` is counted as out of range. The check is wrong by construction, and none of its output is
used anywhere in this record.

## What this leaves

* the ordering family is closed for the production fault (sync present or absent: 1/5 vs 2/5; and the consumer's
  read path is ordered at source);
* localisation by name is closed for a second reason, so task-2's "producing graph `value` named" cannot be
  obtained this way at all - which is what the hard-negative branch of that contract exists for;
* the fault's *coarse* localisation is layer 0 of a T=512 prefill, whole-batch, cascading within one layer;
* the honest next instrument would have to read at the point of production (inside the dispatch, or from a
  non-transient copy) rather than by name afterwards.
