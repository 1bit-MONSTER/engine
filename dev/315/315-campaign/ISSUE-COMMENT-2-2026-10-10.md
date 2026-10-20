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
## In-regime fault captured with the producing values named — the corruption is in the MoE **routing** tensors, and it is whole-batch, not one row

Follow-up to my previous comment (same day). Diagnostics only; **#315 stays OPEN**, and this is not yet a
mechanism: it is the first in-regime fault captured with the whole-graph *named* fingerprint, and it moves the
site.

### The arm

`315row-p1c`, 2026-10-10, build `315row-build` (pin `e44c9d01a4d5` + the extended fingerprint), composition 3
neighbours at 80 GiB, **`gtt_start=98G`** (the frozen pre-registration's in-regime gate is `>= 95 GiB`). Driver
verdict `pair 1 arm c: FAULT nan=1 req=34`, detector `transport_failures 1`, `identical_distinct 1`. The guard
line and the engine's own abort:

```text
2.53.659.050 E common_sampler_sample: HRX returned NaN logits: count=154880 of 154880, index range [0..154879], contiguous=yes, continuing=no sentinel=0
/home/bcloud/wt/llama-315copy/common/sampling.cpp:654: HRX: NaN logits (engine#123)
[New LWP 100365]
```

```sh
ssh strixhalo 'grep "hrx-tensor-fp" ~/wt/rt-det/315row-p1c-*/server.log'
```

### The fingerprint (16 lines, 3 graph calls, verbatim)

```text
call=158945 ffn_moe_weights-4    GET_ROWS   bytes=160     nan=8     of 40      ne0=1      first=32  row=32  hash=56753a5bd9bd2f87
call=158945 ffn_moe_gate-4       MUL_MAT_ID bytes=245760  nan=51141 of 61440   ne0=1536   first=32  row=0   hash=97678a5700290a33
call=158945 ffn_moe_up-4         MUL_MAT_ID bytes=245760  nan=61440 of 61440   ne0=1536   first=0   row=0   hash=d33023c41f384383
call=158945 ffn_moe_down-4       MUL_MAT_ID bytes=327680  nan=50556 of 81920   ne0=2048   first=640 row=0   hash=b44eb621b102d019
call=158945 ffn_moe_weighted-4   MUL        bytes=327680  nan=81880 of 81920   ne0=2048   first=40  row=0   hash=514da62652bda383
call=158945 l_out-4              ADD        bytes=81920   nan=20480 of 20480   ne0=2048   first=0   row=0   hash=79cb0b351c401383
call=159019 ffn_moe_probs-41     UNARY      bytes=2560    nan=640   of 640     ne0=64     first=0   row=0   hash=4e017b66eae91403
call=159019 ffn_moe_weights-41   GET_ROWS   bytes=160     nan=8     of 40      ne0=1      first=32  row=32  hash=56753a5bd9bd2f87
call=159019 ffn_moe_gate-41      MUL_MAT_ID bytes=245760  nan=51104 of 61440   ne0=1536   first=32  row=0   hash=0e72d35347f45563
call=159019 ffn_moe_up-41        MUL_MAT_ID bytes=245760  nan=61440 of 61440   ne0=1536   first=0   row=0   hash=d33023c41f384383
call=159019 ffn_moe_down-41      MUL_MAT_ID bytes=327680  nan=51136 of 81920   ne0=2048   first=0   row=0   hash=58642d3df06abb43
call=159019 ffn_moe_weighted-41  MUL        bytes=327680  nan=81880 of 81920   ne0=2048   first=40  row=0   hash=6ba84ddfb3b8f1c7
call=159019 l_out-41             ADD        bytes=81920   nan=20480 of 20480   ne0=2048   first=0   row=0   hash=79cb0b351c401383
call=159029 norm                 RMS_NORM   bytes=8192    nan=2048  of 2048    ne0=2048   first=0   row=0   hash=89ade87e7a93fb83
call=159029 result_norm          MUL        bytes=8192    nan=2016  of 2048    ne0=2048   first=32  row=0   hash=77c1ef05f75f6127
call=159029 result_output        MUL_MAT    bytes=619520  nan=143745 of 154880 ne0=154880 first=32  row=0   hash=beb27746f7308f47
```

### Four things it establishes

1. **The routing tensors carry NaN, not only the aggregation.** `ffn_moe_weights` is
   `ggml_get_rows(ffn_moe_probs, ffn_moe_topk)` — the *selected route weights* — 40 values = 4 routes × 10
   tokens, of which 8 are NaN with `first=32`: **tokens 8 and 9, all four routes each**. Its **source**
   `ffn_moe_probs-4` carries no NaN at all (the instrument prints every matched tensor with a non-zero NaN
   count, and `probs-4` is absent). A gather whose output is NaN where its source is finite is either reading
   outside its source — an out-of-range id — or not writing its output.
2. **It is not one row at this stage.** `l_out-4` is `nan=20480 of 20480`: **all ten tokens'** 2048-wide hidden
   rows. `ffn_moe_up-4` is `nan=61440 of 61440` (all 40 route rows). By the end of layer 4 this short prefill
   (T=10) is entirely NaN, and the logits are 143745/154880.
3. **It is one instant.** Only three graph calls in ~159 000 produced fingerprint lines, and they are
   consecutive — layer 4, layer 41, the output. The same graphs had been computed thousands of times before
   with no NaN line.
4. **Two probe defects, recorded rather than quietly fixed.** (a) The filter is `std::strstr(want, node->name)`
   — it searches for the node name *inside* the comma-separated list — so the list entry `ffn_moe_weights-41`
   also matches `ffn_moe_weights-4`. The arm covered **layers 4 and 41 only**, which is why the picture jumps
   from 4 to 41. (b) `ffn_moe_topk`, the ids tensor, is not in the list and would not be readable anyway: the
   instrument reads every tensor as `f32`. The hypothesis this arm most directly suggests — a corrupt or
   out-of-range route id — is therefore **still untested**.

### Tension with the "one token's row" invariant, stated rather than smoothed over

My previous comment reported one `n_embd` row of NaN (2048 floats) as the invariant across 16 faulting arms.
That was measured with a different instrument — a `tail:16` window over the last 16 graph nodes — in a different
session, on prefills of T = 87…342. This arm is a T = 10 prefill measured by name, and it is whole-batch. Both
cannot be quoted as "the fault's shape" at once. Until the same instrument is run over the same arms, the safe
statement is the narrower one: **the corruption involves the MoE routing tensors, and the row extent is
instrument-dependent.** I am explicitly not defending the earlier invariant here.

### What it does not say

No rate (one arm, `req=34` of 66 before the abort, and the instrument is known to raise the fault rate). No
claim about layers other than 4 and 41. No claim that the identical `weights`/`up`/`l_out` hashes at layers 4 and
41 mean aliasing — 100 %-NaN tensors hash equally by construction, and for `weights` (only 8 of 40 NaN) the other
reading is an aliased transient read, which is testable and not adopted here.

### Next measurement, queued

The anchored-match fingerprint with `ffn_moe_topk` and the router logits added, printing ids as integers, plus a
finite-subset hash to settle the aliasing question. Then the rate question, which needs the campaign and the box.
