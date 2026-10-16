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

     engine#315 evidence: the first in-regime fault captured with the whole-graph named fingerprint
     (row probe, arm 315row-p1c, 2026-10-10). Verbatim log lines; every number below is a quote from
     that arm's server.log. -->

# Arm `315row-p1c`: an in-regime Face-A fault with the MoE routing tensors fingerprinted

Session `315row` (2026-10-10 10:44 ADT), build `~/wt/315row-build` (pin `e44c9d01a4d5` + the extended
fingerprint: `ne0=`, `first=`, `row=`, and the named-tensor list including `ffn_moe_*`), composition 3
neighbours at 80 GiB, `gtt_start = 98G`, mode performance.

```sh
ssh strixhalo 'ls -d ~/wt/rt-det/315row-p1c-*/'          # the arm directory
ssh strixhalo 'grep -c "hrx-tensor-fp" ~/wt/rt-det/315row-p1c-*/server.log'   # 16
ssh strixhalo 'grep -m1 "HRX returned NaN logits" ~/wt/rt-det/315row-p1c-*/server.log'
```

Verdict from the driver: `pair 1 arm c: FAULT nan=1 req=34 ret3=0 parse=0 gtt_start=98G`. Detector:
`requests 34`, `http_failures 0`, `transport_failures 1`, `identical_distinct 1`. The guard line, and the
engine's own abort immediately after it:

```text
2.53.659.050 E common_sampler_sample: HRX returned NaN logits: count=154880 of 154880, index range [0..154879], contiguous=yes, continuing=no sentinel=0
/home/bcloud/wt/llama-315copy/common/sampling.cpp:654: HRX: NaN logits (engine#123)
[New LWP 100365]
```

## The fingerprint, verbatim and complete (16 lines, 3 graph calls)

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

## What the arm establishes

1. **A Face-A fault in regime, with the producing values named.** `gtt_start = 98 GiB` (the frozen
   pre-registration's in-regime gate is `>= 95 GiB`, `315-preregistration.md` §5), the canonical whole-buffer
   signature, and the graph values carrying NaN are printed with their node names, ops and byte counts. This is
   the shape task-2's first branch asks for.
2. **The corruption reaches the routing tensors, not just the aggregation.** `ffn_moe_weights` is
   `ggml_get_rows(ffn_moe_probs, ffn_moe_topk)` - the *selected route weights*. Its bytes are `160` = 40 f32 =
   4 routes x 10 tokens, and 8 of its 40 values are NaN with `first=32`, i.e. **tokens 8 and 9, all four routes
   each**. Its source `ffn_moe_probs-4` carries **no** NaN (it is absent from the fingerprint, and the instrument
   prints every matched tensor whose NaN count is non-zero). A gather whose output is NaN while its source is
   finite is either reading outside its source (an out-of-range id) or not writing its output.
3. **It is not one row at this stage.** `l_out-4` is `nan=20480 of 20480` = **all ten tokens' 2048-wide hidden
   rows**, and `ffn_moe_up-4` is `nan=61440 of 61440` (all 40 route rows). At the logits, 143745 of 154880 are
   NaN and the guard's own `count` is the full 154880.
4. **It happens in one instant, not continuously.** Only three graph calls in ~159 000 produced fingerprint
   lines, and they are consecutive (`158945`, `159019`, `159029`: layer 4, layer 41, the output). The same
   graphs had been computed thousands of times before without a NaN line.

## Two readings of the identical hashes - and how to settle them

`ffn_moe_weights-4` and `ffn_moe_weights-41` have the **same hash** (`56753a5b…`) and the same NaN mask; so do
`ffn_moe_up-4`/`up-41` (`d33023c4…`) and `l_out-4`/`l_out-41` (`79cb0b35…`). For the last two the content is
100 % NaN, so an equal hash is expected if the NaN payload is constant. For `weights` only 8 of 40 values are
NaN, so equal hashes mean the other 32 values are equal too - which two *different* layers of a 47-layer model
should not produce.

* **Reading A (aliasing / read-before-write).** Both layers' `ffn_moe_weights` are the same reused transient
  buffer and both reads returned the same stale content. That is a visibility fault on a *transient* buffer, and
  it is testable: the fingerprint's `call=` index already distinguishes the two reads, and the transient
  packer's own log (`GGML_HRX_LOG_TRANSIENT_REUSE=1`) records which buffer each layer received.
* **Reading B (coincidence).** Both tensors are dominated by the same NaN payload and the remaining values
  coincide for another reason. This is testable by hashing the finite subset only.

Neither reading is adopted here.

## Two defects in this probe, recorded rather than quietly fixed

1. **The name matching is substring-based and non-anchored.** `ggml-hrx.cpp` filters with
   `std::strstr(want, node->name)`, searching for the *node name inside* the comma-separated `want` list. A list
   entry `ffn_moe_weights-41` therefore also matches a node named `ffn_moe_weights-4`. The arm covered **layers 4
   and 41 only** - not the whole model - and that is why the picture jumps from layer 4 to layer 41.
2. **The ids are never measured.** `ffn_moe_topk` (the top-k route indices) is not in the name list, and the
   instrument reads every tensor as `f32`, so an `i32` ids tensor would be fingerprinted as nonsense. The
   hypothesis this arm most directly suggests - a corrupt or out-of-range id - is therefore still untested.

Both are fixed in the queued follow-up: anchor the match (exact node name, or a `-<layer>,` suffix test), add
`ffn_moe_topk`, `ffn_moe_logits` and the per-layer attention output, and print the first 16 ids as integers when
the tensor type is `i32`.

## What this does not say

* No rate: one arm, `req=34` of 66 before the abort, and the instrument itself is known to raise the fault rate
  (§39). The rate question belongs to the pre-registered campaign.
* No claim that the fault is confined to, or excluded from, any single layer: only layers 4 and 41 were
  instrumented.
* It does **not** reproduce section 47's "exactly one token's row" invariant, and does not refute it either:
  that invariant was measured with a different instrument (`tail:16`, a window over the last 16 graph nodes) in
  a different session, on prefills of T = 87…342, while this arm is a T = 10 prefill measured by name. The two
  observations are in tension, and until the same instrument is run over the same arms, "one row" must not be
  quoted as the fault's shape.

## The complete session tally (added when the run finished)

```sh
ssh strixhalo 'grep -E "^  pair" ~/wt/315row-console.log'
#   pair 1 arm c: FAULT nan=1 req=34 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
#   pair 1 arm l: OK    nan=0 req=66 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=98G)
#   pair 2 arm l: FAULT nan=1 req=34 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
#   pair 3 arm l: OK    nan=0 req=66 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=99G)
#   pair 4 arm l: OK    nan=0 req=66 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=99G)
```

Five arms, **every one at `gtt_start=98G`** (inside the frozen `>= 95 GiB` gate): two faults (`nan=1`, each aborting
at `req=34` of 66) and three arms completing `66/66`. The by-name instrument therefore reproduces the fault in
**2 of 5 in-regime arms**, against the shipped pin's recorded 2/37 = 5.4 % - consistent with section 39's finding
that the instrument is an *inducer*: it raises the rate, which is what makes a few arms informative at all, and
which is why no arm here is eligible and no rate is quoted from it.

Two of the five were the control cell (both build cells are the same binary in this diagnostic). The two faulting
arms are `315row-p1c` and `315row-p2l`; their fingerprints are quoted in the body of this file, and their
disagreements are what exposed the unsynchronised-read defect (section 53), which is the reason those per-node
lines are not used as attribution.
