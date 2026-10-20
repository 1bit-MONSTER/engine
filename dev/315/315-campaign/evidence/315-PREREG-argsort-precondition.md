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

# PRE-REGISTRATION: §51's rank-sort precondition — does NaN reach the argsort's input?

Task 2 of the re-scoped plan. Written before any run. It must not repeat the failure that invalidated the
previous id instrument, so that failure is stated first and the probe is designed around it.

## The failure the probe must not repeat

`315-moe-id-instrument-never-observed-ids.md`: the previous `[hrx-ids-fp]` instrument read buffers that
**held float data at the moment it read** — `1040594397` is `f32 ≈ 0.131`, `min < 0` in 1.28M of 1.7M
records, byte-identical across clean and faulting arms with a constant "sane" count. Its own conclusion is
that those i32 routing tensors are scratch space shared with a float tensor in the captured graph, so the
probe saw the other tensor's contents. **A by-name post-graph read cannot be trusted for these tensors**,
and §53/§59 say the same for the f32 family.

## What is being tested

§51 verified at both ends in the IR (`483a07a`): the rank sort uses only ordered comparisons, so two NaN
elements in a row take the **same** rank — the output is then not a permutation — and the consumer
(`moe_routing_tables.loom:64`) takes the expert index by `index.assume [range(-1, 511)]` rather than
clamping, while the gather on the same ids clamps.

**The precondition is unmeasured: does a faulting arm actually have NaN at the argsort's input?**

## The probe, designed around the aliasing failure

The precondition can be tested **without reading the i32 ids at all**, which is what makes it viable:

1. **Read the argsort's INPUT tensor** — `ffn_moe_probs`, an **f32** tensor — under the fingerprint that
   already exists and is trustworthy for f32 (`hrx-f32-sync`, verified: it compares two reads of the same
   buffer at the same call, so arena reuse changes both together and cannot fake a disagreement).
2. **Report NaN count and the first NaN index**, per call, in the faulting arm.
3. **A NaN at the input is the precondition**, and it is observable on the f32 tensor the instrument is
   already validated for. No i32 read is required, so the aliasing failure cannot recur.

The i32 claim — "the rank map is not a permutation" — is then the **inference** from §51's verified IR, not
a measurement, and is labelled as such.

## Pre-registered cells and configuration

| cell | purpose |
|---|---|
| **faulting arm** (eviction-forced, diagnostic — no arm counts toward a campaign) | does the argsort's input carry NaN in an arm that faults? |

Configuration: `GGML_HRX_TENSOR_FP=ffn_moe_probs,GGML_HRX_BUFFER_TRACE=1` (f32 tensor, existing validated
instrument), `-ub 512`, in-regime, eviction forcer, stop at the first fault.

## Pre-declared readings

| observation in the faulting arm | reading |
|---|---|
| `ffn_moe_probs` carries **≥2 NaN in a row** at the same call the arm faults | §51's precondition **holds**: the rank sort cannot be a bijection on that row, and the unchecked `index.assume` consumer is reachable. The mechanism becomes a verified chain — precondition measured here, producer verified in the IR, consumer verified in the IR. |
| `ffn_moe_probs` carries **0 NaN** throughout the faulting arm | the precondition **fails**, §51's site is dead as the *origin*, and NaN must be introduced after this point — which moves the hunt to the table builders and the matmul's descriptor use (§51's own falsifier). |
| the tensor produces **no records** in the faulting arm | uninformative — the by-name filter missed it; re-run with a wider `tail:<N>` and say so, rather than reading absence as zero. |

## Falsifier

A faulting arm whose `ffn_moe_probs` records show zero NaN at every call, including the faulting one.

## Cost and honesty

One arm (the forcer produced a fault in 1 arm for each of the two previous probes), no instrument edit, no
rebuild — the marker `hrx-f32-sync` is already in `~/wt/315fp-build` and the name list is the only change.
If the reading is the second row, §51's site is removed and the survivor list shrinks by one, which is a
bounded result of the kind the re-scoped criterion 1 accepts.
