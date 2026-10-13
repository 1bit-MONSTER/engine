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

# Pre-registration: the `l_out-<layer>` sweep — which layer first carries the NaN

Written **before** the run. Existing instrument, no rebuild, frozen rig unchanged except `EXTRA_ENV`.

## Why this is now possible, and why it was not before this stretch

The campaign's per-node NaN attribution was weak because its `tail:16` window puts the *first* NaN-bearing node
(`ffn_shexp-41`) near the **head** of the window — the most displacement-exposed position
(`315-which-readings-are-faithful.md`, refinement section).

But the graphs are **per-layer** (verified: `call=158945` holds only layer 4's MoE nodes, `call=159019` only layer
41's, and the node order within a layer is `ffn_moe_weights → gate → up → down → weighted → l_out`). So
**`l_out-<layer>` is the last node of that layer's chunk** — the least displacement-exposed tensor in it, and the
tensor that carries the residual stream forward.

Reading it by name for every layer therefore gives one **faithful** NaN count per layer, in a single arm, with the
existing instrument.

## The run

    EXTRA_ENV="GGML_HRX_TENSOR_FP=l_out-0,...,l_out-63 GGML_HRX_TENSOR_FP_NANONLY=1"

`NANONLY=1` prints only NaN-bearing lines, so the output is exactly the set of layers whose residual output
carries NaN.

## Readout, fixed in advance

| observation in a faulting arm | meaning |
|---|---|
| smallest `N` with `nan(l_out-N) > 0`, and `l_out-(N-1)` clean | the corruption is **inside layer N's chunk** — the layer is named |
| `l_out-0` already NaN | corruption at or before the first layer's output (embedding / input side) |
| **no** `l_out-N` NaN, yet the arm faults | the corruption is introduced **at or after the final norm / head**, not in any layer's residual output — a distinct mechanism |
| arm does not fault | uninformative about the origin; **INVALID** as evidence, never a pass |

## Confounds, declared

- `l_out-<layer>` is the last *named* node in the chunk. Nodes **after** it in `graph->nodes` would still be able
  to displace it, so "last named" is not proven "last allocated". Its exposure is nonetheless strictly less than
  `ffn_shexp-41`'s, which is the point of the experiment — this is a **relative** improvement in faithfulness, not
  an absolute guarantee.
- The detector aborts an arm at its first fault, so only calls up to the abort are fingerprinted.
- `NANONLY=1` suppresses clean lines, so a layer absent from the output is a layer with zero NaN — that is the
  readout, not a missing measurement.
- Instrumentation is itself an inducer (the record holds that the fingerprint raises the fault rate, §39), so no
  arm here is an acceptance arm and no rate is quoted from it.

No criterion changes.

---

# Amendment (before the second attempt): the environment must be the one that produces Face A

The first attempt ran the sweep against a configuration that faulted **6 of 6 arms as Face B** — no NaN anywhere,
so a `NANONLY` NaN-counting sweep could report nothing. That is a mis-targeted run, not a negative.

The environment that has produced **Face A** at scale in this campaign is the one `315nan` used, which recorded
16 faults in 17 arms, all Face A (`nan=1`, `hsa=0`), at `gtt_start=98G`:

    EXTRA_ENV="GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_LOGITS_FP=100000 \
               GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1"

The second attempt therefore keeps that environment and changes **only** the tensor selector, from
`tail:16` to the `l_out-0…l_out-63` name list — which is the whole point of the sweep, since `l_out-<layer>` is
the last node of its layer's chunk and therefore faithfully readable.

**The readout table above is unchanged**, and one check is added because the first attempt needed it:

- if the arms fault as **Face B** (`nan=0`) again, the sweep is again uninformative and must be reported as a
  **mis-targeted run**, never as "every layer clean".

Note the pre-registration's own confound applies with more force here: `COPY_PROBE` and `LOGITS_FP` add further
host-side reads on top of the 64-name list, so this run is more heavily instrumented than any before it, and the
known inducing effect (§39) is correspondingly larger. No rate is quoted from it and no arm is eligible.
