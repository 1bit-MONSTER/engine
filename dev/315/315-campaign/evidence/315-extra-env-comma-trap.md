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

# Instrument trap: `EXTRA_ENV` splits on commas, so only the FIRST fingerprint name survives

Found while running task 2, and it invalidates an assumption several earlier runs rested on.

## The trap

`315-instr-capture.sh` passes environment for the arm like this:

```bash
IFS=',' read -r -a _extra <<< "$EXTRA_ENV"
for kv in "${_extra[@]}"; do SETENV+=(--setenv="$kv"); done
```

`EXTRA_ENV` is a **comma-separated list of `KEY=VALUE` pairs**, so a comma inside a VALUE splits it. A name
list therefore cannot be passed:

```
EXTRA_ENV="GGML_HRX_TENSOR_FP=result_norm,result_output,l_out,ffn_out"
# becomes:  --setenv=GGML_HRX_TENSOR_FP=result_norm
#           --setenv=result_output          (not a KEY=VALUE pair)
#           --setenv=l_out
#           --setenv=ffn_out
```

**Only the first name is ever set.** Everything after the first comma is passed as a malformed `--setenv`.

## What it invalidated

Several runs this session passed multi-name lists — `result_norm,result_output,l_out,ffn_out` and later
`ffn_moe_probs` — and the observation "only `result_norm` matched" was read as evidence that the other names
were layer-suffixed (`l_out-0`, `ffn_moe_up-4`). **That is only half true.** The suffix mismatch is real —
`hrx_fp_name_listed` matches exactly per comma-item — but the list never reached the instrument anyway: the
value was truncated at the first comma, so `result_output` was *also* never requested.

Both facts point the same way for the runs that were done (`result_norm` is unsuffixed, so it matched
whether the list survived or not), so no measurement is retracted. But the *reason* recorded for the
`result_output` gap in `315-f32-twin-result-divergence-on-the-faulting-arm.md` — "the exact name does match
in other runs, so its absence in these two arms is unresolved" — is now resolved in the other direction:
**it was never passed.**

## The rule

* To fingerprint **one** tensor, pass `GGML_HRX_TENSOR_FP=<name>` as the first `EXTRA_ENV` entry, with any
  further options as *separate* comma-separated `KEY=VALUE` pairs.
* To fingerprint **several**, use a position selector (`tail:<N>` / `head:<N>`), which needs no list — but
  note the position window must actually cover the tensor (measured: `tail:16` covers `attn_norm-*` and
  **not** `ffn_moe_probs-*`).
* Never read "no records" as "zero" — it means the filter or the window missed, per
  `315-PREREG-argsort-precondition.md`'s third reading.

## What task 2 therefore ran

`GGML_HRX_TENSOR_FP=ffn_moe_probs-0,GGML_HRX_BUFFER_TRACE=1` — one tensor, the layer the record documents as
the onset (§59: *"the corruption is already present at layer 0, in the layer-0 FFN or its input"*). If layer 0
turns out not to be the onset in that arm, the reading is "no NaN at layer 0" and the layer sweep is a
separate run.
