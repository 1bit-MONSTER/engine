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
<!-- engine#315: the instrumented nan-dump arm hunt. One run, one purpose: put the per-tensor NaN
     fingerprint in front of a fault so the first NaN-bearing stage gets a name. It found a stage and,
     more importantly, found that the instrument itself moves the fault rate by nearly two orders of
     magnitude - so it is a lead, not a measurement. Run on strixhalo, box free at launch (GTT 26.5 GiB,
     lock free), killed by its own 8400 s timeout after 17 of 20 pairs. -->

# The instrumented nan-dump arm hunt (bg task `b17b4c08b`)

```
REGIME_LO=70 REGIME_HI=85 CONTROL_EVERY=999 NOSTOP=1 \
EXTRA_ENV="GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_LOGITS_FP=100000 \
           GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1" \
bash ~/wt/315-paired.sh 20 2 2048 ~/wt/315instr3-build/bin ~/wt/315fp-build/bin 315nan
```

## The tally

| cell | arms | fault | clean |
|---|---|---|---|
| fix (`315fp-build`, carries the fingerprint) | 17 | 16 | 1 (pair 16, `req=66`) |
| control (`315instr3-build`, pair 1 only, same `EXTRA_ENV`) | 1 | 1 | 0 |

Every fault is Face A: `nan=1` on the `common_sampler_sample: HRX returned NaN logits` line, `hsa=0`, and no
`Content-only format` 500 in any arm - this run produced **no Face-B** event. `gtt_start=98G` in every arm, so every
arm is inside the gate, and the gate is not what moved. Fault request index across the run: 6, 8, 10, 10, 10, 33, 33,
33, 33, 34 (control), 35, 41, 46, 54 - not a fixed index, unlike the earlier 10/19/34 clustering. Exactly one fault
line per arm; the detector stops the arm at its first fault.

## This tally is not the shipped rate, and says so

16/17 = 94.1% against the recorded 2/37 = 5.4%: Fisher exact two-sided **p = 1.2e-10**. The control cell faulting
too - and that build not implementing `GGML_HRX_TENSOR_FP` - puts the trigger in the environment both cells share,
not in the fix-cell build. Both candidate instruments are host-side interventions aimed at exactly the buffers this
defect concerns: `GGML_HRX_SENTINEL` fills the logits allocation with a NaN payload, and `GGML_HRX_TENSOR_FP=tail:16`
adds 16 device-to-host reads per `graph_compute`. Until an instruments-off arm is run, **this run measures the
instrument, not the regime**, and the stage names below may mark where my reads land rather than where the
production fault is born. The single clean arm (`req=66`, a full 66-request arm) does establish that the arms are not
structurally broken and that the scoring rule is intact.

## What it does establish: the first NaN-bearing stage

The fingerprint's first NaN-bearing call (`call=51764`) carries one token's row, NaN, propagating outward:

| node | op | NaN |
|---|---|---|
| `ffn_shexp-41` | MUL_MAT (shared expert) | 2048 of 352256 |
| `ffn_swiglu-41` | GLU | 2048 of 264192 |
| `ffn_moe_out-41` | ADD | 1984 |
| `ffn_out-41` | ADD | 1984 |
| `l_out-41` | ADD | 1984 |
| `norm-42` (final norm) | RMS_NORM | 2048 of 352256 |

2048 is one `n_embd` row, so a single token's hidden vector is NaN from the last block's FFN onward. The next call's
`node_2985 FLASH_ATTN_EXT` (815496 of 1761280) and its `(permuted)` PERMUTE (1254220 of 1761280) carry it forward as
consumers, not producers. That agrees with section 23 - the producer is upstream of the final norm and the LM head is
exonerated - and narrows it from "somewhere upstream" to the FFN path. The window is `tail:16`, so it begins after
`ffn_norm-41`: it cannot say whether the FFN's input row was already NaN (the attention/residual of block 41) or the
FFN matmul fabricated NaN from clean input.

## Next, in order

1. **Attribute the 94%**: one arm with `GGML_HRX_TENSOR_FP` unset, then one with `GGML_HRX_SENTINEL` unset, then one
   with both unset. If the last returns to a few percent, the fingerprint is an inducer and nothing above it is a
   mechanism; if it stays high, the trigger is something else in the environment and the stage list becomes evidence.
2. **Widen the window** to `tail:64` (or a name list over `ffn_norm-41`/`inp_attn-41`) on a faulting arm, to decide
   the FFN-input question above.
3. Only then return to the MLA `flash_attn_ext` branch, which remains the one path with no test coverage.

## Housekeeping

The run was killed at its 8400 s timeout with 19 arm directories on disk (17 scored pairs plus the control and a
partial). No `315` server survived the kill; GTT returned to 26.9 GiB. Logs:
`~/wt/rt-det/315nan-p<N>{c,l}-GLM-4.7-Flash-Q4_K_M/server.log` plus `~/wt/315nan-console.log` on strixhalo (the
glob must be `*-*l-*`, not `*-l-*`).

## Housekeeping follow-up: the timeout killed the wrapper, not the run

The 8400 s timeout terminated the wrapper, not `315-paired.sh`: the driver (pid 3182848) survived, still held
the box lock, and was mid **pair 19** when this was found - so the session went on to pairs 18-20 after the
timeout was reported, and its console log is longer than the "17 pairs" read above. Treating what looked like an
orphan arm server (`315fp-build`, port 19650) as garbage and SIGKILLing it destroyed **pair 19's fix-cell arm**:
that arm's outcome is my doing and must not be counted in any tally. Anything after pair 17 in the console has to
be read with that in hand.

The env-var inventory is what decides how the faulting control arm reads. `315instr3-build` implements neither
`GGML_HRX_SENTINEL` nor `GGML_HRX_TENSOR_FP` nor `GGML_HRX_COPY_PROBE`, and **no** build implements
`GGML_HRX_LOGITS_FP` (passed throughout, a no-op everywhere). So the control arm ran with no instrument code
active at all - which is why "the instrument induces it" cannot rest on that arm: at the recorded 5.4%, a single
faulting control arm has probability 0.054, i.e. unremarkable. Attribution requires the interleaved design in
`315-attrib.sh` (stock and instrumented arms alternating inside one composition and one box state).
