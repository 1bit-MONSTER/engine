<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Pre-registration: the §40 chain probe **plus** the early layers, in one arm

Written before the run. This replaces the queued `tail:16` run, because a single arm can now answer two questions
that `tail:16` answers only one of.

## The two questions, and why one arm suffices

1. **§40's claim**: *"Every earlier call's window was clean, so the corruption is **born at block 41**, not
   inherited from block 0."* — the campaign already wrote, and explicitly did not run, the probe that settles it
   (`315-mechanism.md` §40).
2. **This session's gradient**: `l_out-0` carries NaN (1,984), growing monotonically to saturation by layer 15 —
   which says the origin is at or before layer 0's output and therefore **contradicts** (1).

Both are answered by one name list, at the same instrumentation dose (~16 names) that produced the Face-A
signature with a clean quiet cell.

## The run

    BIN=~/wt/315fp-build/bin
    EXTRA_ENV="GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_LOGITS_FP=100000 \
      GGML_HRX_TENSOR_FP=l_out-0,l_out-1,l_out-2,l_out-3,l_out-4,l_out-40,ffn_inp-41,ffn_norm-41,ffn_shexp-41,\
    ffn_swiglu-41,ffn_moe_out-41,ffn_out-41,l_out-41,attn_norm-42,ffn_inp-42,ffn_norm-42 \
      GGML_HRX_TENSOR_FP_NANONLY=1"

The block-41 chain is **verbatim from §40**; the early layers are added ahead of it. 16 names total.

## Readout, fixed in advance

| observation in a faulting arm | meaning |
|---|---|
| `l_out-0` NaN **and** `ffn_inp-41`/`ffn_norm-41` **clean** | §40's generalisation is **refuted**: block 41's input was clean while layer 0's output was not, so the corruption is earlier. The early origin stands. |
| early layers **clean** and `ffn_inp-41`/`ffn_norm-41` NaN | **this arm is a different episode from this session's gradient**; §40's account holds for it and the two must be compared arm-by-arm, not generalised from either |
| `l_out-0` NaN **and** `ffn_inp-41` NaN | both are downstream of the same early NaN — consistent with the gradient, and block 41's chain adds nothing about the origin |
| clean arm (66/66) | nothing, as always |

Second, within-block-41 causality, exactly as §40 specifies: if `ffn_inp-41`/`ffn_norm-41` are clean while
`ffn_shexp-41`/`ffn_swiglu-41`/`ffn_moe_out-41` are not, **the FFN branch fabricated it**; if `ffn_inp-41` is NaN,
**the block's input was already poisoned**.

## Confounds, declared

- **Dose.** ~16 names, matching the configuration whose quiet cell was clean (66/66) and whose eviction cell gave
  Face A. Whether the *origin layer* is itself dose-dependent is untested; this arm shares the dose of the gradient
  it is checking, which is what makes them comparable.
- **Name-matched reads are displacement-exposed**; `l_out-*` is least so (last in its chunk), and the block-41
  chain nodes are mid-chunk. A "clean early layer" reading is therefore weaker evidence than a NaN reading.
- **The instrument induces**; no rate is quoted and no arm is an acceptance arm.
- One arm cannot distinguish "born here" from "inherited" for a *single* node; it distinguishes the two competing
  **accounts**, which is the point.
