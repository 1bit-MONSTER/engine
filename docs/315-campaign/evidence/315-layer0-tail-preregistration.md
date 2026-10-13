<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Pre-registration: `tail:16` to localize within layer 0

Written **before** the run. Selector is the only change from the configuration that produced Face A at scale.

## What is already established, and what this asks

`315-nan-origin-is-layer-0.md` (arm `315lout4` round 1 arm e, Face A, `gtt_start=98G`): the residual stream carries
**1,984 of 200,704 NaN at `l_out-0` (0.99 %)** and grows to 100 % by layer 15, with `l_out-15 … l_out-45` all
saturated to one identical hash. So **the origin is at or before layer 0's output**, and layer 0 is the one region
the record has never covered.

`l_out-0` is the **last** node of layer 0's chunk, and a `tail:N` window covers a chunk's **last N** nodes — the
faithful class. Since the corruption at layer 0 is **small** (0.99 %, not saturation), a window over layer 0's last
16 nodes should show the count **growing inside the window**, which is what localizes it.

## The run

    BIN=~/wt/315fp-build/bin
    EXTRA_ENV="GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_LOGITS_FP=100000 \
               GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1"

This is the **`315nan` configuration** — 16 Face-A faults in 17 arms, the highest Face-A yield on the record —
with nothing changed except the session tag. The selector returns to `tail:16`, which is what that yield was
measured with.

## Readout, fixed in advance

For the **earliest chunk** in a faulting Face-A arm (layer 0's):

| observation | meaning |
|---|---|
| the window's **first** node already carries ≈1,984 NaN | the producer is **before** the window — layer 0's attention or the embedding — and this run cannot localize further |
| a **gradient inside the window**: later nodes carry more NaN than earlier ones, with a first NaN-bearing node partway in | **that node is the producer**, and it is named in the line |
| every window node carries 0 NaN while a later chunk is NaN | the corruption is in the gap between chunks — recorded as such, not as a contradiction |

For any chunk, the **earliest NaN-bearing node in node order** is the candidate producer for that chunk.

## Confounds, stated before the run

- **`tail:16` is the instrument that induces.** The record holds that this fingerprint raises the fault rate by
  nearly two orders of magnitude (§39) and that `315nan` produced 16/17 faults. No rate is quoted from this run and
  no arm is an acceptance arm.
- **The window is the *last* 16 nodes.** Anything earlier in the chunk — including layer 0's attention, if it lives
  in the same chunk — is outside it, and is not measured. A "producer before the window" outcome is a limit of the
  instrument, not a property of the fault.
- **Growth is expected from any single-row origin**, so a growing count inside the window does not by itself
  distinguish "produced here" from "inherited and mixed".
- **Chunk boundaries are inferred**, from the recorded node order and the call indices. If a layer is split across
  chunks, the first chunk's own tail is what matters and should be read first.
- One arm's gradient is not a rate; the layer-0 finding itself is **n = 1** so far, and a second Face-A arm
  reproducing it is the first thing to check.
