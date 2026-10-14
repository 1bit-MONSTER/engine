<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Proposed + pre-registered: the DEVICE-buffer producer probe — the instrument that can reach the onset tensor

Written because the host-only producer probe (`315prod`, queued) provably cannot answer the question the
positional finding raises. No build and no run are attempted here; this is the design and its readout, fixed
in advance so the run is a test rather than a search.

## Why the existing probe cannot reach the onset

`producer_probe_target` requires `ggml_backend_buffer_is_host(t->buffer)` **and** a size in 256 KiB … 8 MiB.
Against the recorded inventory that admits exactly one tensor — the 619,520-byte **logits** region. The
`hrx-producer` dump prints `host=0` for every layer node (`q-0`, `norm-0`, `q_nope-0`, …), so the layer
residuals are device-side and can **never** be armed by it.

But the positional result (`315-one-row-at-the-front-of-the-buffer.md`) puts the onset at exactly such a
device tensor: **one `n_embd`-wide row, at row 0 or 1, in a layer residual**. So the decisive question —
*did the device write that row?* — is one the host-only instrument is structurally unable to ask.

## The instrument is feasible with primitives already in the fork

    ggml-hrx.cpp:442  static void backend_set_tensor_async(backend, tensor, data, offset, size)
    ggml-hrx.cpp:466  static void backend_get_tensor_async(backend, tensor, data, offset, size)
    ggml-hrx.cpp:542  backend_set_tensor_async,      <- registered in the backend iface

Both take an arbitrary HRX tensor (device **or** host) through `ggml_backend_hrx_tensor_binding`, and each
falls back to a synchronous copy when the host side is not a registered host buffer. So the symmetric probe is
the existing pattern with the `is_host` clause dropped:

- **before** the graph: `ggml_backend_tensor_set(node, pattern, 0, ggml_nbytes(node))` — fill a **device**
  tensor;
- **after** a drained stream: `hrx_stream_synchronize` then `ggml_backend_tensor_get(node, buf, 0, nbytes)`;
- classify, exactly as the host probe does.

## Design decision: a FINITE sentinel, not a NaN payload

The host probe fills `0x7FC0DEAD`, which is a NaN. For a **layer residual that is also an op's input** that is
a serious confound: if the op fails to overwrite it, the injected NaNs propagate through the rest of the layer
and produce *the very symptom under study* — the run would be diagnosing its own sentinel, and every
downstream observation in that arm would be unusable.

Use a **finite** pattern instead (`0xDEADBEEFu`), so:

- `sentinel == n` → the op did not write the tensor at all;
- `sentinel == 0` → the op wrote every word, and any NaN present is **genuine**;
- `0 < sentinel < n` → the op wrote part of it — and this is the interesting case, because a partially
  written row-oriented buffer is exactly what "one row corrupted at the front" would look like from the
  writing side.

The collision risk (a legitimate word coincidentally equal to `0xDEADBEEF`) is stated and is ~0 for a single
fixed 32-bit pattern over 2⁵² words, and would in any case *reduce* the measured `sentinel` count.

## The run

    BIN=<a build carrying the dropped-is_host probe, see below>
    EXTRA_ENV="GGML_HRX_PRODUCER_PROBE=2 GGML_HRX_PRODUCER_FINITE=1 \
               GGML_HRX_TENSOR_FP=l_out-0,…,l_out-47 GGML_HRX_TENSOR_FP_NANONLY=1"
    ~/wt/315-disc.sh 8 315dev

**This needs a build**, which is why it is a proposal and not a queued run: the host-only clause and the NaN
payload are both compiled in, so `~/wt/315probe-build` cannot do this. The change is small (drop one clause in
`producer_probe_target`, add a finite-payload branch in `producer_probe_fill`) and is **diagnostic-only**, so it
cannot weaken any guard.

**Target selection** must be widened deliberately, and the selector is the hard part: the band should be
`n_embd × T × 4` for the arm's T, and restricted to residual-family nodes so the probe does not arm every
activation in the graph. The pre-registered selector is **any device tensor whose byte size is within ±1 % of
`2048 × T × 4`**, which for a prefill chunk targets the residual-stream tensors and nothing else. If that arms
nothing, the run is INVALID and the selector — not the conclusion — is revised.

## Readout, fixed in advance

| observation | verdict |
|---|---|
| `sentinel == 0` on the onset tensor in a faulting arm, with `nan > 0` | the device **did** write the whole tensor, and the NaN is genuine ⇒ the corruption is *not* an unwritten-buffer/visibility defect; the writer is an op |
| `0 < sentinel < n`, and the unwritten words are **contiguous** | the op wrote a prefix and left the rest — a row-oriented write that did not cover the region, which is a mechanism |
| `sentinel == n` on the onset tensor | the tensor was never written at all: the consumer reads pre-fill content ⇒ the visibility/staleness family, at the point of production |
| `sentinel == n` on a tensor that is *not* the reported onset | the probe armed the wrong tensors; INVALID, fix the selector |
| no `armed` line | INVALID, widen the selector |

## Confounds, declared before the run

- **Filling a live tensor perturbs the graph.** Even a finite pattern changes downstream values for that
  tensor; the classification at the probed tensor is still valid (it asks only whether the op overwrote the
  words), but the arm's *other* observations are not comparable to a natural arm.
- **The probe adds a host⇄device copy per hooked graph**, which the record already establishes is itself an
  inducer. So this run characterises *the probe's regime*, not the ambient one.
- **`n_embd × T × 4` assumes T from the arm, and T varies within an arm across chunks.** The selector must
  therefore be size-band based, not a single number.
- No rate: a per-graph classifier, no denominator.

## What this would decide that nothing else on the record can

It separates the two families the goal names and the engine cannot distinguish — *"the device wrote these
bytes"* versus *"a restore left a pre-eviction copy" / "nothing wrote them"* — **at the tensor the positional
evidence points at**, rather than at the logits region the host-only probe is confined to.
