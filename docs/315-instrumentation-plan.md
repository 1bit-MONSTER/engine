# engine#315 — instrumentation plan (task-2) with the hypothesis it discriminates (task-3)

Pin: `ggml/src/ggml-hrx` at `e44c9d01a4d5110cecaf45472a717f46cd1abbf9`.

## What is already established (from the issue record, not re-derived)

- The NaN guard is **consumer-side**: `common/sampling.cpp` logs
  `HRX returned NaN logits at vocab index %d … (engine#123)` and calls `GGML_ABORT`. It therefore
  reports the *shape* of the damage (whole vocabulary) and cannot see the producer.
- The arena-knob and transient-reuse *configurations* are falsified as fixes, and the logits were
  shown **not** to be host-staged (hard negative on the writeback instrumentation).

## Hypothesis H1 — deferred writeback × buffer reuse

In `ggml/src/ggml-hrx/runtime/command-program-executor.cpp`, a host writeback is **deferred**:

- `GraphReplayStreamState::add_host_writeback(host_destination, mapped_source, size, buffer)` records
  the destination, the **mapped source pointer**, the size, and retains the buffer.
- The actual `std::memcpy(host_destination, mapped_source, size)` happens later, in
  `GraphReplayStreamState::mark_stream_synchronized()`.
- `clear()` drops pending writebacks without copying.

Between registration and sync the source region can be reused (transient arena reuse, graph replay),
so the sync-time copy can read bytes that no longer belong to that tensor. That yields wrong/NaN
logits at the consumer even though the producing kernel was correct — intermittent, regime-dependent,
and invisible to the consumer-side guard. This is consistent with the recorded commit
"ggml-hrx: transient arena reuse without false dependencies".

**Falsifiable predictions**

1. On a faulting arm, at the faulting event, the source fingerprint taken at registration differs
   from the fingerprint taken immediately before the copy.
2. On clean arms the two fingerprints are always equal.
3. If the producer itself wrote non-finite values, the registration-time fingerprint is *already*
   non-finite — that is H2 (producer codegen), which H1 excludes.

## Instrumentation (env-gated, no effect when off)

`GGML_HRX_WRITEBACK_FINGERPRINT=1`:

- in `add_host_writeback()`: fingerprint the source region immediately — FNV-1a over the bytes, a
  non-finite lane count for f16/bf16 payloads (exponent all-ones), and the first/last 16 bytes;
- in `mark_stream_synchronized()`: recompute the source fingerprint immediately **before** the
  memcpy, then fingerprint the destination **after** it;
- one log line per writeback: `size`, source offset, `rec_fp`, `sync_fp`, `dst_fp`,
  `rec_nonfinite`, `sync_nonfinite`, `dst_nonfinite`, `rec_eq_sync`.

Cost is bounded: enabled only by the flag, and the LM-head buffer is ~310 KB per token step.

The struct may need a small extension to carry a stable identity (tensor/op tag, or the recording
sequence number) so a writeback can be tied to a graph node; that is the only functional change and
it is inert when the flag is unset.

## Pre-registered reading of the outcome

| rec_nonfinite | sync_nonfinite | dst_nonfinite | reading |
|---|---|---|---|
| 0 | > 0 | > 0 | **H1**: the region changed between registration and sync → reuse/visibility fault |
| > 0 | > 0 | > 0 | producer wrote non-finite values → H1 excluded, H2 stands |
| 0 | 0 | > 0 | the copy itself corrupts the destination → instrument the copy path |
| 0 | 0 | 0 | this writeback is not the faulting route → move to the next route on the logits path |

## Procedure

**Tension to resolve first, not assume away.** The issue records a hard negative that the logits are
*not* host-staged, while also naming the direct download copy as the route to instrument. H1 therefore
applies to whichever copy actually carries the logits buffer; if that is a different path (for example
the dmabuf mapping read directly by the sampler rather than through a writeback), this instrumentation
shows row 4 and the plan moves to that path. Establishing which copy carries the logits buffer is the
first step of task-2, not an assumption of it.

1. Build with the flag support; confirm the flag off ⇒ byte-identical behaviour on one arm.
2. Run in-regime arms with the flag on (same regime gate as the pre-registration, ≥95 GiB).
3. Capture the record for a faulting event and for at least one clean arm.
4. The mechanism claim follows only if row 1 is observed on a faulting arm **and** the fingerprint is
   stable on clean arms; anything else is reported as the table's corresponding reading.

No configuration knob is a fix: the recorded studies already show the arena knobs do not move the
rate, so the fix must follow from the mechanism, not from a toggle.
