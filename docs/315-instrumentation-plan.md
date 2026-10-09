# engine#315 — instrumentation plan (task-2) with the hypothesis it discriminates (task-3)

Pin: `ggml/src/ggml-hrx` at `e44c9d01a4d5110cecaf45472a717f46cd1abbf9`.

## What is already established (from the issue record, not re-derived)

- The NaN guard is **consumer-side**: `common/sampling.cpp` logs
  `HRX returned NaN logits at vocab index %d … (engine#123)` and calls `GGML_ABORT`. It therefore
  reports the *shape* of the damage (whole vocabulary) and cannot see the producer.
- The arena-knob and transient-reuse *configurations* are falsified as fixes, and the logits were
  shown **not** to be host-staged (hard negative on the writeback instrumentation).

## Hypothesis H1 — deferred writeback × buffer reuse

**Refined against the pinned source (measured, not inferred).** The download path in
`runtime/command-program-executor.cpp` is:

1. `insert_barrier` → optionally enqueue a stream execution barrier (`"insert HRX host download
   barrier"`);
2. `allocate_mapped_host_staging_buffer(device, length, download_buffer, download_data)` — a
   **mapped host staging buffer**, whose host pointer is `download_data`;
3. `hrx_stream_copy_buffer(stream, staging.buffer, 0, download_buffer, 0, length)` — the copy from
   the tensor into staging is **enqueued on the stream, i.e. asynchronous**;
4. `add_host_writeback(staging.host_data, download_data, length, download_buffer)` — the host-side
   copy is **deferred**, and
5. the actual `std::memcpy(host_destination, mapped_source, size)` happens later, in
   `GraphReplayStreamState::mark_stream_synchronized()`, called from
   `command-program-executor.cpp` and from `ggml-hrx.cpp`.

So there is a window between (3) and (5) in which the host holds a mapping of a staging buffer that
**the device has not necessarily written yet**. Reading it early yields zeros, stale bytes, or
reinterpreted garbage — at the consumer that is indistinguishable from a producer NaN, and it is
timing- and pressure-sensitive, which matches the observed intermittency and regime dependence.

This is not a novel claim: the issue record already argued from the code that *"the flush precedes
the wait, so the download copy is covered"*. The point of instrumenting is that a code reading cannot
establish that ordering **under load** — a fingerprint of the staging buffer at the two moments can.

A second, explicitly synchronous path exists (`synchronous_download_fallback`, for arbitrary GGML
pointers), which is the natural control: it should never show the window.

### Falsifiable predictions

1. On a faulting arm, at the faulting event, the staging-buffer fingerprint taken at registration
   differs from the one taken immediately before the memcpy, and the pre-copy state is a
   not-yet-written pattern (zeros / stale), not a finite tensor.
2. On clean arms the pre-copy fingerprint matches the post-copy destination fingerprint (the copy had
   landed before the host read it).
3. If the device-side source itself holds non-finite values, the pre-copy fingerprint is already
   non-finite — that is H2 (producer codegen), which H1 excludes.
4. The synchronous fallback path never shows the window.

## Instrumentation (env-gated, no effect when off)

`GGML_HRX_WRITEBACK_FINGERPRINT=1`:

- in `add_host_writeback()`: fingerprint the source region immediately — FNV-1a over the bytes, a
  non-finite lane count for f16/bf16 payloads (exponent all-ones), and the first/last 16 bytes;
- in `mark_stream_synchronized()`: recompute the source fingerprint immediately **before** the
  memcpy, then fingerprint the destination **after** it;
- one log line per writeback: `size`, source offset, `rec_fp`, `sync_fp`, `dst_fp`,
  `rec_class`, `sync_class`, `dst_class`, `rec_eq_sync`, the barrier flag for that staging entry, and
  the registration→sync delay; where `class` ∈ {ZERO, NONFINITE, FINITE} so that "not yet written" is
  distinguishable from "written and non-finite".

Cost is bounded: enabled only by the flag, and the LM-head buffer is ~310 KB per token step.

The struct may need a small extension to carry a stable identity (tensor/op tag, or the recording
sequence number) so a writeback can be tied to a graph node; that is the only functional change and
it is inert when the flag is unset.

## Pre-registered reading of the outcome

| pre-copy class | dst class | reading |
|---|---|---|
| NONFINITE | NONFINITE | the device-side source already held non-finite values → **H1 excluded, H2 (producer) stands** |
| ZERO or stale | NONFINITE | the host read the staging buffer before the device copy landed, and the bytes it got were non-finite → **H1 (deferred/async ordering)** |
| ZERO or stale | FINITE | same window, non-fatal in this instance — still a real ordering defect and evidence for H1 |
| FINITE | FINITE | this writeback is clean → if the arm faulted, the faulting buffer is not this route; move to the next route |

## Procedure

**Tension to resolve first, not assume away.** The issue records a hard negative that the logits are
*not* host-staged, while also naming the direct download copy as the route to instrument. H1 therefore
applies to whichever copy actually carries the logits buffer; if that is a different path (for example
the dmabuf mapping read directly by the sampler rather than through a writeback), this instrumentation
shows row 4 and the plan moves to that path. Establishing which copy carries the logits buffer is the
first step of task-2, not an assumption of it.

1. Build with the flag support; confirm the flag off ⇒ **byte-identical behaviour on one arm**: run
   `315-instr-capture.sh <instr_bin> … 0` (flag off) and require (a) the request phase completes as
   before, (b) **zero** `[hrx-wb-fp]` lines — the diagnostic is inert — and (c) both faces silent.
   Then run the same arm with the flag on and require the records to appear. An instrumented binary
   that logs with the flag off, or stays silent with it on, fails this step.
2. Run in-regime arms with the flag on (same regime gate as the pre-registration, ≥95 GiB).
3. Capture the record for a faulting event and for at least one clean arm.
4. The mechanism claim follows only if row 1 is observed on a faulting arm **and** the fingerprint is
   stable on clean arms; anything else is reported as the table's corresponding reading.

No configuration knob is a fix: the recorded studies already show the arena knobs do not move the
rate, so the fix must follow from the mechanism, not from a toggle.

## Artifacts

- `docs/315-instrumentation.patch` — the env-gated diagnostic as a unified diff against
  `ggml/src/ggml-hrx/runtime/command-program-executor.{cpp,h}` at
  `e44c9d01a4d5110cecaf45472a717f46cd1abbf9`. `git apply --check` against the pinned originals passes.
- Fingerprint self-test (off-box, `g++ -std=c++17`): ZERO, FINITE, NONFINITE-f32, NONFINITE-f16 and
  EMPTY all classify correctly, and a single flipped bit changes the hash. The helper block therefore
  compiles and its classification is fixed before any arm is run.
- Binary marker for the build check: the string `[hrx-wb-fp]` must appear in `libggml-hrx.so`, in the
  same spirit as `build-3971.sh`'s existing binary-level assertions.
- Build recipe (from `~/wt/build-3971.sh`, the study's own): `-DCMAKE_BUILD_TYPE=Release`,
  `CMAKE_C/CXX_COMPILER=/opt/rocm-therock/bin/amdclang{,++}`, `-DGGML_HIP=OFF -DGGML_HRX=ON`,
  `-DHRX_SOURCE_DIR=~/wt/pin-fork-hrx/third_party/hrx-system`, `-DLLAMA_BUILD_SERVER=ON`,
  `-DLLAMA_BUILD_TESTS=OFF -DBUILD_TESTING=OFF`, `-j16`.
