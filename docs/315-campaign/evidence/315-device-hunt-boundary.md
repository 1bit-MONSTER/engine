<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Where the device-side hunt actually stops: the runtime does not infer memory hazards

Written after the host-side family was closed (`315-write-side-unaddressed.md`), to follow the one
ordering question §32/§33 left standing — "whether the host-side recorder actually records every
producer/consumer pair it should". Read at source, the runtime answers for itself, and the answer
narrows the remaining surface rather than opening it.

## The answer, in the code

`runtime/src/iree/hal/drivers/amdgpu/aql_block_processor.c`, in
`iree_hal_amdgpu_aql_block_processor_packet_control`:

    // The replay hot path is intentionally additive: command recording has
    // already encoded execution-barrier scopes in |packet_flags|, and submission
    // policy only overlays wait, queue-kernarg, and terminal-signal visibility.
    // Do not infer memory hazards here by walking command operands.

And the barrier-bit decision itself (`..._packet_has_barrier`) consumes only:

- `PACKET_FLAG_EXECUTION_BARRIER` and `PACKET_FLAG_FINAL` — **set by command recording**;
- `packet_index == 0` with a waiting submission, or with a non-NONE inline acquire scope — the
  submission-level overlay (`max_scope`, applied only at `logical_packet_index == 0`).

Nothing walks buffer operands. Nothing derives a producer/consumer pair. The runtime's stated policy is
that **hazards are already encoded by the compiler**, and the submission layer adds only wait,
queue-kernarg and terminal-signal visibility.

## Why this matters

It answers §33's standing question in the direction that **removes the runtime as the site**: the
"recorder may miss a producer/consumer pair" hypothesis requires the runtime to *derive* pairs, and it
explicitly does not. So the ordering between a global-memory producer and its consumer in GLM's stream is
whatever the compiler emitted, and a missing dependency would be **an emission gap, not a runtime gap**.

This is consistent with the two other device-side results already on the record:

- §33's static audit found no GLM-relevant kernel with an intra-dispatch global handoff lacking a
  `barrier<global>` — the kernels that hand work across waves through global memory are exactly the ones
  that carry the fence;
- §32's block-boundary asymmetry was refuted by its own step 1 (`packet_index_base` is a profiling
  prefix, 0 for every block, so the per-block conditions hold per block).

## What is left, stated as the boundary rather than a story

The remaining candidate surface is the **compiler's emission of execution-barrier scopes** for GLM's
stream (loom / IREE codegen), not the runtime and not the host. That surface is large and is *not* in this
goal's declared scope: the boundaries name the engine pin and the ggml-hrx fork side, not the loom
codegen. Pursuing it would be an internals expedition with no bounded discriminating test available at
the level this issue has been worked at.

So the honest position: the cause is **not located**, and the evidence now favours an
emission-level ordering gap that this goal's scope does not reach. That is a boundary to record, not a
mechanism to claim — and it is the reason criterion 2 should not be claimed as achievable here.

## What would change this

A bounded test would need to compare GLM's emitted barrier scopes against a model that does not fault
(e.g. a dense model of similar depth) and find a producer/consumer pair GLM omits. That is a compiler-side
experiment requiring codegen instrumentation, and it is outside the declared scope
(`~/wt` harness, engine pin, ggml-hrx fork).
