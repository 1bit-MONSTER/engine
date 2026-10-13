<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The read fix addressed 1 of 4 host accesses — the write side is unaddressed

Written while arm B (`315ab1b2`) runs, triggered by its refutation of the read-path cause. If the reads
were not the source, the natural question is what else touches the same non-coherent mapping from the
host. Reading `ggml-hrx.cpp` at the fix's own tree answers it: **four** host accesses go through the
mapped `context->base`, and the prepared fix touched only one.

## The four accesses (ggml/src/ggml-hrx/ggml-hrx.cpp)

    line 210  buffer_memset   memset(context->base + destination_offset, value, size)   WRITE
    line 231  buffer_set      memcpy(context->base + destination_offset, data, size)   WRITE
    line 262  buffer_get      memcpy(data, context->base + source_offset, size)        READ  <- the fix
    line 298  buffer_clear     memset(context->base, value, buffer->size)              WRITE

Each is guarded by the same `context->base != GGML_HRX_FAKE_PTR_BASE` test, i.e. each is the
"buffer is mmap'd" branch. The mapping is the same non-coherent `HOST_LOCAL | DEVICE_VISIBLE = 0x52`
allocation the mechanism chain names (no `HOST_COHERENT`).

So three of the four accesses are **host writes into a non-coherent mapping**, and the prepared patch
does not touch any of them.

## Why this is a candidate cause rather than a curiosity

A CPU write into a non-coherent mapping can sit in the CPU's cache and never reach device-visible
memory. If the device later reads that region, it sees stale contents — which is the same *shape* of
fault as the read-path story, one direction over, and it predicts the same two faces (device consumes a
stale value -> NaN accumulation, or a stale/partial token buffer -> malformed output).

It also explains the read fix's failure exactly: routing the *read* through a coherent transfer cannot
repair a stale *write*, because the device never received the correct bytes to begin with.

## The counterpart API exists

The runtime exposes the write-direction call the fix's read used, in the opposite direction:

    hrx_synchronous_h2d(hrx_device_t device, const void* host_src, hrx_buffer_t dst,
                        size_t dst_offset, size_t size);           # hrx_runtime.h:636

and the read one it mirrors:

    hrx_synchronous_d2h(hrx_device_t device, hrx_buffer_t src, size_t src_offset,
                        void* host_dst, size_t size);              # hrx_runtime.h:640

There is still **no** CPU cache-maintenance API (`hrx_runtime.h` has only `hrx_stream_flush`, which is
command-buffer semantics, lines 576-582) — the same finding that made the read fix a queued transfer
rather than an invalidate. So if the write side is the site, the repair has the same shape: route the
write through `hrx_synchronous_h2d` instead of the raw `memcpy`/`memset`.

## What this is NOT

- Not a named cause. It is a hypothesis with a named site, an existing API, and a test that is cheap on
  the eviction rig (~5 arms per cell, ~20 min) — the same standing the read hypothesis had before it
  was tested.
- Not yet discriminated. The decisive check is the same A/B form: one guarded knob routing the writes
  through `hrx_synchronous_h2d`, run on the eviction rig, read against cell `e`.
- Not established to be on the logits path during decode, and the first check **weakened** it:
  `src/llama-context.cpp:2186` does call `ggml_backend_buffer_clear(buf_output.get(), 0)` on the output
  (logits) buffer, but it sits inside the output-buffer construction block, right after
  `buf_output.reset(ggml_backend_buft_alloc_buffer(...))` — i.e. it runs **once at allocation**, not
  per token. So the logits buffer is not host-written every decode, and the naive "stale write per
  token" story does not hold for it.
  The hypothesis survives only in a narrower form: it would matter if the output buffer is
  **reallocated mid-run** (a `n_outputs`/batch change), which the same block implies can happen. That
  is checkable — does `buf_output` ever reallocate during a GLM decode? — and until it is checked this
  remains a lead, not a site.
- Not the only remaining write, either. `buffer_set` (line 231) is on the load/`ggml_backend_tensor_set`
  path (`llama-context.cpp:2617, 2699`), which is where a host write of weights into a non-coherent
  mapping would be most damaging and most persistent — a different, arguably stronger, place to look
  than the output buffer.

## Citations

Read in the working tree `~/wt/llama-315coherent` and `~/wt/validate-329/third_party/hrx-system` on
strixhalo. The engine's pinned `llama.cpp` revision must be re-checked before these line numbers are
used in KNOWN-ISSUES; the load-bearing facts (four host accesses, one fixed; `hrx_synchronous_h2d`
exists) are structural and should be re-read at the pin.
