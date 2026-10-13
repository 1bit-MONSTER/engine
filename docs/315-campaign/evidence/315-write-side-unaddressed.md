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

---

# UPDATE (same turn) — the logits half of the write-side lead is FALSIFIED

Checked the "output buffer reallocated mid-run" form promised above, and it does not hold for this
campaign's shapes.

## The code

    uint32_t llama_context::output_reserve(int32_t n_outputs)   # llama-context.cpp:2096
        ...
        const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
        const size_t new_size  = (...) * sizeof(float) + (...) * sizeof(llama_token);
        // alloc only when more than the current capacity is required
        if (!buf_output || prev_size < new_size) {          # the branch holding the clear
            ...
            buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
            ...
            ggml_backend_buffer_clear(buf_output.get(), 0);  # line 2186
        }

`output_reserve` **is** on the decode path — it is called from `llama_decode` (`:1848`) and
`llama_encode` (`:1446`) on every call — but the allocation-and-clear branch is guarded by
`prev_size < new_size`. With the campaign's fixed shapes (`-c 8192`, `--no-webui`, identical request
size, `n_outputs_max` fixed at init) `new_size` does not grow, so the branch is taken on the first call
and never again.

**Therefore the logits/output buffer is not host-written per token**, and the "stale host write to the
logits mapping" story is falsified for this defect in this campaign.

## What survives

Only the `buffer_set` load-path form (below), which is a different claim: a host write of **weights or
other inputs** through `ggml_backend_tensor_set` (`llama-context.cpp:2617, 2699`) into a non-coherent
mapping. That is not falsified here and is not tested here. It is also a poor fit for the observed
behaviour on its own — a stale weight write would not be *intermittent* and eviction-driven the way the
fault is, since it happens once per load.

## Net position of this file

The write side remains architecturally relevant — three of the four host accesses are writes, against
one read that the fix addressed — but the specific "host write per decode to the logits buffer" form is
dead, and the surviving form (load-time weight writes) does not obviously explain an intermittent,
eviction-correlated fault. So the write side is **not** currently the leading explanation of arm B's
result; it is recorded because the access asymmetry is real and because the next hypothesis should not
have to rediscover it.

---

# CLOSED (2026-10-13) — the host-write path is not exercised during GLM decode either

The logits half was falsified earlier (`output_reserve`'s clear is once-at-allocation). This closes the
remaining half: are weights or other inputs host-written into the non-coherent mapping **per token**?

Enumerated every `ggml_backend_tensor_set` site in `src/llama-context.cpp` — there are exactly three:

    2617  llama_io_read_host::~llama_io_read_host()   flushes host reads into tensors   <- MODEL LOAD
    2699  llama_io_read_file::read_tensor()           file read -> temp buffer -> set   <- MODEL LOAD
    3432  training path (labels)                      sparse label write                <- TRAIN ONLY

The first two are the model-loading IO classes (`llama_io_read_host`, and the file reader); they run once
when the GGUF is loaded. The third is the training path, which `llama-server` inference never enters.

So **nothing host-writes a tensor on the GLM decode path**, and the non-coherent `context->base` mapping
is not written from the host between tokens. Combined with the earlier falsification of the logits
`buffer_clear` and of `buffer_get` as the read site (the A/B:

    cell e, arm A knob OFF 0 OK / 4 FAULT   vs   arm B knob ON 0 OK / 2 FAULT

both faces, knob verified live), the whole host-side access set is now accounted for:

    buffer_memset  (line 210)  WRITE   not called on the decode path
    buffer_set     (line 231)  WRITE   load-time only (2617, 2699) or train-only (3432)
    buffer_get     (line 262)  READ    the fix; REFUTED by its own A/B
    buffer_clear   (line 298)  WRITE   once at allocation, not per decode

## What this closes, and what it leaves

**Closed**: the host-side incoherence family. Neither direction of host access to the non-coherent
mapping is exercised per token, so no host-side cache-coherence defect can be producing a per-token,
eviction-correlated fault. This retires the lead this file proposed.

**Left**: the corruption is on the **device** side — something reads stale/inconsistent data from the
server's own evicted-then-restored device memory, which is where the trigger (forced eviction, 0/22 ->
7/10) already points. The surviving candidate families are the ones §55/§56 left open: executor-level
cross-dispatch ordering, or a different class entirely. Note that §55/§56 cleared the *consumer's own*
host read path via `hsa_signal_wait_scacquire` + timepoint, which is consistent with this: the answer is
expected to be device-side, not host-side.

## Honest note on the ledger

This file has now proposed a lead and closed it in the same session, after the logits half was falsified
in between. That is recorded rather than tidied away: the host-write idea was worth checking, the check
was cheap and decisive, and its result is that the answer is not here.
