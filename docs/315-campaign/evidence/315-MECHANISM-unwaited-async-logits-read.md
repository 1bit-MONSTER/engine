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

# THE MECHANISM: the logits read is asynchronous and nothing waits for it before the sampler reads

**Named, falsifiable, and source-complete.** `llama_context::decode` obtains the logits through an
asynchronous read that this backend never completes, and the decode's own completion wait is commented out
on a premise that is false here — so the sampler reads the destination while the copy is still in flight.

## The four links, each at file:line

**1. The logits are extracted asynchronously.** `src/llama-context.cpp:1483-1489`:

```cpp
    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        ...
        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }
```

**2. The generic layer's wait is skipped for exactly this backend.** `ggml/src/ggml-backend.cpp:268-280`:

```cpp
void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ...
    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);          // <-- the wait that would have made this safe
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}
```

HRX **does** implement the async iface (registered at `ggml-hrx.cpp:532`), so the `synchronize` branch is not
taken.

**3. HRX's implementation enqueues and returns.** `ggml-hrx.cpp:454-473`:

```cpp
static void backend_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * backend_context = static_cast<ggml_backend_hrx_context *>(backend->context);
    ...
    ggml::hrx::HostBufferRef destination = backend_context->device->host_buffers.find(data, size);
    if (destination.valid()) {
        // Registered host buffers can participate directly in the stream command buffer.
        HRX_CHECK(hrx_stream_copy_buffer(backend_context->stream, context->buffer, tensor_base + offset,
                                         destination.buffer(), destination.offset(), size));
    } else {
        synchronous_download_fallback(...);
    }
}
```

No wait on either branch. **And the registered-buffer branch is the one taken in practice** — measured, not
assumed: the read probe logged the engine's logits read as
`name=result_output bytes=619520 host_dst=0x7fed46100000 registered=1`.

**4. Nothing else in `decode` waits on the logits path.** `src/llama-context.cpp`, the end of
`llama_context::decode` (1707-2091):

```cpp
    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}
```

The comment states the premise — *obtaining the model output waits* — and it is false here, because
obtaining the output is links 1-3. The two other `synchronize()` calls near the end of `decode` do not
cover this path: `:1784` is guarded by `if (!embd_seq.empty())` (embeddings, not logits), and `:1560` is in
**`encode`** (1399-1707), not `decode`, and is T5-specific.

## The mechanism

> On the logits path, `llama_decode` returns with the logits copy — and possibly the graph that produces
> them — **still in flight**. The sampler then reads `logits.data`, which for HRX is a registered host
> destination at a fixed virtual address. What it reads is whatever the copy has published so far: the
> **previous** token's logits (finite, plausible, wrong) or a region **not yet written**. The first is
> Face B; the second is Face A.

## Why this fits every hard fact, including the ones that killed earlier candidates

| fact on the record | how this explains it |
|---|---|
| **Face A: `count=154880 of 154880`, `contiguous=yes`** | the destination read before the copy published anything — an untouched region is uniformly wrong |
| **Face B: finite, plausible, wrong text → HTTP 500** | the previous token's logits: finite and well-formed, so the parser rejects the *body*, not the numbers |
| **intermittency** | whether the copy has completed by the time the sampler reads is a scheduling outcome |
| **coherence is not the carrier** (§13: `hv56=1` and still faulted) | correct and expected — a missing **wait** is not a visibility property |
| **the §15 sync cell did not fix it** | the sync was placed at the end of `graph_compute`, which is **before** the logits copy is issued; it cannot cover the copy it precedes |
| **"ordering the reads changes nothing"** (§59) | that ordered the **fingerprint's** reads, not the engine's logits read |
| **the f32 twin's `memcpy_stable=0` on `result_norm`** | `result_norm` is the input to the logits `MUL_MAT`; the graph is still running during the fingerprint's reads, so consecutive reads of it differ — **independent of memory type** (cells A and B: `0x52` and `0x56`, rates 13.7% and 14.8%), and **not** the deferred host writeback (626 reads, 0 pending) |
| **Face A's damage is whole-vocabulary, not a row** | the read is of the logits *destination*, not of any layer tensor, so a bad read is not confined to one row |

## The fix, and why it is forward-only

Restore the wait on the path that needs it — **before the sampler reads the logits**. The minimal
forward-only change is the engine-side line the comment disabled, or its equivalent:

```cpp
    // wait for the computation to finish (automatically done when obtaining the model output)
    synchronize();
```

This is an engine-side change in `src/llama-context.cpp` (the engine's own tree, not a revert of any pinned
revision), and it is unaffected by §15's null result because §15 synchronized at a point that precedes the
copy it needs to cover.

## The discriminating experiment, pre-declared

**Prediction:** with the wait restored on the logits path, in-regime faults go to zero on both faces — and
without it they do not. A stronger, *instrument-level* confirmation that does not depend on the fix:
**measure whether the copy is complete at the moment the sampler reads the destination.** If the
destination is verifiably complete at every read in a faulting arm, this mechanism is dead.

**Falsifier:** a faulting arm in which the logits destination is confirmed complete before the sampler
reads it.

## What is NOT yet established, stated plainly

* The mechanism is read from source and is consistent with every measurement, but **the fix has not been
  run**. Until it is, this is a named candidate with a discriminating experiment, not a demonstrated cause.
* Nothing here shows the *graph* is unfinished as well as the copy; the twin's divergence on `result_norm`
  is consistent with the graph running, but the logits-specific claim needs only the copy to be unwaited.
* §14 recorded the `*_async` read and that the sampling path does not synchronise; what is new here is
  the **commented-out wait at `:2087`** whose comment asserts the opposite, and the measurement that the
  registered-buffer branch (no wait at all) is the one taken.
