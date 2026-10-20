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

# What the synchronise drains, and the one candidate it eliminates

`315-read-twice-result-buffer-moves-under-the-reader.md` ends on a question: `result_norm` changes between
two consecutive identical reads even though the fingerprint synchronises first, so **what is writing it?**
This records the source reading, including the candidate it rules out — stated because a ruled-out
candidate is what keeps the next step from re-treading it.

## The backend has two streams; the synchronise drains one of them

`backend-context.h`:

```c
std::mutex                                     buffer_stream_mutex;
hrx_stream_t                                   buffer_stream = nullptr;   // :54
hrx_stream_t                                   stream;                    // :59
```

`ggml-hrx.cpp:506` — the primitive the fingerprint relies on:

```cpp
static void backend_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    static thread_local ggml::hrx::WaitHistory synchronize_wait_history;
    if (HRX_CHECK(ggml::hrx::stream_synchronize_sleeping(context->stream, synchronize_wait_history))) {
        context->graph_replay_state.mark_stream_synchronized();
    }
}
```

It drains **`context->stream` only**. `buffer_stream` is a separate stream and is not named here. On that
fact alone the natural candidate is: *the writer is on `buffer_stream`, which `backend_synchronize` never
covers.*

## That candidate is eliminated

`ggml-hrx.cpp:152-158` — the only path that submits on `buffer_stream` waits for it itself:

```cpp
static bool buffer_submit_and_wait(ggml_backend_hrx_device_context * device,
                                   hrx_status_t (*submit)(hrx_stream_t, void *),
                                   void * user_data) {
    std::lock_guard<std::mutex> lock(device->buffer_stream_mutex);
    if (!HRX_CHECK(submit(device->buffer_stream, user_data))) {
        return false;
    }
    return HRX_CHECK(hrx_stream_synchronize(device->buffer_stream));
}
```

The three uses of `buffer_stream` are this function's submit and the create/release at `:1193` and `:86-87`.
**Every submit on it is followed by its own `hrx_stream_synchronize` under `buffer_stream_mutex` before the
caller proceeds**, so a `buffer_stream` write cannot be in flight when `backend_synchronize` returns. The
two-stream asymmetry is real but it does **not** explain the observed race, and a future note should not
re-propose it without new evidence.

## What that leaves

The writer of `result_norm` during the host's read is therefore **not identified**, and what is established
is narrower than a mechanism:

* `backend_synchronize` orders the caller against `context->stream`;
* the `buffer_stream` path orders itself;
* and yet, in a faulting arm, two adjacent plain reads of `result_norm` returned different bytes 75 times
  out of 75.

Candidates that remain open, in the order the source suggests them:

1. **The graph executor's own return.** §14 records that `GraphExecutor::execute` has "no wait, fence or
   synchronize anywhere in its body" — so the sequence is `execute()` (returns without waiting) →
   `backend_synchronize()` (drains `stream`) → fingerprint reads. If any of the work the fingerprint reads
   is not on `stream`, that sequence does not order it. The `mark_stream_synchronized` call is the only
   evidence about what the sync believes it covered, and **what it is used to gate has not been traced here.**
2. **A second writer of the same arena slot** — `result_norm` is not its own allocation (there is no
   8192-byte host-visible buffer; see `315-f32-twin-result-divergence-on-the-faulting-arm.md`), so a later
   user of that slot is a candidate. §53/§59 already established the slot is reused.
3. **The replay path.** §4047348 traces it as upload → launch → flush → `wait_stream_commands`, which is a
   full wait, so this is the least likely of the three and is listed for completeness.

**Next measurement:** the `tail:16` + `GGML_HRX_BUFFER_TRACE` run already launched covers the logits node
by position and records the memory type. If its faulting arm shows `memcpy_stable=0` **on the logits**, the
same race reaches the tensor that actually carries the NaN, and locating the writer becomes the whole
question. If the logits records are absent or show `memcpy_stable=1`, this note's remaining candidates stay
open and the divergence remains a `result_norm` fact.
