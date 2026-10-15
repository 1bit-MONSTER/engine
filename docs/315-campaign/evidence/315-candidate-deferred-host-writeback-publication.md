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

# MECHANISM CANDIDATE (source-grounded): a device→host download is a DEFERRED CPU memcpy, published only by `mark_stream_synchronized()`

Found while tracing what could write `result_norm` between two adjacent host reads, after
`315-read-twice-result-buffer-moves-under-the-reader.md` established that it does (75/75 records
`memcpy_stable=0` in a faulting arm). The mechanism is stated **by the code's own comment**, which is what
makes it worth naming rather than inferring.

## The download is not a device operation

`runtime/command-program-executor.h:39-62` — the state that stages downloads:

```cpp
struct GraphReplayStreamState {
    void mark_stream_synchronized();
    void add_host_writeback(void * host_destination, const void * mapped_source, size_t size, hrx_buffer_t buffer);
    // True while a host download staged by an earlier replay has not been
    // published to host memory yet (see mark_stream_synchronized).
    bool has_pending_host_writebacks() const { return !pending_host_writebacks_.empty(); }
  private:
    std::vector<PendingHostWriteback> pending_host_writebacks_;
};
```

`runtime/command-program-executor.cpp` — the two operations:

```cpp
void GraphReplayStreamState::add_host_writeback(void * host_destination, const void * mapped_source,
                                                size_t size, hrx_buffer_t buffer) {
    if (buffer != nullptr) { hrx_buffer_retain(buffer); }
    pending_host_writebacks_.push_back({ host_destination, mapped_source, size, buffer });   // QUEUES ONLY
}

void GraphReplayStreamState::mark_stream_synchronized() {
    for (PendingHostWriteback & writeback : pending_host_writebacks_) {
        if (writeback.host_destination != nullptr && writeback.mapped_source != nullptr && writeback.size > 0) {
            std::memcpy(writeback.host_destination, writeback.mapped_source, writeback.size);  // PUBLISHES
        }
        ...
    }
    pending_host_writebacks_.clear();
}
```

**A device→host download is a plain CPU `memcpy` from the mapped device-visible source into the host
destination, and it does not run when it is staged — it runs when the stream is next marked synchronised.**

## The hazard, in the code's own words

`command-program-executor.cpp:873-882`, guarding the host-staging **upload** path:

> A host download staged by an earlier replay is only published to host memory by
> `mark_stream_synchronized()`; `add_host_writeback` merely queues the memcpy. Uploading the same host memory
> before that has happened would **read the previous contents**. The split scheduler computes splits back to
> back and only synchronizes a backend when it inserts a cross-backend input copy, so **nothing else
> guarantees the ordering**.

So the codebase (a) knows the destination holds the **previous** contents until publication, and (b) has
written a guard for **exactly one direction** — an *upload* must not read the destination before a pending
download is published. There is no equivalent guard recorded on the **read** side.

## Why this fits every hard fact

| fact on the record | how this explains it |
|---|---|
| **two adjacent plain reads of the same host buffer differ** (`memcpy_stable=0`, 75/75) | the publication `std::memcpy` runs *between* them; there is no cache involved |
| **Face A: whole-vocabulary NaN** (`count=154880 of 154880`) | the reader gets the destination **before** publication — the previous contents, or an unwritten region |
| **Face B: finite but wrong text** | the previous contents are the *previous token's* logits: finite, plausible, wrong → the parser rejects the body |
| **coherence is not the carrier** (lever 1, §13: `hv56=1` and still faulted) | the transfer is a CPU memcpy; making the memory coherent changes nothing about whether it has run |
| **ordering the reads changes nothing** (§59) | a stream/device ordering primitive does not publish a queued CPU memcpy either |
| **the sync cell did not fix it** (§15) | `backend_synchronize` *does* call `mark_stream_synchronized` (`ggml-hrx.cpp:510`), so a sync at the right point publishes — the question is whether the **reader's** path passes through one at the right time, which §15's placement does not settle |
| **intermittency, and per-token dependence** | whether the reader beats the publication is a scheduling outcome |
| **§14's own correction**: "a copy issued through the buffer stream is invisible to them" | the download is not a device copy at all; instruments watching device transfers would not see it |

## The discriminating experiment (proposed, not yet run)

The mechanism makes one prediction no surviving alternative makes:

> **At the moment the engine reads the logits, a `PendingHostWriteback` for that destination is still
> queued.** In a faulting arm `has_pending_host_writebacks()` is true at the read; in a clean arm it is
> false or was already published.

Concretely: log `graph_replay_state->has_pending_host_writebacks()` (and the destination range) at the
`ggml_backend_tensor_get` for `result_output`, and correlate with the fault. This is a **one-line observation
plus a counter**, not a fix, and it does not require a new transfer path.

**Falsifier:** if `has_pending_host_writebacks()` is false on every read in a faulting arm, this candidate is
dead and the writer is something else (see `315-what-the-synchronise-drains.md` for the remaining list).

## Honest bounds

* This is a **candidate read at source**, not yet measured. Nothing here shows a pending writeback actually
  exists at a read in a real arm.
* It does not yet name the *reader* that loses the race. The engine's sampling-path read is the natural
  candidate (§14 records that the sampling path does not synchronise; `llama-context.cpp:1488` issues the
  logits copy with `*_async` and returns), but that has not been traced to this queue.
* It does not claim the hazard is unintended: the upload-side guard shows the authors consider the ordering
  real and worth guarding; whether the read side is intended to be covered elsewhere is not established here.
* `mark_stream_synchronized` also being called at `ggml-hrx.cpp:510` and `graph-program-cache-limit.cpp:56`
  means publication can happen at several points; which of them precedes a given read is the measurement,
  not an assumption.
