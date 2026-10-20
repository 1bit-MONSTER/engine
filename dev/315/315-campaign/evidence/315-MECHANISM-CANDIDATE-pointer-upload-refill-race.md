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
<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# MECHANISM CANDIDATE (source-grounded): a per-step graph input is uploaded by POINTER and the engine refills that buffer before the queued update commands execute

The record's own code documents the hazard class, guards one direction of it, and leaves the other direction
open. This is the strongest source-grounded candidate on the record and it is falsifiable with one instrumented
arm.

## The upload path, read from the fork

`runtime/host-memory.cpp`:

    kLargeHostUploadBytes = 1024 * 1024          // >=  1 MiB  -> upload_synchronous (blocking, snapshots nothing: it copies)
    kMaxInlineUploadBytes = 63 * 1024            // <   1 MiB  -> upload_async, chunked at 63 KiB

    Status HostTransferManager::upload_async(stream, host_source, destination, offset, size) {
      if (size >= kLargeHostUploadBytes) return upload_synchronous(...);
      const uint8_t* host_bytes = static_cast<const uint8_t*>(host_source);
      size_t uploaded = 0;
      while (uploaded < size) {
        const size_t chunk_size = min(size - uploaded, kMaxInlineUploadBytes);
        hrx_stream_update_buffer(stream, host_bytes + uploaded, chunk_size, destination, offset + uploaded);
        uploaded += chunk_size;
      }
      return status;                              // NOTE: returns immediately
    }

and `runtime/command-program-executor.cpp:905`:

    context.host_transfers->upload_async(context.stream, staging.host_data, staging.buffer, 0, staging.length);

**`upload_async` queues commands that reference `staging.host_data` by pointer and returns.** The device reads
that host memory when the stream reaches the update command — not when the call returns. So `staging.host_data`
must remain unchanged until the stream has executed those updates.

## The hazard, in the authors' own words

Immediately above that call:

    // A host download staged by an earlier replay is only published to host memory by
    // mark_stream_synchronized(); add_host_writeback merely queues the memcpy. Uploading the
    // same host memory before that has happened would read the previous contents. The split
    // scheduler computes splits back to back and only synchronizes a backend when it inserts
    // a cross-backend input copy, so nothing else guarantees the ordering.
    if (context.graph_replay_state != nullptr && context.graph_replay_state->has_pending_host_writebacks()) {
      stream_synchronize_sleeping(context.stream, writeback_wait_history);
      context.graph_replay_state->mark_stream_synchronized();
    }

**This is exactly the right hazard** — host memory being read before its content is what the reader expects —
**and the guard covers one direction: a pending *writeback* (download) has not yet published into host memory,
so an upload of the same memory would read the previous contents.** That is handled.

**The opposite direction is not guarded.** If the engine writes a *new* input into `staging.host_data` while a
previous `upload_async`'s queued `hrx_stream_update_buffer` commands have not yet executed, the device consumes
the **new** bytes for the **old** update. The comment says why nothing stops it: *"The split scheduler computes
splits back to back and only synchronizes a backend when it inserts a cross-backend input copy, so nothing else
guarantees the ordering."* **Refilling a staging host buffer is not a cross-backend input copy.**

## Why this fits every measured property of Face B

| measured fact | explained |
|---|---|
| finite logits, no NaN in any of 18 events | the device receives a **valid** value — another step's input — not poison |
| intermittent for identical input (3 bit-identical runs, 1 divergent) | a race between the engine refilling the buffer and the stream executing the update |
| forced eviction 15.1× the control hazard | anything that shifts engine-vs-stream timing widens the window |
| enters at a clean boundary, keeps its prefix | only the step whose input was torn is wrong |
| 500 only 2 of 18 | needs the wrong token to persist long enough for the parser to reject |
| index 0 never fails (0/2259) | the prefill's input is uploaded once and is not racing a refill |
| front-loaded over ~7 decode steps, 4× under eviction, **zero** early events in control/quiet | the first steps are exactly where the engine issues the next input while the previous uploads are still in flight |
| `graphs reused = 410` | replay is the mode in which a fixed staging host buffer is refilled most aggressively |

## Why the earlier candidates failed and this one does not

- **The read-path/visibility hypothesis** predicted a memcpy-vs-`hrx_synchronous_d2h` disagreement; the record
  has 253,744 paired readings across two instruments with **zero** disagreements. That hypothesis is contradicted
  by measurement, and this one does not need a read-path divergence: both reads see the same bytes, because the
  **wrong bytes were genuinely uploaded**.
- **The allocation-flag/coherence hypothesis** was closed from source: the flags never reach an allocation.
- **The snapshot-at-capture hypothesis** (`copy_source_into_graph = true`) is **refuted**: that function,
  `iree_hal_streaming_memcpy_value_to_device`, has **no callers anywhere**; every upload uses the by-reference
  variant, which is why the pointer matters at all.

## The discriminating experiment

**Instrument the host source buffer and the device destination at a torn step.**

- Record the content hash of `staging.host_data` **at the moment `upload_async` is called** (queue time), and
  record the content hash of the destination buffer on the device **after the stream has executed the updates**.
- **differ** → the device consumed bytes the engine did not intend — confirmed, and the fix is to make the
  staging buffer immutable until the stream passes the update (double-buffer it, or synchronize before refill);
- **agree** → the upload is faithful and the fault is downstream, and this candidate is refuted.

An even cheaper first form: print the hash at queue time and again at the *next* queue time; if they differ while
the stream has not been synchronized, the window is open, and the stream's own command count tells whether the
update had executed.

**No mechanism is claimed as verified.** What is claimed: the upload path is pointer-based and asynchronous, the
fork's own comment documents this exact hazard class, the guard covers only the writeback direction, and the
unguarded direction reproduces every measured property of Face B including the trigger's 15× dose-response and
the immunity of the prefill's own output.
