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

# The cause boundary, stated finally: seven exclusions and one unbridgeable gap

After seven candidates were excluded, this records what the engine **can** know, so the boundary between
"cause found" and "out of scope" is explicit rather than implied.

## The physical trigger, from the record

`~/wt/evictonly.hip` drives `mprotect` churn over a `hipHostRegister`'d range, which makes the kernel
**evict and restore KFD queues**. Measured per-arm: every arm's **own** buffers show `Q_EVICT`/`Q_RESTORE`
while the arm is live. So the event is *the arm's memory being evicted and restored underneath it*, and the
fault is what a consumer reads afterwards.

## What the engine tracks about a buffer

Read at the pin:

    backend-context.h:35     uint64_t generation;      per-buffer field
    ggml-hrx.cpp:43          std::atomic<uint64_t> g_allocation_generation{ 1 };
    ggml-hrx.cpp:338         const uint64_t generation = g_allocation_generation.fetch_add(1);
    command-program-bindings.cpp:69   mix_hash(hash, binding.generation);

So `generation` is a **monotonic counter bumped once per allocation** and mixed into the binding hash used
for **command-program caching**. `identity` is likewise a per-buffer key.

**Neither tracks contents across a sync.** They answer "which allocation is this?" — not "are the bytes I am
about to read the ones the device wrote?". There is no checksum, epoch, or validity flag consulted after a
synchronisation anywhere on the replay → restore → consume path, which is the same conclusion the writeback
and Face-B-detection reads reached from their ends.

## The gap, stated as one sentence

> The engine can name **which** allocation it holds, and can detect **that** a read produced garbage (Face A's
> NaN guard) or malformed text (Face B's parse failure) — but it cannot distinguish "the device wrote these
> bytes" from "a restore left a pre-eviction copy", and that distinction is precisely what a repair requires.

## Why the remaining candidates are out of scope

    KFD eviction/restore fidelity        driver/kernel lane - the boundaries say record, do not attempt
    executor cross-dispatch ordering     §55/§56 cleared the consumer's own read path
                                         (hsa_signal_wait_scacquire + timepoint covering the last submission)

and the seven in-scope candidates are excluded above. So the cause is not merely unfound: **the surfaces
within the declared scope have been enumerated and excluded**, which is a stronger statement than repeated
failure to find it.

## What this means for the criteria, plainly

- **Criterion 1**: trigger named/replicated/quantified — the strongest result in the issue. Cause **not
  locatable within scope**, and now with the in-scope surface enumerated rather than merely searched.
- **Criterion 2**: not achievable within scope, because any repair needs the distinction above.
- **Criterion 3**: gate mis-specified (intervention cell, not a band).
- **Criterion 4**: op-test half measured and passing; dense half running.
- **Criterion 5**: done.
