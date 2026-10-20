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

# Correction: the engine reads host-visible buffers through a PERSISTENT mapping, so `memcpy_stable=0` has a third explanation

Found while asking the question the falsified writeback candidate left behind: *if no queued writeback
populates the destination, what does `buffer_get` actually read?* The answer changes the reading of the
read-twice result.

## What `buffer_get` reads

`ggml-hrx.cpp:320-338`, at host-visible buffer allocation:

```cpp
uint8_t * base = reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE);
if (host_visible && size > 0) {
    void * mapped = nullptr;
    if (!HRX_CHECK(hrx_buffer_map(allocation, HRX_MAP_READ | HRX_MAP_WRITE, 0, size, &mapped))) { ... }
    base = static_cast<uint8_t *>(mapped);
}
```

and `buffer_get` (`:238`):

```cpp
if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
    std::memcpy(data, context->base + source_offset, size);
    return;
}
```

**The mapping is created once, at allocation, for the whole buffer, and is never re-mapped and never
invalidated.** Every engine read of a host-visible buffer is therefore a plain CPU `memcpy` out of a
*mapping established at allocation time*.

## Why that matters to the read-twice result

`315-read-twice-result-buffer-moves-under-the-reader.md` read `memcpy_stable=0` (two consecutive plain reads
of the same range returning different bytes, 75 of 75 in a faulting arm) as *"the buffer moved between two
identical reads"*, and its control was built on the dichotomy:

* `memcpy#1 == memcpy#2 != sync` ⇒ a stale host cache line (the path is the difference);
* `memcpy#1 != memcpy#2` ⇒ the buffer moved (the writer is the difference).

With a persistent mapping there is a **third** case that produces `memcpy_stable=0` and is neither: **the
CPU's cached view of a non-coherent mapping changing between the two reads**, because cache lines for that
mapping were evicted and re-fetched in between. Nothing device-side moves; the reader's own view does.

That is not a hypothetical here. The arm that produced the 75/75 was running **with the eviction forcer**
(`evictonly 500`, which mprotect-churns host memory) in the same cell as the earlier control — and eviction
pressure is precisely what changes which lines the CPU holds. So `memcpy_stable=0` is consistent with

1. the device writing the range between the reads, **or**
2. the CPU's cached view of a non-coherent mapping changing between the reads,

and the control as run **does not separate them.** My earlier note's stronger reading is corrected here.

## What is and is not re-opened

* **Re-opened, in a narrow form:** coherence, but as a property of the *persistent mapping*, not of the
  allocation flags that §13 tested. §13 set `HOST_COHERENT` on the allocation (`hv56=1`) and the arm still
  faulted — so a coherent allocation does not remove the fault. Whether a coherent allocation also removes
  the *mapping's* read-to-read divergence is a different question, and it is measurable: the same twin in a
  `0x56` cell against a `0x52` cell (`315-two-cell-memory-type-preregistration.md`, written earlier for
  exactly this comparison and still unrun).
* **Not re-opened:** the falsified writeback candidate stays falsified — that measurement was a *counter*,
  not a hash comparison, and is unaffected by this.
* **Still true:** §51's rank-sort precondition (NaN at the argsort's input) is unmeasured, and
  `result_output` — the tensor the NaN guard reports — has still never been read divergently in a faulting
  arm, only `result_norm`.

**The measurement this hands forward:** run the twin with `GGML_HRX_BUFFER_TRACE=1` in both memory-type
cells, so the divergence count is attached to a measured type rather than to an inferred one — the
comparison `315-two-cell-memory-type-preregistration.md` pre-registered.
