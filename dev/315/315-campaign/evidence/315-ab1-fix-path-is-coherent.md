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

# The fix's mechanism was sound — so the refutation is informative, not trivial

Written while arm B (`315ab1b2`) is still running, in response to its first result (round 1 cell `e`
FAULT, Face B). It checks the one objection that would make the refutation *uninformative*: that
`hrx_synchronous_d2h` might itself end in an incoherent CPU read, in which case arm B's fault would
prove nothing about the corruption site.

It does not. The transfer path is coherent end to end, so the refutation stands as evidence.

## The chain, read at source

1. `buffer_get`'s fix calls `hrx_synchronous_d2h`
   (`third_party/llama.cpp` at the pin, `ggml/src/ggml-hrx/ggml-hrx.cpp`);
2. `hrx_synchronous_d2h` issues a **real HAL download**, not a host read —
   `third_party/hrx-system/libhrx/src/libhrx/transfer.c:118-148` builds
   `IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD` and calls `hrx_hal_queue_transfer_and_wait`, returning
   before any host-side copy;
3. the amdgpu host queue turns that into a **staging** download —
   `runtime/src/iree/hal/drivers/amdgpu/host_queue_transfer.c:224-231` calls
   `iree_hal_amdgpu_staging_transfer_create_download` for the DOWNLOAD case;
4. the staging pool the download lands in is created **HOST_COHERENT** —
   `runtime/src/iree/hal/drivers/amdgpu/host_queue_staging.c:427`:

       IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
           IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE

So the destination of the fixed read is coherent host memory, and the device→host leg is a queued
transfer rather than a raw pointer dereference. The fix does what it claims: it removes the raw read of
the non-coherent `0x52` logits mapping.

## Why that matters

The one way arm B's fault could have been dismissed was "the fix didn't actually change coherence, so
its failure proves nothing." The source rules that out. Therefore arm B's cell-`e` fault (if the
remaining rounds confirm it) is genuine evidence that **fixing the coherence of the logits read does
not stop the fault** — the corruption is not the CPU observing a stale logits page.

It also rules out the simple inverse reading ("the read path was fine all along, so `0x52` was
harmless"): the allocation genuinely lacks `HOST_COHERENT` while the fix's path genuinely supplies a
coherent one. The difference is real; it just is not where the fault comes from.

## Scope

This file cites the runtime at `~/wt/validate-329/third_party/hrx-system` on strixhalo, which is a
working tree, not the engine's pinned submodule revision. The pin must be re-checked at its own
revision before these line numbers are used in KNOWN-ISSUES; the *fact* that the staging pool is
created `HOST_COHERENT` is the load-bearing claim, and it should be re-read at the pin when the reading
is finalized.
