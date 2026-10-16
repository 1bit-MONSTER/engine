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

# The coherence candidate is FALSIFIED by the decompiled runtime — and it explains the record's refuted A/B

Found in the **Ghidra RE dump at `~/amd-oss/hrx-re`** (the lane that was loading the box), which decompiles
`libhrx`, `LoomC` and the `amdhip64`/HIP binding. Reading `hrx_allocator_allocate_buffer` and its callees
settles the premise I could not resolve from source, and it settles it **against** the candidate.

## The type values, read from the headers (not recalled)

    runtime/src/iree/hal/buffer.h:45-112
      OPTIMAL        = 1u << 0 = 0x01
      HOST_VISIBLE   = 1u << 1 = 0x02
      HOST_COHERENT  = 1u << 2 = 0x04
      HOST_CACHED    = 1u << 3 = 0x08
      DEVICE_VISIBLE = 1u << 4 = 0x10
      HOST_LOCAL     = HOST_VISIBLE | (1u << 6) = 0x42
      DEVICE_LOCAL   = DEVICE_VISIBLE | (1u << 5) = 0x30
      DEVICE_UNCACHED= 1u << 7 = 0x80

and the HRX types are bit-identical to these (a wall of `static_assert`s in `hrx_internal.h`).

## What the decompiled allocator does

    hrx_allocator_allocate_buffer(allocator, params, size, buffer)          [decomp/libhrx/]
      hrx_mem_pool_backing_retain_allocation_source(device, params.type, params.usage, &pool)
      iree_hal_pool_allocate_buffer(pool, request, …)
      hrx_buffer_create_from_hal(hal_buffer, device, params.type, size, NULL, buffer)

and the pool selector:

    hrx_mem_pool_backing_retain_allocation_source(device, memory_type, usage, out_pool)
      kind = HRX_MEM_POOL_SOURCE_HOST;
      if ((~memory_type & 0x42) != 0) { kind = memory_type >> 7 & …; }      // 0x42 == HOST_LOCAL
      …
      hrx_mem_pool_backing_create_source(device, kind, usage, out_pool);    // no memory_type argument

**`0x42` is exactly `HOST_LOCAL`, and `HOST_COHERENT` is `0x04` — which is not in that mask, and is not passed
to pool creation either.** So:

| question | answer from the decompilation |
|---|---|
| does the coherence bit change **which pool** backs the buffer? | **No.** The kind is decided by `HOST_LOCAL` (0x42) alone; `0x42 & 0x04 == 0` |
| does the pool get the memory type at all? | **No** — `create_source(device, kind, usage, …)` takes kind and usage only |
| does the buffer record it? | Yes: `hrx_buffer_create_from_hal(…, mem_type, …)` stores the requested type |
| is it consulted in `query_mapping_requirements`? | **No** — that also routes through the same pool selector |

And the staging request is `HOST_LOCAL | DEVICE_VISIBLE = 0x52` — **precisely the type the record measured for
the logits buffer**, and the type the campaign's arm B′ was going to OR `0x04` into.

## Consequence 1: my coherence fix is a no-op, and I stopped the campaign

`315ab` (A′ vs B′, differing only in `GGML_HRX_STAGING_COHERENT`) was **killed** rather than allowed to consume
box time: the bit cannot change the pool, so both arms would have allocated from the **same** pool with the
same usage. A null was structurally guaranteed, and a null would have read as evidence about coherence.

**Honest limit, stated because it is the difference between "no-op" and "proven inert":** I could not read the
VMM map path (`hrx_allocator_virtual_memory_map` decompiled to nothing in this dump). A downstream mapping step
could in principle branch on the type recorded on the buffer. What is established is that the **backing pool —
which determines the memory — is chosen without reference to the coherence bit**, so the mechanism I proposed
(a different memory provider for a coherent request) is dead.

## Consequence 2: it explains the record's refuted coherence A/B

The campaign's earlier coherence experiment toggled `GGML_HRX_USE_UNIFIED_MEMORY` and **both arms faulted**,
which was read as coherence being irrelevant. The decompilation gives the structural reason:

    ggml-hrx.cpp:313-314   if (direct_host_binding) { memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT; }
    ggml-hrx.cpp:1190      device->use_direct_host_bindings = environment_flag_enabled("GGML_HRX_USE_UNIFIED_MEMORY");

That knob changes only the **type recorded on the buffer** (`0x52` → `0x56`). By the analysis above it cannot
change the pool. **So the knob could not change the memory, and both arms faulting is exactly what that
implies.** A negative result that has sat in the record as "coherence does not matter" is, on this evidence,
better read as *"the knob could not test coherence"* — the same failure mode as the `tail:16`-yield trap and
the disjoint-instrument-trees defect: **the intervention never reached the thing it was named for.**

## What this leaves standing

- **The 256-byte granularity** — 96.6% of partial readings begin on a 64-element boundary, ≥89.5% within
  **every** tensor family, against 1.6% chance. Unaffected by any of this.
- **The two remaining interpretations** of it: one **wave64** (64 lanes × 4 B = 256 B exactly) and the
  record's **MoE-table block** alignment (independent support: the FRONT group is MoE-dominated). The
  copy/region reading was already weakened (no 256-**byte** copy granularity in libhrx).
- **The wave mask**, built and pre-committed, which decides region-vs-lane in one run.
- **The coherence/staging family is now closed** as the explanation, on decompiled evidence, unless a
  map-path effect is found — and that is a specific, cheap next read, not an open-ended search.

---

# SCOPE CORRECTION: the falsification covers the POOL path only — the REGISTER path has explicit coherence, and its decider is missing from the dump

The section above concludes "the coherence/staging family is closed". Reading further into the same dump
narrows that: it is closed for **one** of the two host-memory paths and **open for the other**, and the
function that would close the second is one of the dump's gaps.

## There are two host-memory paths, and they treat coherence differently

| path | how memory is obtained | is coherence consulted? |
|---|---|---|
| **pool** (what the ggml-hrx staging buffer uses) | `hrx_allocator_allocate_buffer` → `hrx_mem_pool_backing_retain_allocation_source` → pool chosen by `HOST_LOCAL` (**0x42**); `create_source(device, kind, usage)` never sees the type | **no** — this is the falsification, and it stands |
| **register / import** (what `hipHostRegister` and managed buffers use) | `iree_hal_streaming_memory_register_host(context, ptr, size, flags, &buffer)` | **yes** — there are explicit flags |

## The register path's coherence flags, read from the headers

    libhrx/src/binding/common/internal.h:1089-1107
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT          = 0
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_READ_ONLY        = 1ull << 3   (0x8)
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_UNCACHED     = 1ull << 28
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_COHERENT     = 1ull << 30
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_NON_COHERENT = 1ull << 31

and the decompiled `hipHostRegister` (which decompiled **OK**):

    hipHostRegister(ptr, size, flags)
      flags_00 = flags | FLAG_READ_ONLY;
      if ((flags & 8) == 0) {                       // 8 == hipHostRegisterReadOnly
        flags_00 = flags & (HIP_NON_COHERENT | HIP_COHERENT | HIP_NUMA_USER | HIP_UNCACHED |
                            WRITE_COMBINED | MAPPED | PORTABLE);        // <- overwrites the READ_ONLY default
      }
      status = iree_hal_streaming_memory_register_host(context, ptr, size, flags_00, &buffer);

## What that means for the #315 trigger specifically

The forcer `~/wt/evictonly.hip` registers with **`hipHostRegisterDefault`, i.e. `flags = 0`**:

    flags_00 = 0 | READ_ONLY = 0x8   ->  then (flags & 8) == 0 is TRUE, so overwritten by  0 & (...) = 0

so it passes **`FLAG_DEFAULT` = 0 — neither `HIP_COHERENT` (1<<30) nor `HIP_NON_COHERENT` (1<<31) set.**
**The registration is coherence-unspecified**, and `iree_hal_streaming_memory_host_flags` (decompiled OK)
likewise *initialises* `*out_flags = FLAG_DEFAULT` before overwriting it from the buffer.

**So the behaviour of registered host memory under the #315 trigger is decided by what
`iree_hal_streaming_memory_register_host` does with an unspecified coherence flag — and that function is
ABSENT from the dump** (no JSON, not among the failures either, so it was never captured).

That is now the single most valuable missing function in the dump, ahead of
`hrx_allocator_virtual_memory_map`: the trigger registers with flags = 0, so *its own memory's coherence is
whatever that function's default is*. If it defaults to **non-coherent**, the eviction churn in the trigger is
operating on non-coherent registered memory — which is exactly the scenario the record's coherence hypotheses
were reaching for — and the earlier "coherence does not matter" reading would be wrong for a second,
independent reason.

## Corrected standing of the coherence family

- **Closed:** the **pool** path cannot be influenced by a coherence bit (the falsification; the fix I had
  prepared is a no-op).
- **Open:** the **register/import** path distinguishes coherence explicitly, the trigger registers with the
  flag *unset*, and the deciding function is not in the dump. So the family is **not** closed — it is closed
  *for the buffer I had proposed to patch* and open *for the path the trigger actually uses*.

**Nothing here is a mechanism claim.** It is a precise statement of which path is decided, which is not, and
which one function would decide the rest.
