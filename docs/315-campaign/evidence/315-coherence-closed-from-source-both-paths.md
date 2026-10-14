<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Coherence is CLOSED, definitively and from source: this runtime cannot make host memory coherent

Following the register path to the end (the source is in the tree, so no decompilation was needed) settles the
question the record has carried for weeks, and settles it in the strongest form yet.

## Both host-memory paths ignore coherence, and one of them hardcodes it away

**Path 1 — pool** (the ggml-hrx staging buffer):

    hrx_mem_pool_backing_retain_allocation_source(device, memory_type, usage, &pool)
      kind = HRX_MEM_POOL_SOURCE_HOST;  if ((~memory_type & 0x42) != 0) { … }     // 0x42 == HOST_LOCAL
      hrx_mem_pool_backing_create_source(device, kind, usage, out_pool);          // no memory_type argument

`HOST_LOCAL = 0x42`, `HOST_COHERENT = 0x04`; `0x42 & 0x04 == 0`. The bit is not consulted, and the pool never
receives the type at all.

**Path 2 — register/import** (`hipHostRegister`, managed buffers) — and this is the one the #315 trigger uses:

    memory.c:2510-2530   iree_hal_streaming_memory_register_host_with_context_mode(…)
      iree_hal_buffer_params_t params = {
          .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,   // = 0x52, no 0x04
      };
      iree_hal_allocator_import_buffer(context->device_allocator, params, &external_buffer, …);
      …
      wrapper->host_register_flags = flags;        // the computed coherence flags: STORED, not used

**The type is hardcoded.** Whatever `flags` arrive — `HIP_COHERENT`, `HIP_NON_COHERENT`, or nothing — the
imported buffer is always `0x52`.

## The flags are pure bookkeeping

Every consumer of `host_register_flags` in libhrx, with the assignment lines excluded:

| site | what it does |
|---|---|
| `memory.c:755` | `out_ref->host_register_flags = owner_wrapper->host_register_flags;` — copies it into a ref |
| `memory.c:2667` | `*out_flags = retained_ref.host_register_flags;` — **reports it back via `hipHostGetFlags`** |
| `memory.c:2405-2406` | stores `allocation_flags` |

and `FLAG_HIP_COHERENT` appears exactly twice outside the enum, both in the HIP binding:

    api.c:754-755   if (hip_flags & hipHostMallocCoherent)  { flags |= …FLAG_HIP_COHERENT; }
    api.c:6908-6910 if (internal_flags & …FLAG_HIP_COHERENT) { hip_flags |= hipHostMallocCoherent; }

**So the coherent bit round-trips — set from the HIP flag, stored on the wrapper, reported back by
`hipHostGetFlags` — and never reaches a memory type.** It is set, stored, reported, and never honoured.

## What this closes, and the three-recorded-negatives it explains

**The coherence hypothesis family is closed.** Not "we could not make coherence matter" but "this runtime
cannot make host memory coherent": both paths allocate `0x52`, and no flag changes that. Three separate
recorded results are now explained by one mechanism:

1. `GGML_HRX_USE_UNIFIED_MEMORY=1` (the A/B in `315-mechanism.md`): only changes the **type recorded** on the
   ggml buffer (`ggml-hrx.cpp:313-314`) — the pool is chosen by `HOST_LOCAL` alone, so both arms got the same
   memory. **"Coherence does not matter" was really "the knob could not change the memory".**
2. The fix I prepared this session (OR `HOST_COHERENT` into the staging params) is a **no-op** — same pool.
3. The register path's coherence flags — computed, stored, reported — **never affect the allocation**.

## The sharper question this leaves

If host-visible memory is **always** non-coherent, then something else must make host reads of device-written
data correct *most of the time* — the campaign's reads are overwhelmingly fine. Whatever that something is, a
fault that appears **intermittently** cannot be "the memory is non-coherent" (that would be a constant, and it
would break every read, not one arm in twenty). So the mechanism must be something that **usually works and
sometimes does not**, and the coherence family is not it.

That reframes the remaining candidate cleanly, and it is consistent with everything else measured:

- the corruption is **one 256-byte-aligned region** (96.6 %, ≥89.5 % in every tensor family, vs 1.6 % chance)
  — 256 B = one wave64 of f32, or the record's MoE-table block;
- it appears in **all three cells including the quiet one**, so the forcer is not necessary;
- and it is **intermittent**, which is what a *race or a lane-level drain* looks like, and is not what a
  missing-invalidate looks like.

**Nothing here names the mechanism.** It removes a family that six weeks of the record could not remove, and it
removes it in a form that also explains why the earlier attempts failed rather than merely disagreeing with
them.
