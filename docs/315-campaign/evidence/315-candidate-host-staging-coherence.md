<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Candidate mechanism: the deferred host-staging download is a CPU memcpy out of a device-written host buffer that requests no HOST_COHERENT

Read at the pin (`e44c9d01a` / hrx-system `98d05d94a9`) while the box was blocked. **This is a named,
falsifiable mechanism with a cheap discriminating experiment — and it comes with a falsifier I have not
eliminated, which is stated first because it decides whether this is the mechanism at all.**

## The falsifier that must be eliminated first

The runtime's HIP VMM backend **unconditionally grants coherence to host-located memory**:

    libhrx/src/binding/hip/vmm.c:468-473
      if (properties->location.type == hipMemLocationTypeHost) {
        // The location selects the physical backing owner, not merely which agent
        // may access the eventual mapping. ...
        return HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_HOST_COHERENT |
               HRX_MEMORY_TYPE_DEVICE_VISIBLE | cache_type;
      }

So **requesting no `HOST_COHERENT` does not mean the achieved mapping lacks it** — the backend may add it
regardless. The record's `memory_type=0x52` (host-visible, **no** `0x04`) was measured on a *different*
allocation path (`buffer_alloc` → the `HRX0_HOST` logits buffer), not on this staging path, so it is not
evidence either way here.

**Therefore the mechanism below is a candidate whose central premise is untested.** The discriminating
measurement is one line of logging (below), and until it is run this must not be reported as the cause.

## The candidate

The graph-input download path stages device data through a mapped host buffer and then publishes it with a
**plain CPU `memcpy`**:

    runtime/command-program-executor.cpp:1093-1107   (per staging download)
      allocate_mapped_host_staging_buffer(context.device, staging.length, download_buffer, download_data);
      hrx_stream_copy_buffer(context.stream, staging.buffer, 0, download_buffer, 0, staging.length);
      add_host_writeback(staging.host_data, download_data, staging.length, download_buffer, ...);
      hrx_buffer_release(download_buffer);

    runtime/command-program-executor.cpp:123-130   (at sync)
      std::memcpy(writeback.host_destination, writeback.mapped_source, writeback.size);

and the staging buffer is allocated **without** the coherence bit:

    runtime/host-memory.cpp:1299-1304
      hrx_buffer_params_t params = {
          HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,   // <- no HOST_COHERENT
          HRX_MEMORY_ACCESS_ALL,
          HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
          0,
      };

The copy at `1099` is **enqueued** (`hrx_stream_copy_buffer`) — the device writes the mapping — and the
staging→host copy is **deferred** to `mark_stream_synchronized`, which then reads that mapping with **no
invalidate**.

## The same file tree states the rule, for an analogous buffer

    libhrx/src/binding/common/event_timestamp_pool.c:136-146
      // The device writes a tick into this buffer at a queue point and the host
      // maps it back, so the allocation has to serve both: DEVICE_VISIBLE is what
      // states the device writes here, and every other host allocation this binding
      // hands the device asks for it beside HOST_LOCAL. HOST_COHERENT is a
      // requirement here and not a preference: the read path never invalidates, so
      // non-coherent memory would return stale ticks.
      params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT;

This is a device-writes / host-reads buffer, exactly like the staging buffer, and the authors call coherence
*"a requirement here and not a preference"* with the reason given explicitly: **"the read path never
invalidates, so non-coherent memory would return stale"** data. The staging path has the same shape and does
not ask for it.

## Why this is NOT the falsified lever

The campaign's coherence A/B toggled `GGML_HRX_USE_UNIFIED_MEMORY`, and both arms faulted, which refuted the
knob. That knob sets coherence **only** inside `buffer_alloc`:

    ggml-hrx.cpp:307-314
      const bool direct_host_binding = host_visible && type_context->device->use_direct_host_bindings;
      if (direct_host_binding) { memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT; }
    ggml-hrx.cpp:1190
      device_ctx->use_direct_host_bindings = environment_flag_enabled("GGML_HRX_USE_UNIFIED_MEMORY");

It is a **ggml buffer-type** decision. `allocate_mapped_host_staging_buffer` does not consult it, does not take
a ggml buffer type, and hardcodes its own params. **So the A/B never touched this buffer**, which is a
coherent explanation for why a coherence experiment refuted coherence: it was applied to the wrong allocation.

## What it would explain, if the premise holds

| observation | how this mechanism produces it |
|---|---|
| intermittency | whether the CPU holds a stale line for that address depends on cache state, so the fault is timing- and pressure-dependent by nature |
| the eviction correlation | eviction/paging disturbs cache residency and the mapping — the record's forced-eviction trigger is a cache-pressure amplifier |
| **row 0 or 1, front of buffer** | the first cache lines of a freshly reused arena slot are the ones most likely to still hold the previous tenant's data |
| **arena slot reuse** (the record's own finding) | it is what makes the stale bytes *plausible* rather than obvious garbage — the previous tenant was a same-shaped tensor |
| Face B (finite but wrong → HTTP 500) | stale finite values are a plausible-looking wrong answer |
| Face A (NaN logits) | a stale line that held a NaN pattern, or an unwritten region, yields NaN |
| faults in the **quiet** cell | cache/mapping state is box-wide, not forcer-local — which is why both faces appeared with `side=0/0` |
| `finite + nan == extent` (complete read) | the read *is* complete; it is the *contents* that are stale, so nothing about completeness is violated |

## The discriminating experiment — cheap, and it settles it before any fix

Log the **achieved** memory type for the staging allocation (or read it back) in one arm:

| outcome | conclusion |
|---|---|
| achieved type **lacks** `HOST_COHERENT` | the staging mapping is non-coherent while being device-written and host-read with no invalidate ⇒ **the mechanism is live**, and the fix is to add `HRX_MEMORY_TYPE_HOST_COHERENT` to the staging params (host-memory.cpp:1299) |
| achieved type **includes** `HOST_COHERENT` | the backend grants it anyway ⇒ **this candidate is dead**, and the earlier "read path never invalidates" reasoning does not apply to this buffer |

This is a single `GGML_LOG` beside the allocation, run in the configuration that already faults. It does not
need the device probe, a build of the probe, or a new campaign — and it is worth doing **before** spending box
time on anything else, because a positive result names the mechanism and specifies a forward-only fix inside
the declared scope (the fork-side `ggml-hrx` runtime).

## What is proven and what is not

**Proven, read at source:** the staging params omit `HOST_COHERENT`; the device writes that mapping and the
host later reads it with a plain `memcpy` and no invalidate; the deferred writeback exists and is ordered as
described; the coherence knob does not reach this allocation; and an analogous buffer in the same tree asks for
coherence explicitly, calling it a requirement because the read path never invalidates.

**NOT proven:** that the *achieved* mapping is non-coherent — the HIP VMM path may add it
(`vmm.c:468-473`). **That single fact decides the mechanism, and it is the first thing to measure.**
