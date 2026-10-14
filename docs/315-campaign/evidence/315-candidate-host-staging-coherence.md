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

### The discriminating experiment, made runnable: toggle the bit on *this* allocation

The readback version above is not runnable as written — there is **no getter for a buffer's achieved memory
type** (`hrx_buffer_get_size`, `hrx_buffer_get_device_ptr`, `hrx_buffer_lookup`, `hrx_buffer_map` exist; no
`get_memory_type`), and the `[hrx-buf]` trace prints `memory_type` only inside `buffer_alloc`
(`ggml-hrx.cpp:322`), not for buffers allocated directly through `hrx_allocator_allocate_buffer`. So the
achieved type cannot be read out without new API.

The runnable form is therefore a **direct A/B on this one allocation**, which is also the stronger test because
it measures the fault rather than the flag:

| arm | change | prediction if the mechanism is live |
|---|---|---|
| A′ | `allocate_mapped_host_staging_buffer` unchanged | faults at the recorded rate |
| B′ | add `HRX_MEMORY_TYPE_HOST_COHERENT` to the params at `host-memory.cpp:1300` | **fault rate falls** |

**This is the experiment the earlier coherence A/B should have been.** That one toggled
`GGML_HRX_USE_UNIFIED_MEMORY`, which reaches only `buffer_alloc`'s ggml buffer types and provably cannot touch
this allocation; both its arms faulted, which refuted the knob and not the property. A′ vs B′ moves the bit on
the buffer that is actually device-written and host-read with no invalidate.

**Falsifier, unchanged and still first in this document:** if the backend already grants coherence to
host-located memory (`vmm.c:468-473` is the reverse mapping and suggests it might), then B′ changes nothing
and the fault rate is unmoved — in which case this candidate is dead and A′ vs B′ is the cheapest possible way
to discover that.

### Honest status after reading the provider paths

- **Not our provider:** `vmm_slab_provider.c:401-409` **rejects** any request carrying `HOST_LOCAL |
  HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` ("VMM slab provider requires device-local memory without host
  access"), so the staging buffer cannot be served there.
- **Unresolved:** the hip path handles `hipMemLocationTypeHost` (`vmm.c:797` is access-location validation, not
  the type conversion), and I could not locate the forward conversion from a requested hrx memory type to the
  `hipMemAllocationProp` actually used. **So the achieved coherence remains unresolved from source**, and this
  document does not claim otherwise.

**Bottom line: a well-founded candidate with one unresolved premise and a cheap A′/B′ that settles it either
way.** That is a strictly better position than the record had — it is a named mechanism with a discriminating
experiment and an in-scope, forward-only, one-bit fix — but it is **not** yet the named mechanism, and it must
not be reported as the cause.

---

# PRE-REGISTRATION: the A′/B′ staging-coherence campaign (`315abA` / `315abB`)

Written **before** the run. The design above is a sketch; this fixes the shape, the readout, the stopping
rule and the confounds, so the result is a test rather than a search.

## Shape

Two conditions, **one binary**, differing in exactly one environment variable:

| condition | `EXTRA_ENV` | tag | rounds |
|---|---|---|---|
| **A′** (as shipped) | `GGML_HRX_PRODUCER_PROBE=1 GGML_HRX_PRODUCER_DEVICE=1 GGML_HRX_TENSOR_FP=<15-name l_out list> GGML_HRX_TENSOR_FP_NANONLY=1` | `315abA` | 4 |
| **B′** (bit added) | the same **plus** `GGML_HRX_STAGING_COHERENT=1` | `315abB` | 4 |

Run **back to back under one lock**, A′ first, so session drift and the box's thermal state are shared rather
than split across conditions. 4 rounds × 3 cells = **12 arms per condition**, of which **4 are eviction-cell
arms** — the cell the recorded rate lives in.

The two instruments ride along in both arms: the **producer probe** (device mode, finite sentinel) reports
whether the device wrote the residual-family tensors, and the **fingerprint** provides the fault-yield
enabler and the onset readout. Neither is a difference between the conditions.

## Readout, fixed in advance

Primary: **fault count per condition over the eviction-cell arms** (`e`), both faces counted.

| observation | verdict |
|---|---|
| B′'s eviction-cell fault rate is materially **lower** than A′'s | the mechanism is **live** — coherence on this allocation matters, and the fix direction is confirmed |
| B′ and A′ fault at **similar** rates, with ≥3 eviction-cell faults total | the candidate is **dead**; the achieved mapping was already coherent, and this line is closed |
| ≤1 eviction-cell fault in **both** conditions | **INVALID / under-powered** — reported as such, never as "no difference" |
| a condition produces **no Face-A arm at all** | that condition contributes nothing to the rate; reported, not read as clean |

Secondary (free, from the same arms): the producer probe's `armed` / `anomaly` lines — `sentinel == n` means the
device never wrote that tensor, `0 < sentinel < n` partial, `sentinel == 0` fully written.

## Stopping rule

Fixed: run both conditions to completion at 4 rounds each. **No early stopping on a trend**, and no extra
rounds added after seeing the counts — if the result lands in the under-powered row, the correct response is a
**new** pre-registration with more rounds, not an extension of this one.

## Confounds, declared before the run

- **Not a rate estimate for acceptance.** 4 eviction arms per condition cannot bound a rate; this is a
  directional diagnostic, and no criterion-3 claim may be read from it.
- **The probe perturbs the regime.** Filling live tensors changes downstream values and adds a host⇄device
  copy per hooked graph; the record already holds that this instrumentation is itself an inducer. So this
  characterises *the probe's regime*, and a difference between A′ and B′ is a difference **within** that regime.
- **The box was thermally saturated earlier today** (`thermal_pauses` to 29, `tctl` to 95 °C, `stopped_frac`
  to 0.82) and ran under foreign decompilation load. Both conditions share that, which is why they are run back
  to back — but a thermally noisy box widens the interval on any rate, and that is stated rather than adjusted
  for.
- **A face may be under-represented.** Face A and Face B have not been observed at equal rates, and Face B has
  never been measured with a capable instrument; a fault-count comparison that is all-Face-A is reported as
  such.

## BUILT and RUNNING (2026-10-14 06:00) — both instruments compile and coexist in one binary

    [ 26%] Building CXX object .../runtime/host-memory.cpp.o
    [ 26%] Linking CXX shared library ../../../bin/libggml-hrx.so
    [100%] Built target ggml-hrx
    === EXIT=0 ===
    strings bin/libggml-hrx.so.0.18.0 | grep -cE 'GGML_HRX_STAGING_COHERENT|GGML_HRX_PRODUCER_DEVICE'  ->  2

So the A′/B′ toggle **compiles** and the device probe is still in the same artifact, which is what makes a
single run answer both questions. `host-memory.cpp` backed up at `/tmp/host-memory.cpp.orig`
(sha256 `cc7efbc4b873fe8d…`) before patching.

**Campaign launched**: `315ab` — A′ (`315abA`) then B′ (`315abB`), 4 rounds each, back to back under one lock,
behind the gate-aware wrapper that holds no lock and no GTT while waiting. **Still NOT claimed: that either
condition will fault, or that B′ differs from A′.** The arm verdicts and the producer-probe lines are that
evidence, and they do not exist yet.

**Running with the probe in both arms is a declared confound, not an oversight**: the probe fills live tensors
and adds a host⇄device copy per hooked graph, and the record holds that this instrumentation is itself an
inducer. Both conditions carry it identically, so the A′/B′ contrast is within the probe's regime — but it is
not the ambient regime, and the pre-registration says so.

---

# The chain, completed: device reads are clean, so the corruption is real and the staleness must enter host-side

Reading `HostTransferManager::download_synchronous` settles which reads are trustworthy and where staleness can
enter. Both matter, and one of them refutes a competing hypothesis I was about to record.

## Two download paths, only one of them clean

| tensor class | path | trustworthy? |
|---|---|---|
| **device** tensors (layer residuals — everything the fingerprint read for the size/position findings) | `backend_get_tensor_async` → unregistered pointer → `synchronous_download_fallback` → `download_synchronous` → **`hrx_stream_synchronize` then `hrx_synchronous_d2h`** | **yes: a real runtime D2H** |
| **host-visible** tensors (the logits region) | generic ggml layer short-circuits for host buffers → **plain memcpy, no backend call, no invalidate** | **no** |
| graph inputs published by the **staging download** | `add_host_writeback` → **deferred plain memcpy** from a mapped staging buffer | **no** |

    runtime/host-memory.cpp:1421-1425   (download_synchronous)
      hrx_stream_synchronize(stream);
      hrx_synchronous_d2h(device, source, offset, host_destination, size);

So the "the fingerprint's own reads are stale" hypothesis is **refuted**: my instrument reads device tensors
through a synchronised D2H, not through the deferred mapped copy. **That is a validation of this session's
measurements** — the one-row size invariant, the front-of-buffer position, and `finite + nan == extent` are
readings of **real device state**, not of instrument memory. It also agrees with the record's independent
result that the read path was refuted (138,256 of 138,256 identical when every tensor was read twice).

## What that forces

The corruption was measured on the **device**, through a clean read. So the wrong bytes were on the device, and
the question becomes how they got there — and the answer is available in the same file: the **host→device**
direction.

    command-program-executor.cpp:994-996   (comment, on the upload path)
      "A host download staged by an earlier replay is only published to host memory by
       mark_stream_synchronized(); add_host_writeback merely queues the memcpy. Uploading the same
       host memory before that has happened would read the previous contents."

The code guards *that* ordering explicitly. **The chain that survives is therefore:**

1. a graph input is published to host memory by the **deferred, non-coherent** staging writeback (or read
   before its invalidate);
2. that host buffer is then **uploaded to the device** for a later graph (`staging.upload`);
3. the device computes from a **stale input**, so a real device tensor — the onset residual row — becomes NaN
   or wrong;
4. the fingerprint reads that residual through a **synchronised D2H** and reports it **accurately**.

This is more parsimonious than a device-side defect: every measurement stays valid, the corruption is real and
device-resident, and the mechanism is stale host data being re-uploaded. It also keeps every observation that
had to be explained — intermittency (cache state), the eviction correlation (cache pressure), the front of the
buffer (first cache lines of a reused staging buffer), arena/staging reuse (which makes the stale bytes
plausible, i.e. Face B), the quiet-cell faults (host cache state is box-wide), and the refutation of the
coherence knob (wrong buffer).

**The A′/B′ campaign now tests exactly this bit**: adding `HOST_COHERENT` to the staging allocation makes the
published bytes visible before they are re-uploaded. If B′ reduces the fault rate, the chain holds.

### Amendment, still BEFORE any arm: `GGML_HRX_WRITEBACK_FINGERPRINT=1` added to BOTH conditions

A design risk found while re-reading the publish path, and it would have made the run uninterpretable.

**The risk.** The chain above needs the staging path to be **warm**. The record already measured the deferred
writeback route as **cold** — *"0 `[hrx-wb-fp]` records with the flag on over a complete 66/66 arm"*. If it is
cold in the disc-rig configuration too, then `add_host_writeback` is never called, no graph input is ever
published by this path, and **A′ and B′ cannot differ** — a null result that would say nothing about the
mechanism while looking like evidence against it.

**The amendment.** Both conditions now also carry `GGML_HRX_WRITEBACK_FINGERPRINT=1` (the instrument the record
used; inert unless set). It is added to **both** arms, so it remains a non-difference between the conditions,
and it converts the risk from an assumption into a measurement. This is a diagnostic added to the *existing*
run; the primary readout, the arm counts, and the stopping rule are unchanged.

**The extra readout, fixed in advance:**

| observation | verdict |
|---|---|
| `[hrx-wb-fp]` lines present in the arms | the staging writeback path **was exercised** ⇒ the A′/B′ contrast is meaningful |
| **no** `[hrx-wb-fp]` line in either condition | the path is **cold in this configuration** ⇒ the chain has no first step here, the A′/B′ null is **structurally expected and uninformative**, and this candidate is dead **for the disc rig** (it would then need a configuration that actually exercises host-staging downloads — the multi-ubatch/split path that #90's message implicates — as a new pre-registration) |

That second row is the important one: it makes the run **self-falsifying**, so a null cannot be mistaken for a
refutation and cannot be mistaken for support either.
