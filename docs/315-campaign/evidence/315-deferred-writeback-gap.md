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

# A self-documented ordering gap: the deferred host writeback (`mark_stream_synchronized`)

Found while working the §55/§56 cross-dispatch leg after four families were closed. This is the first
candidate located **by the code saying so about itself**, rather than by inference from behaviour.

## The mechanism, read at the pin (`e44c9d01a`, ggml-hrx)

`ggml/src/ggml-hrx/runtime/command-program-executor.cpp` keeps a **deferred host writeback queue**:

    GraphReplayStreamState::add_host_writeback(host_destination, mapped_source, size, buffer)   // :51
        -> push_back({host_destination, mapped_source, size, buffer})   // merely QUEUES the memcpy

    GraphReplayStreamState::mark_stream_synchronized()                                          // :38
        for each pending writeback:
            std::memcpy(writeback.host_destination, writeback.mapped_source, writeback.size)
        pending_host_writebacks_.clear()

So a **host download staged by an earlier replay is not published to host memory when it happens** — it is
queued, and only performed later, at whatever point `mark_stream_synchronized()` runs.

`mark_stream_synchronized()` is reached from exactly three places:

    :888   the host-staging UPLOAD guard (see below)
    :1926  mark_graph_replay_synchronized(...) after a replay
    :1985  mark_graph_replay_synchronized(...) after a "wait for HRX command program reading host memory"

## The gap, in the code's own words

The upload path carries this comment (`:874-880`), and it is the load-bearing find:

> A host download staged by an earlier replay is only published to host memory by
> `mark_stream_synchronized()`; `add_host_writeback` merely queues the memcpy. **Uploading the same host
> memory before that has happened would read the previous contents.** The split scheduler computes splits
> back to back and only synchronizes a backend when it inserts a cross-backend input copy, **so nothing
> else guarantees the ordering.**

Two things follow:

1. The code **knows** there is a window in which host memory holds the *previous* contents while a
   download that should have updated it is still queued. The comment names the exact consequence: a
   subsequent read "would read the previous contents".
2. The only guard is at the **host-staging upload** site, and it is conditional:
   `if (context.graph_replay_state != nullptr && has_pending_host_writebacks())`. Nothing guarantees the
   ordering generally — the comment says so outright.

## Why this is a candidate for THIS fault

- **It is a stale-read of host memory**, which is the fault's shape: a consumer observes "the previous
  contents" of a buffer rather than what a producer wrote. That is Face B almost literally (a stale token
  buffer producing malformed output), and plausibly Face A if the stale bytes are non-finite.
- **It is deferred across a window.** Unlike `buffer_get` (the refuted candidate), which read at one
  instant, this queue holds a *pending* copy that is flushed at some later, conditionally-chosen point.
  A preemption/eviction landing anywhere inside that window can change what the queued `mapped_source`
  contains, which is exactly the trigger the pooled rig establishes (28 of 29 eviction-cell arms fault;
  `315-pooled-rig-verification.md`).
- **It explains why the read-path fix failed.** The A/B refuted `hrx_synchronous_d2h` at the `buffer_get`
  site; this is a *different* host read of a mapped buffer, on a different path, with a deferral the
  refuted site did not have.
- **It is on GLM's decode path.** The writeback queue is driven by host-staging downloads, which is how
  the executor publishes results back to host memory during decode.

## What is NOT established

- **Not shown to be reached with GLM's shapes.** Whether GLM's decode actually produces
  `pending_host_writebacks_` entries, and whether an upload of the same host memory ever follows while
  they are pending, is checkable — the queue is countable and `GGML_HRX_LOG_*` exists — but has not been
  checked yet.
- **Not shown to be a defect.** The guard at `:888` may cover the only ordering that matters in practice,
  and the `:1926`/`:1985` sync points may be sufficient for every real sequence. The comment's "nothing
  else guarantees the ordering" is about the *general* case, not proof of a reachable failure.
- **Not a cause.** It is a located, self-documented hazard of the right shape on the right path.

## The bounded next steps, cheapest first

1. **Count it.** Run one GLM decode with the existing instrumentation and establish whether
   `pending_host_writebacks_` is ever non-empty, and whether `mark_stream_synchronized` runs between a
   download and a later upload of the same host memory. Zero arms, no regime, ~2 minutes — the same
   class of probe that established the graph reorder was live (`315-graph-reorder-lever.md`).
2. **If it is reached**, the discriminating test has the same cheap A/B form: force the sync
   (`mark_stream_synchronized`) before every host-staging upload unconditionally, and compare cell `e`
   against the shipped conditional guard. If the fault rate drops, this is the carrier.
3. **If it is not reached**, this family closes like the others, with a measured reason.

Recorded because it is the first candidate found by the code documenting its own ordering hazard, and
because it is the only surviving cause candidate that is both in scope and cheaply testable.

---

# The instrument for this already exists — and has never been run

Checked before building anything, and the writeback queue is **already instrumented** in the tree that
`315coherent-build` was built from (`~/wt/llama-315coherent`, and also `~/wt/llama-315copy`):

    GGML_HRX_WRITEBACK_FINGERPRINT=1
      -> fingerprint_writeback_log(...)   command-program-executor.cpp:84
         "[hrx-wb-fp] value=%d size=%zu barrier=%d rec_hash=%016llx sync_hash=%016llx dst_hash=%016llx
          rec_class=%s sync_class=%s dst_class=%s rec_eq_sync=%d
          rec_nf16=%llu sync_nf16=%llu dst_nf16=%llu rec_nf32=%llu sync_nf32=%llu dst_nf32=%llu"

That is purpose-built for exactly the question this lead poses. Per writeback it records:

- **`rec_hash`** — the source fingerprint **when the writeback was recorded** (the download);
- **`sync_hash`** — the source fingerprint **at the sync point**, when the deferred `memcpy` would run;
- **`dst_hash`** — the destination after the copy;
- **`rec_eq_sync`** — **whether the source changed between recording and sync.** `rec_eq_sync=0` is the
  deferred-window hazard made directly observable: the queued `mapped_source` held different contents at
  record time than at sync time;
- **`rec_nf16/32`, `sync_nf16/32`** — non-finite counts on each side, which is the link to Face A.

And there is a driver already written for it: `docs/315-campaign/315-fp-hunt.sh` runs instrumented
in-regime arms and reports `wb_records=$(grep -c "\[hrx-wb-fp\]" ...)` per arm.

**Neither has ever produced output.** A search of the whole record (`docs/`, `KNOWN-ISSUES.md`) and of the
box (`~/wt/*console*.log`, `~/wt/rt-det/*/server.log`) finds **zero** `[hrx-wb-fp]` lines anywhere. The
instrument was built on 2026-10-13 and, as far as the record shows, never switched on.

## Consequence for the plan above

Step 1 as written ("count it with existing instrumentation") needs no new code and no fault — one GLM
decode with `GGML_HRX_WRITEBACK_FINGERPRINT=1` answers whether the queue is used at all, and if it is,
whether `rec_eq_sync` is ever 0 under the eviction rig. That probe is staged at
`~/wt/315wb-probe.sh` and queued behind the peer agent's campaign.

This also means the four closed families were worked while a purpose-built instrument for a *fifth*
candidate sat unused in the same binary — worth recording as a process observation, not a criticism: the
instrument was built for a different framing (download fingerprints) and its writeback half had no
consumer until this lead gave it one.

---

# The queued source is a NON-COHERENT `0x52` mapping — the same type the refuted candidate named

Traced the writeback's source buffer to its allocation, which sharpens this lead from "a deferred read"
to "a deferred read of the same memory type the refuted fix was about".

Chain (`~/wt/llama-315coherent`, the tree `315coherent-build` came from):

    download_prepared_host_staging(...)                      command-program-executor.cpp:1065
      for each prepared.host_staging with .download:
        allocate_mapped_host_staging_buffer(...)             -> download_data (a mapped pointer)
        hrx_stream_copy_buffer(staging.buffer -> download_buffer)   device -> staging copy
        add_host_writeback(staging.host_data, download_data, ...)   QUeues: dest=host_data, src=download_data

    allocate_mapped_host_staging_buffer(...)                 runtime/host-memory.cpp:1287
        hrx_buffer_params_t params = {
            HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,     // :1300   == 0x52
            HRX_MEMORY_ACCESS_ALL,
            HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        };
        hrx_allocator_allocate_buffer(...);
        hrx_buffer_map(buffer, HRX_MAP_READ | HRX_MAP_WRITE, 0, size, &host_data);   // :1310

So `download_data` — the pointer the deferred `memcpy` reads from, at whatever later moment
`mark_stream_synchronized()` runs — is a **`HOST_LOCAL | DEVICE_VISIBLE` = `0x52` mapping with no
`HOST_COHERENT`**: exactly the memory type the read-path mechanism chain named, and exactly the type the
refuted candidate read, just at a different site.

## Why that makes this materially stronger than the refuted candidate

- **The refuted candidate** read the non-coherent logits mapping at a **single, immediate** point, and its
  fix (route the read through a coherent queued transfer) was shown not to help — cell `e` faulted 4/4
  with the fix on (`315-ab1-arm-B-interim.md`).
- **This one is deferred.** The non-coherent source is read at the *sync point*, arbitrarily later than the
  device copy that populated it, and the code states the ordering is not otherwise guaranteed. A
  preemption/eviction landing anywhere in that window is precisely the trigger the pooled rig establishes
  (28 of 29 eviction-cell arms; `315-pooled-rig-verification.md`).
- The instrument already in the binary measures exactly this: `rec_hash` at record time vs `sync_hash` at
  sync time, with `rec_eq_sync=0` marking a source that changed while queued.

## Still not established

- Whether GLM's decode produces `host_staging` buffers with `.download` at all, and therefore whether
  `pending_host_writebacks_` is ever non-empty on this path. That is what the staged probe
  (`~/wt/315wb-probe.sh`) answers.
- Whether the `:994` guard covers every reachable sequence even when the queue is non-empty.
- It is not a cause; it is a located, self-documented hazard of the right memory type, deferred across the
  window the trigger opens.

---

# The logits are structurally expected to take this path: `direct_host_binding=0`

Completed the chain from the other end — from GLM's logits binding to the deferred writeback — using the
recorded allocation trace rather than inference.

## The gating condition

The writeback is queued in `materialize_bindings` (`command-program-executor.cpp:764`) for any binding that
is host-visible and not a graph input:

    staging.download = access.write && !binding.graph_input;

guarded by `if (binding.host_data == nullptr) { ... continue; }` (`:750`) — i.e. only **host-visible**
bindings reach it. A host-visible binding that the graph **writes** therefore queues a deferred writeback.

## The logits are exactly that, and not by the direct path

The recorded trace (`docs/315-mechanism.md:34`, and `315-task9-session-state-2026-10-12.md:3304`) shows the
logits allocation as:

    type=HRX0_HOST  host_visible=1  direct_host_binding=0  memory_type=0x52  size=619520

and `ggml-hrx.cpp:319` defines the flag:

    const bool direct_host_binding = host_visible && type_context->device->use_direct_host_bindings;

with `use_direct_host_bindings = environment_flag_enabled("GGML_HRX_USE_UNIFIED_MEMORY")` (`:1202`), which
**defaults off**. So on the shipped configuration the logits are `host_visible=1` but
`direct_host_binding=0` — host-visible *without* the direct binding — which is precisely the condition that
routes them through host staging rather than binding host memory in place.

## What that means for this lead

- The logits are **host-visible** (so `host_data != nullptr` holds) and are **written by the graph** (so
  `access.write` holds), and they are **not** a graph input. All three terms of
  `access.write && !binding.graph_input` are satisfied.
- They are **not** on the direct-binding fast path (`direct_host_binding=0`), which is the path that would
  bypass staging entirely.
- The staging buffer that the deferred read later dereferences is the non-coherent `0x52` mapping traced
  above.

So the deferred-writeback path is **structurally expected** to carry the logits, not merely reachable in
principle.

## Still owed, and it is only a measurement

All of the above is static. Whether `pending_host_writebacks_` is non-empty at the right moments during a
GLM decode, and whether `rec_eq_sync` is ever 0 under the eviction rig, is what the staged probe
(`~/wt/315wb-probe.sh`, one decode with `GGML_HRX_WRITEBACK_FINGERPRINT=1`) reports. Static expectation is
not a result, and this section is explicitly not claiming one.

---

# Completing the mechanism: the writeback is queued at the END of a replay and NOT flushed there

Traced the replay path end to end. The deferral is not incidental — the queue is populated as the last
act of a replay, and the replay returns without flushing it.

## The three download sites, and what follows each

`download_prepared_host_staging(context, prepared, insert_barrier)` is called from three places:

    :1978   download_prepared_host_staging(context, prepared)              insert_barrier = true (default)
            -> followed by `result.success = true; return result;`        (the diagnostic-kernel path)
    :2047   download_prepared_host_staging(context, prepared, false)       insert_barrier = FALSE
            -> followed by `result.success = true; return result;`        (the main path)
    :2082   download_prepared_host_staging(context, commands)             (the execute_program path)

Inside, the function queues each download and returns:

    for each host_staging with .download:
        allocate_mapped_host_staging_buffer(...)        -> non-coherent 0x52 mapped buffer
        hrx_stream_copy_buffer(staging.buffer -> download_buffer)    device -> mapped copy
        add_host_writeback(staging.host_data, download_data, ...)    QUEUES the host memcpy
    return status;

**None of the three call sites calls `mark_stream_synchronized()` afterwards.** The two that return
directly (`:1978`, `:2047`) return with the writeback still pending; the third hands control back to
`execute_prepared_command_program`, whose flush point is conditional.

## So the sequence on the main path is

    replay launches the graph
      -> download_prepared_host_staging(..., insert_barrier=false):
           device -> mapped (0x52) copy, then QUEUE a host memcpy from that mapped buffer
      -> replay returns success, writeback still PENDING

and the queued memcpy is executed only when some later caller reaches `mark_stream_synchronized()` —
which, per the code's own comment, is the host-staging **upload** guard plus the two
`mark_graph_replay_synchronized` sites, and "nothing else guarantees the ordering."

## Why the asymmetry matters

The `insert_barrier` argument controls whether a **stream execution barrier** is inserted before the
download — a device-side ordering fence. It does **not** control the host-side flush of the queued
writeback. So:

- the device-side copy into the staging buffer is ordered by the barrier when `insert_barrier=true`, and
  by ring program order otherwise;
- the **host-side** read of that buffer (`download_data`) happens at an unrelated, later moment, and
  nothing ties it to the device copy that populated it beyond the eventual sync.

That is a genuine window, and it is wider on the main path (`insert_barrier=false`, `:2047`) than on the
diagnostic one (`:1978`).

## The instrument answers this exactly

`GGML_HRX_WRITEBACK_FINGERPRINT=1`'s `rec_hash` is taken when the writeback is queued and `sync_hash` when
it is executed. **`rec_eq_sync=0` means the mapped source held different bytes at flush time than when it
was recorded** — which is what an eviction or the CP writing through during the window would produce. The
probe staged at `~/wt/315wb-probe.sh` reports exactly that tally.

Explicitly still static: this section shows the window exists and is unflushed at the end of a replay. It
does not show the source ever actually differs at flush time, which is the measurement.

---

# MEASURED: the logits ARE host-visible `0x52`, and the writeback path is still NOT entered

The buffer trace was run in the build the earlier probes used, with both knobs on, to settle whether this
lead is reachable at all. It produced the sharpest result of the whole lead — and it **contradicts the
static reading** above.

## The trace

    [hrx-buf] type=HRX0      host_visible=0 direct_host_binding=0 memory_type=0x30 size=17895352832
    [hrx-buf] type=HRX0_HOST host_visible=1 direct_host_binding=0 memory_type=0x52 size=619520     <- the logits
    [hrx-buf] type=HRX0      host_visible=0 direct_host_binding=0 memory_type=0x30 size=443547648
    [hrx-buf] type=HRX0      host_visible=0 direct_host_binding=0 memory_type=0x30 size=81788928
    [hrx-buf] type=HRX0_HOST host_visible=1 direct_host_binding=0 memory_type=0x52 size=44048640

and, from the same run:

    writeback records: 0

So the static prediction **held**: the logits are `HRX0_HOST`, `host_visible=1`, `direct_host_binding=0`,
`memory_type=0x52` (not `HOST_COHERENT`) — exactly the allocation the chain above says should flow through
host staging. **And no writeback was queued anyway.**

## Where the static reading and the measurement disagree

`backend-buffer-binding.cpp:76-80` appears to support the static reading outright, in its own comment:

    const bool directly_bindable = !ggml_backend_buffer_is_host(buffer) || context->direct_host_binding;
    // Coherent HRX host allocations are directly device-addressable. ... Noncoherent host allocations
    // remain host data.
    binding.buffer    = directly_bindable ? context->buffer : nullptr;
    binding.host_data = directly_bindable ? nullptr : context->base;

with `direct_host_binding = 0` in the trace, `directly_bindable` should be `!is_host(buffer)`. For the logits
that is the open question — and the honest statement is that **this file's comment predicts staging for a
non-coherent host allocation, while the measurement shows staging is not entered.** One of the two is not
being read correctly, and the measurement is the one that cannot be wrong about the event count.

The likely resolutions, none verified:

- the logits' binding reaches the **first** early-return branch (`backend-buffer-binding.cpp:45-68`), which
  sets `binding.host_data = base` and returns immediately — host data, but **never passed through
  `materialize_bindings`**, so `staging.download` is never evaluated;
- or `ggml_backend_buffer_is_host(buffer)` is true for the logits, making `directly_bindable` false and
  sending them down a different route than assumed;
- or the host-staging entries are materialized elsewhere for this binding shape.

Distinguishing these is a bounded read of one more function, not a GPU question.

## What is now established, and what is not

- **Established**: the logits are a non-coherent `0x52` host-visible allocation in this build (measured,
  not inferred).
- **Established**: the `GGML_HRX_WRITEBACK_FINGERPRINT` instrumentation is compiled in and callable, and
  it recorded nothing across two sound probes (10 varying graphs) plus this trace run.
- **NOT established**: that the deferred writeback is unreachable in general. It is unreached on the paths
  exercised here.
- **Corrected**: the earlier section's claim that the logits are "structurally expected" to take the
  staging path is **not supported by measurement**. It rests on reading one comment as decisive when the
  event count says otherwise. That inference is withdrawn pending the bounded read named above.
