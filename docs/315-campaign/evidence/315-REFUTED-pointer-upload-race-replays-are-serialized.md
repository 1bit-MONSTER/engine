<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# REFUTED: the pointer-upload refill race — replays are serialized by a full completion wait

`315-MECHANISM-CANDIDATE-pointer-upload-refill-race.md` proposed that the engine could refill `staging.host_data`
while a previous `upload_async`'s queued `hrx_stream_update_buffer` commands were still pending, so the device
would consume new bytes for an old update. **The replay path closes that window, and this document records it.**

## The deciding code

`command-program-executor.cpp`, the replay path in order:

    upload_prepared_host_staging(context, prepared);            // queues pointer-based updates
    hrx_graph_exec_launch(recorded.exec, context.stream);       // launches the graph
    flush_stream_commands(context, "flush HRX graph replay commands");
    wait_stream_commands(context, "wait for HRX graph replay commands", recorded.exec);   // <-- here

Note that the `hrx_stream_synchronize` further down is **conditional** — it fires only when
`GGML_HRX_DIAGNOSTIC_GRAPH_SYNC` is set — so the default ordering rests entirely on `wait_stream_commands`:

    static Status wait_stream_commands(context, label, work) {
      stream_wait_sleeping(context.stream, wait_history_for(work));
    }

and `runtime/hrx-sleeping-wait.cpp`:

    hrx_status_t stream_wait_sleeping(hrx_stream_t stream, WaitHistory & history) {
      if (blocking_wait() || stream == nullptr) return hrx_stream_wait(stream);
      ...                                    // history-based sleep is an OPTIMISATION only
      hrx_status_t status = hrx_stream_wait(stream);     // <-- ALWAYS a full completion wait
      ...
    }

**The sleep-then-query is an optimisation; every path ends in `hrx_stream_wait(stream)`.** So a replay does not
return until the stream is quiescent, and the next replay's staging refill cannot precede the previous replay's
uploads. **The race is closed, and the candidate is refuted.**

## What was eliminated on the way, and what that leaves

Four candidates have now been eliminated by direct evidence rather than by argument:

| candidate | eliminated by |
|---|---|
| snapshot-at-capture (`copy_source_into_graph = true`) | `iree_hal_streaming_memcpy_value_to_device` has **no callers anywhere** |
| coherence / allocation flags | closed from source: all three host paths hardcode `0x52`, flags are stored only |
| read-path visibility (memcpy vs `hrx_synchronous_d2h`) | **253,744** paired readings across two instruments, **zero** disagreements |
| pointer-upload refill race | replays are serialized by a full `hrx_stream_wait` |
| (weights cache keyed on allocation generation) | correct by design — weights are immutable, so an allocation-scoped key is right |

and two earlier claims of my own were **retracted** (the layer-41 onset; the fixed 32-entry boundary).

## What still stands, unanswered

The phenomenology is intact and none of the refutations touch it:

- a per-decode-step hazard of ~1e-3, **finite**, entering at a clean boundary, all 18 events clean-prefixed;
- **forced eviction 15.1×** the control hazard (0.001195 vs 0.000079 over 4,184 and 12,651 run-steps);
- **index 0 immune, 0 of 2,259** — the prefill's own output is always correct, so the fault is in the decode loop;
- front-loaded over ~7 decode steps under eviction (4× the arm's own rate) with **zero** early events in control
  and quiet arms;
- Face B's HTTP 500 is the server rejecting a degenerate generation, and **invalid UTF-8 cannot occur** (the
  garbage is U+FFFD, a valid codepoint).

## Why this is the end of what the record can decide

Every remaining step needs the same thing: **observe a decode step that is known to diverge, with an instrument
that reads something the record has never read at that step.** The archive names all 18 divergent
arm/request/step triples, and `patch-input-f32-sync-twin.py` (verified: applies, compiles) reads the f32 pair and
the unnamed graph input — but **neither has ever been run**, and no further arrangement of existing bytes can
substitute for it. Four candidates have been eliminated from the armchair; the fifth needs the box.

**No mechanism is named. Four are excluded, the phenomenology is pinned to a decode step with a rate and a
dose-response, and the next candidate cannot be tested without an arm.**
