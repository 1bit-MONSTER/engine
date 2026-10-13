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
