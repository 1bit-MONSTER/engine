<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# A NEW in-scope lever, never exercised: the graph reordering (`GGML_HRX_GRAPH_ORDER`)

Found by working the three bounded checks on the hazard recorder that the re-scope identified. Two of the
three checks came back clean; the third is a live, untested mechanism candidate.

## The surface, at the pin (`e44c9d01a`, ggml-hrx)

`ggml/src/ggml-hrx/runtime/graph-record-order.cpp` builds a dependency graph from byte-range hazards and
may **reorder** the recorded commands before submission:

    order_prepared_commands_for_graph(...)                     # :239
      -> lookahead_order(deps)                                 # reorders to save barriers
      -> use_level = enabled && level_barriers < program_barriers
                     && transient_extent <= kMaxTransientExtent        # :285
      -> returns the reordered list, or {} to keep program order

and the result is applied to the **live** submission path:

    ggml/src/ggml-hrx/runtime/command-program-executor.cpp:1479
        const std::vector<PreparedCommand> ordered = order_prepared_commands_for_graph(prepared.commands);
        record_prepared_command_list(..., ordered.empty() ? prepared.commands : ordered, ...)

The gate is `GGML_HRX_GRAPH_ORDER` (`:35`): ordering is enabled unless the value is exactly `"program"`.

## The three checks

**Check 2 (collect/update ordering) — CLEAN.** `collect` runs for *every* binding of a command in the
first loop, and `update` for the same command only in the second loop afterwards; self-edges are then
explicitly removed (`deps[c].erase(remove(... == c))`). No self-dependency and no dropped-pair bug from
call ordering.

**Check 1 (write clears readers) — CLEAN as a hazard rule.** `update()` on a write sets
`last_writer = command` and clears `readers`, which is the correct write-after-read discipline: a later
reader depends on the last writer, which is itself ordered after those readers. No pair is lost.

**Check 3 (`kMaxTransientExtent`) — LIVE, and this is the finding.** The 16 MiB cap is not merely a
guard; it is the **selection criterion for whether reordering is used at all**:

    use_level = enabled && level_barriers < program_barriers && transient_extent <= kMaxTransientExtent

and the code's own comment states which programs fall inside it:

    // They are told apart by their transient arena extent: decode programs measured 0.2-6.5 MB,
    // 512-token prompt programs 12-335 MB.

**GLM's decode is 0.2-6.5 MB, i.e. inside the gate, so GLM's decode is reordered.** Larger programs are
not. This is a structural difference between decode and large-batch work *inside the engine*, on the
live path, that changes **which kernels overlap**.

## Why this is a candidate for THIS fault

- It is the one mechanism class the trigger points at. Forced eviction opens a preemption window; what
  can race across that window is decided by **which kernels overlap**. Reordering is precisely the thing
  that decides overlap.
- It is specific and reachable: only programs whose transient extent is <= 16 MiB are reordered, which
  selects decode-phase shapes — the shape GLM faults in.
- It is distinguishable from everything already falsified. Note carefully: the record's null lever is
  `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC` (a stream sync *after each replay*). That is a different lever:
  GRAPH_SYNC adds a fence after the work is ordered; GRAPH_ORDER changes the ordering itself. A null for
  the former says nothing about the latter.
- It is **never exercised anywhere**: `GGML_HRX_GRAPH_ORDER` appears in no console or log on the box and
  in no recorded study. It is genuinely untested.

## The discriminating experiment (cheap, pre-registerable)

One binary, one env var, the existing eviction rig — the same A/B form that refuted the read-path fix:

    A arm: the frozen three-cell rig, cell e, GGML_HRX_GRAPH_ORDER unset    (reordering ON, ~70% expected)
    B arm: the SAME rig and binary, GGML_HRX_GRAPH_ORDER=program           (reordering OFF)

    branch 1  B clean in cell e while A faults   -> reordering is implicated; this is a real cause path
    branch 2  B ~= A                             -> reordering is NOT the cause; lever falsified
    branch 3  both clean                          -> UNINFORMATIVE, per the disc2 discipline

Cost: the eviction cell faults at ~70%, so ~5 arms per cell discriminates (~20 min), not 60. This fits the
remaining box-time budget where a full campaign would not.

## Honest caveats

- This is a **candidate, not a finding**. Nothing here shows reordering causes a stale read; it shows the
  reordering exists on the live path, is selected by a criterion GLM's decode satisfies, has never been
  tested, and is the lever that governs kernel overlap — which is what the trigger's preemption window
  would need.
- The `use_level` gate also requires `level_barriers < program_barriers`, so reordering only engages when
  it saves barriers. Whether it engages for GLM's *specific* decode programs is checkable in one run with
  the existing `GGML_HRX_LOG_GRAPH_ORDER` diagnostic, which prints exactly this:
  `"graph order: N commands, L levels, barriers P in program order, K in lookahead order, transient extent E (X used)"`.
  **That log line is the cheapest possible first step: one decode, zero arms, no regime, and it says
  whether GLM's decode is reordered at all.**

---

# Coordination (2026-10-13 ~11:25) — the concurrent agent is on the HIP/runtime lane, not this one

While this finding was being written, the second agent on this box was identified precisely rather than
guessed at. Its live process tree is:

    node .../rea function build/hrx-install/lib/libamdhip64.so.7.9999.0 \
        --procedure hipMemset --provider ghidra --format json
      -> java ... ghidra.app.util.headless.AnalyzeHeadless ... -import /tmp/rea-ghidra-*/target-*.0

So it is running **REA + Ghidra against `libamdhip64.so`** (the HIP runtime), procedure by procedure.
Its working copy is `/home/bcloud/work/hrx-system` (checked out 10:53, HEAD `a5b9a7e`, a Loom compiler
commit) — i.e. it is on the **HIP/runtime + Loom-codegen** lane.

**This does not collide with the finding in this file.** The graph-reorder lever is in the engine's
`ggml/src/ggml-hrx/runtime/graph-record-order.cpp`, on the ggml-hrx side, and the discriminating
experiment needs only the existing eviction rig and one env var. The two lanes are complementary:

    their lane : HIP runtime API surface (REA/Ghidra on libamdhip64.so), Loom codegen
    this lane  : engine ggml-hrx graph recording / reordering, and the arm campaign

The one thing to avoid is a **second driver on the box**: the arm campaign must hold the box lock, and
only one 315 driver may run at a time (see `~/wt/315-COORDINATION.md`). Decompilation is CPU-only and
does not take the GPU, so the two can proceed in parallel; the campaign just must not run alongside
another *campaign*.
