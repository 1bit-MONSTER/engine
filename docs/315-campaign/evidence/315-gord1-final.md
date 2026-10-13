<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Graph-reorder A/B FINAL — reordering is NOT the cause. The family is CLOSED.

Pre-registered at `101ab04` (`315-gord1-preregistration.md`), which fixed the four branches before the
run. This is the completed, two-arm result; it supersedes the interim notes
(`315-gord1-arm-A-result.md`, `315-gord1-arm-B-interim.md`).

## The result

    arm A  reorder ON  (GGML_HRX_GRAPH_ORDER unset)    315gord1a, 4 rounds, 12 arms
    arm B  reorder OFF (GGML_HRX_GRAPH_ORDER=program)  315gord1b, 4 rounds, 12 arms

                 cell q            cell e              cell c
    arm A     4 OK / 0 FAULT    0 OK / 4 FAULT      4 OK / 0 FAULT
    arm B     4 OK / 0 FAULT    0 OK / 4 FAULT      3 OK / 1 FAULT

**Cell `e` (the forced-eviction test cell) faulted 4/4 in BOTH arms — a 100% fault rate with the reorder
enabled and with it disabled.** Computed by `tools/gord-analysis.py` from the two consoles:

    cell-e fault rate: A=100% (4/4)   B=100% (4/4)
    READING: both fault in every cell-e arm -> B ~= A -> H-gord REFUTED (lever closes)

## Validity, checked rather than assumed

- **The knob was genuinely engaged.** The arm-B server's own `/proc/<pid>/environ` carried
  `GGML_HRX_GRAPH_ORDER=program` while its cell-`e` arms ran, and the session header records it in
  `extra_env`. The zero-arm probe separately proved the knob changes the submitted order on this model
  (the 26-command route-FFN graph is reordered under shipped settings, 23 -> 20 barriers), so arm B's
  arms were not branch-3-by-construction: the intervention provably acted.
- **All arms in-regime**, `gtt_start` 97-98G, zero OOR, zero INVALID in both arms.
- **Both faces appeared in both arms.** Arm A cell `e`: r1, r2 Face A (NaN); r3, r4 Face B (Content-only
  500). Arm B cell `e`: r1 Face B; r2 Face A; and (from the full tally) 4/4 faults total.
- **The eviction intervention was live**: `side=1/1` on every cell-`e` arm in both arms.

## The pre-registered branch, met

> **B ~= A (both fault in cell `e` at a similar rate) -> H-gord REFUTED. Reordering is not the carrier;
> the family joins the closed list.**

## What is now closed

The graph reorder is **not** the carrier of the fault, in either of the two shapes the tempering note
allowed:

1. "the reorder increases the amount of independent work in flight across the preemption window" — false:
   removing the reorder entirely leaves the rate at 100% in cell `e`;
2. "the reorder exposes an incomplete hazard set in `RangeHazards`" — false for the same reason: with the
   reorder off, the recorded order is submitted verbatim and the fault still occurs at the same rate.

So `ggml/src/ggml-hrx/runtime/graph-record-order.cpp` — its `lookahead_order`, its `RangeHazards`
coverage, and the 16 MiB `kMaxTransientExtent` gate — is **exonerated as a cause**, and the AQL publication
analysis (`315-aql-publish-vs-reorder.md`) is confirmed to be a correct reading of the commit/doorbell
contract that simply is not where the fault comes from.

## The incidental control-cell fault, recorded not explained away

Arm B's cell `c` (CPU-pressure control, no GPU side load) faulted once, Face A. Arm A's cell `c` was 4/4
clean, and so were the earlier runs'. This is the same pattern as the previously recorded control-cell
fault: the control cell drifted into the faulting band. It is recorded as an observation about the rig,
not as evidence for any mechanism, and it does not affect the cell-`e` comparison (which is cell `e`
against cell `e`).

## Where this leaves the goal

A third mechanism family is now closed by a pre-registered discriminating experiment:

    read path (hrx_synchronous_d2h)     REFUTED  (315-ab1-arm-B-interim.md)
    host-side coherence (all 4 accesses) CLOSED   (315-write-side-unaddressed.md)
    runtime AQL submission policy        CLOSED   (315-device-hunt-boundary.md)
    graph reorder (GGML_HRX_GRAPH_ORDER) REFUTED  (this file)

The cause remains unlocated, and the surviving candidates are the ones §55/§56 left open
(executor-level cross-dispatch ordering) plus the MoE id-producing-kernel family the peer agent is
working. No fix is proposed or landable on the current evidence.
