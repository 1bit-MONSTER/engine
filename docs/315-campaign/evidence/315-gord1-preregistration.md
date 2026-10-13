<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

engine#315 pre-registration. Written and committed BEFORE the run so the reading cannot be chosen post hoc.
-->

# Pre-registration: is GLM's graph REORDERING the cause? (`315gord1`)

Tests the lever established by `315-graph-reorder-lever.md`. The zero-arm probe proved the reordering is
**live on GLM's decode path** (26-command routed-FFN graph reordered in every layer, 23 -> 20 barriers;
transient extent 237,568 B against a 16 MiB gate). This asks whether it causes the fault.

## Hypothesis

**H-gord.** The graph reordering performed by `order_prepared_commands_for_graph`
(`ggml/src/ggml-hrx/runtime/graph-record-order.cpp:239`, applied at
`command-program-executor.cpp:1479`) changes which kernels overlap, and that overlap is what a forced
eviction's preemption window races. Forcing program order should therefore reduce or remove the fault.

## Design — one binary, one env var, the frozen eviction rig

Same form as the read-path A/B that was refuted, so the two are directly comparable:

    driver : 315-disc.sh 4 <TAG>, the SAME frozen three-cell rig as disc1/disc2/315ab1a/315ab1b2
    BIN    : ~/wt/315coherent-build/bin   (unchanged; already carries the knob, verified with strings)
    cells  : q quiet / e evictonly / c CPU pressure, rotating per round
    A arm  : EXTRA_ENV = the standard detector env (knob UNSET -> reordering ON, the shipped behaviour)
    B arm  : EXTRA_ENV = the standard detector env + GGML_HRX_GRAPH_ORDER=program  (reordering OFF)

Cell `e` is the test cell; `q` and `c` remain internal controls. 4 rounds per arm unless the result is
already clear.

## Why this is expected to be INFORMATIVE (unlike a knob that never engages)

Unlike an inert switch, the knob provably changes the submitted order on this model: the probe shows the
26-command graph is reordered under shipped settings. A clean cell `e` under the knob is therefore a real
intervention result, not branch 3 by construction.

## Prediction, and the four branches, fixed BEFORE the run

    B (program order) faults at ~0 in cell e, A (reordering on) faults at ~50-70% in cell e
        -> H-gord SUPPORTED. The reordering is the cause path, and the fix is to submit program order
           (or to fence the reordered overlap).
    B ~= A (both fault in cell e at a similar rate)
        -> H-gord REFUTED. Reordering is not the carrier; the family joins the closed list.
    B faults but A does not
        -> CONTRADICTED in direction; report as such (it would suggest reordering is protective).
    both clean in cell e
        -> UNINFORMATIVE, per the disc2 discipline. Given the probe shows the knob engages, this branch
           would most likely mean the rig failed to induce; re-run or report as uninformative.

## Named expectation about WHERE it would act

If supported, the effect should be attributable to the **26-command routed-FFN graph** (the reordered
one), not the 9-command attention graph (which is already submitted in program order and so is unchanged
by the knob). That asymmetry is the mechanism-level detail to check if H-gord survives.

## Cost and budget

The eviction cell faults at ~70% (arm A: 4/4; arm B 2/2), so ~5 arms per cell discriminates: roughly
20-25 minutes per arm, ~45 minutes total. This is chosen deliberately over a 60-arm campaign because the
box-time record puts this goal turn at ~11.1 h of a <=12 h ceiling, and because a discriminating
intervention needs power against a 70% rate, not a 5% one.

## Stopping rule

Stop at 4 rounds per arm, or earlier only if cell `e` is unanimous across the rounds already run AND the
other arm's cell `e` has also produced at least 2 scored arms. No arm pruning. INVALID/OOR arms are
recorded as invalid, never as passes.

## What this does and does not settle

- **Would settle**: whether the graph-reorder family carries the fault — a real step toward criterion 1's
  "named, falsifiable mechanism", with a discriminating experiment.
- **Would NOT settle by itself**: criterion 2's forward-only fix. If H-gord survives, the fix shape has to
  be argued (program order is a performance regression by construction — it costs the 3 saved barriers
  per layer graph — so the real fix may be to fence only the overlap, which is a different change).
- **Does not touch criteria 3 or 4**, which need a landed fix first.
