<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Graph-reorder A/B arm B — INTERIM: cell e faults with reordering DISABLED (H-gord refuted)

**Status: INTERIM.** Arm B (`315gord1b`) is still running (round 1 of 4 complete at the time of writing).
Recorded immediately so the observation cannot be lost or re-read after the fact; the full tally follows.

## What is already in

Same binary, same frozen three-cell rig as arm A, with the reorder **disabled**
(`GGML_HRX_GRAPH_ORDER=program`), box acquired after the peer agent's campaign series released the GPU.

    BIN=/home/bcloud/wt/315coherent-build/bin \
    EXTRA_ENV="GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1 GGML_HRX_GRAPH_ORDER=program" \
    bash ~/wt/315-disc.sh 4 315gord1b

    round 1 arm q: OK     gtt 97G->98G   side=0/0
    round 1 arm e: FAULT  nan=0 req=7 http=1 s500=1 unparsed=2   gtt_start=98G  side=1/1   <- FACE B

Arm validity checked, not assumed:

- **the knob was genuinely set**: the arm server's own `/proc/<pid>/environ` carried
  `GGML_HRX_GRAPH_ORDER=program` while the arm ran, and the session header records it in `extra_env`;
- **in-regime**: `run.log` shows `gtt_start 98GiB mode performance cell=e side_pids=<evictonly>`;
- **the intervention was applied**: `side=1/1`, the forcer alive for the whole arm;
- **the failure is Face B by the recorded signature**: detector gives `http_failures: 1`,
  `first_failure: "HTTP 500"`, `truncated: true`, `requests: 7` (of 66), and the server logged
  `common_chat_peg_parse: unparsed Content-only output` followed by
  `"code":500,"message":"The model produced output that does not match the expected Content-only format"`.

That is the same Face-B signature arm A's cell-`e` arms produced (r3e, r4e).

## The pre-registered reading

The pre-registration (`101ab04`) fixed four branches before the run. This lands in:

> **B ~= A (both fault in cell `e` at a similar rate) -> H-gord REFUTED. Reordering is not the carrier;
> the family joins the closed list.**

Arm A's cell `e` was 4 FAULT / 0 OK; arm B's first cell-`e` arm is also a FAULT, **with the reorder
demonstrably disabled**. If the remaining rounds confirm it, disabling GLM's graph reordering does not
remove the fault, and the graph-order hypothesis is falsified as a cause.

## What this refutes, precisely

- **Refuted**: the graph reorder (`order_prepared_commands_for_graph`,
  `ggml/src/ggml-hrx/runtime/graph-record-order.cpp`) as the carrier of the fault — including the two
  shapes recorded in the tempering note, "more independent work in flight across the preemption window"
  and "the reorder exposes an incomplete hazard set". Disabling it entirely changes nothing observable.
- **Not refuted**: the AQL publication analysis (`315-aql-publish-vs-reorder.md`) remains a correct
  reading of the commit/doorbell contract; it simply is not where the fault comes from.

## Not yet claimed

A single round is not "B ~= A". The claim waits on arm B's full cell-`e` tally. If later rounds are
clean, the reading becomes mixed and must be reported as such rather than resolved post hoc.
