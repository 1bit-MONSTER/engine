<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Independent replication of the eviction rig by a peer agent (315ids1)

Recorded because it is **independent corroboration** of the rig this goal's A/B rests on, from a different
agent, a different build, and a different instrument — and because the fact appeared while coordinating
GPU access, not from this goal's own runs.

## What the peer ran

    tag      315ids1        rounds 3        started 2026-10-13T12:21:24
    bin      ~/wt/315ids-build/bin   (version 10613, e44c9d01a — the pin)
    rig      315-disc.sh, the SAME frozen three-cell design (q quiet / e evictonly / c CPU pressure)
    env      GGML_HRX_TENSOR_FP=ffn_moe_topk,ffn_moe_argsort,ffn_moe_probs  (MoE routing instrumentation)

Note the build differs from this goal's `315coherent-build`: the peer used the plain pin build with
instrumentation on the MoE routing tensors, i.e. an unrelated experiment. It nevertheless ran the same
rig.

## Its result

    cell q: OK=3 FAULT=0 OOR=0 INVALID=0
    cell e: OK=0 FAULT=3 OOR=0 INVALID=0
    cell c: OK=3 FAULT=0 OOR=0 INVALID=0

## Why this matters

It is the **third independent reproduction** of the same shape, now on a different build:

    disc1/disc2 (earlier)   cell e: 7 FAULT / 10 arms
    this goal's arm A       cell e: 0 OK / 4 FAULT      (315gord1a, reordering ON)
    peer's 315ids1          cell e: 0 OK / 3 FAULT      (315ids-build, MoE instrumentation)

Every one of them has the eviction cell faulting at or near 100% while the quiet and CPU-pressure cells
stay clean, in-regime, with no OOR and no INVALID. Whatever else is uncertain about this issue, **the rig
itself is reproducible across builds and agents**, which is the property the A/B's validity depends on.

## What it does not establish

Nothing about the graph-reorder hypothesis — the peer's variable was MoE routing instrumentation, not
ordering. It is recorded as corroboration of the apparatus, not as evidence for or against any mechanism.

## Provenance

The peer is a separate agent working the same box; its consoles live at `strixhalo:~/wt/315ids1-console.log`
and `315ids2-console.log`. This file cites only the summary lines from its driver output, read directly
from the box. The peer's own conclusions are not reproduced or relied on here.
