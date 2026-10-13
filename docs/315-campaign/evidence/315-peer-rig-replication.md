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

---

# Coordination finding (2026-10-13 13:33): the peer's driver does NOT take the box lock

Discovered while waiting for the GPU, and it invalidates the obvious arbitration strategy. While the peer
agent's campaign `315ids3` was actively running, the box lock was **free**:

    $ flock -n ~/.cache/lax-decode/box.lock -c "echo LOCK-IS-FREE"
    LOCK-IS-FREE

    $ for p in $(pgrep -f "315-disc.sh 2 315ids3"); do ls -l /proc/$p/fd | grep -c box.lock; done
    0
    0
    0

So the peer's driver runs **without holding `~/.cache/lax-decode/box.lock`**, even though
`~/wt/315-COORDINATION.md` and the goal's own constraints say to take it. Consequences:

1. **The lock cannot arbitrate between the two agents.** A lock-wait would acquire a lock that the peer
   is not holding, then run concurrently with the peer's live campaign anyway.
2. **The only reliable signal is GPU/process idleness** — observing that no other `315-disc.sh` driver is
   alive. That is what arm A used, and it is what arm B now uses.
3. **A "60-second clean window" requirement is too strict** against a peer running a tight campaign
   series (`ids1` -> `ids2` -> `ids3`, each launched as the previous ends). Waiting for genuine driver
   absence with a short settle, and starting immediately, is the workable approach; the residual race is
   a few seconds of teardown, not a whole campaign.

Recorded because it is the kind of coordination assumption that silently produces concurrent GPU runs —
which is precisely the contamination class this issue has already been burned by once (the two-driver
incident on tag `315ab1b`, `315-ab1-two-drivers-incident.md`). The peer's non-locking behaviour is
outside this goal's control; what is within it is not relying on the lock for mutual exclusion.
