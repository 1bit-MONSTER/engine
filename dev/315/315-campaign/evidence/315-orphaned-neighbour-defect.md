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

# Tenth harness defect: an orphaned regime neighbour survives and is silently reused

While setting up the sync-instrumented re-run, `pgrep -f "315-idprobe|315-attrib"` reported the probe as
**still running** — long after it had printed its closing summary line. The match was not the probe:

    PID    PPID   ELAPSED   RSS     CMD
    106189 1      02:50:10  248232  /home/bcloud/wt/315coherent-build/bin/llama-server -m .../GLM-4.7-Flash-Q4_K_M.gguf

**Parent PID 1**, elapsed 2h50m — so it started at ~16:49, **before** the id probe (19:10). It was not a
neighbour the probe *started*; it was an orphan from the earlier coherent-read work that the probe's
composition step **found already running and counted toward the target GTT**.

Two consequences, both bad for the regime gate:

1. **The composition baseline is not what the run reports.** `gtt_before=44G` at the start of the id-probe
   session was not "the box at rest" — it was a box still holding an orphaned 248 MB server plus whatever else.
   The arms are labelled by their own `gtt_start`, which stays correct, but the *composition arithmetic* that
   projects "neighbours at 85 GiB → detector at 101-106 GiB" is computed from a baseline that may already
   include someone else's leftovers.
2. **A process-count check is not a liveness check.** `pgrep` on a script name can match a long-lived child of
   a *previous* run, which is the same shape as the `pgrep` self-match defect recorded earlier in this
   campaign: the check answers a question adjacent to the one being asked.

Cleanup: the orphan was killed (it is this campaign's own build, `315coherent-build`, the
`GGML_HRX_COHERENT_READ` knob build from the refuted read-path candidate; GTT fell 44G → 26G). **Two
`315coherent-build` servers remained** and were left alone pending identification — they may be the current
probe's own neighbours. Two unrelated servers (`1bit-pr-agent`, `1bit-canonical`) were **not** touched: they
are other workloads and out of this campaign's scope.

Not yet established: whether orphaned neighbours have silently inflated the baseline for earlier runs in this
campaign. The arms carry their own `gtt_start`, so those labels remain valid; the compositions are what need
re-checking.
