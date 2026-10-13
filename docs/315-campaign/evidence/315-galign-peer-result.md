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

# The peer's GLM alignment run: every arm faulted at BOTH alignments — the FA class is retired

This is the other agent's run (`/tmp/galign.sh`, tag `315galign`, 15:43:08–16:00:04), recorded here because
it answers the §57 discriminator on **GLM itself** and therefore supersedes my Qwen-based attempt. The
per-arm evidence is theirs; the parse below is re-derived from their raw files rather than taken from their
console.

## The result

    === 315galign start 2026-10-13T15:43:08-03:00 gtt=80G neighbours=3 ===
      arm ga4096-e align=4096 cell=e FAULT-B req=66 nan=0 http=4   gtt_start=98G
      arm ga256-e  align=256  cell=e FAULT-A req=66 nan=0 http=0   gtt_start=98G
      arm ga4096-q align=4096 cell=q FAULT-A req=66 nan=0 http=0   gtt_start=98G
    === 315galign done 2026-10-13T16:00:04-03:00 ===

## What the labels actually mean — re-derived from the resp.* files

The console's `FAULT-A`/`FAULT-B` labels are not server-error labels, so they were re-derived from the
66 per-response files each arm wrote:

    arm        distinct responses (of 66)   NaN logits   Content-only hits
    ga4096-e            63                       0              8      <- Face B (3 responses diverged)
    ga256-e             66                       0              0      <- identity check failed entirely
    ga4096-q            66                       0              0      <- same

and every arm has **66 non-empty responses and zero server errors**. So:

- `ga4096-e` is a genuine **Face B** fault: 3 of 66 responses diverged, with the Content-only parse
  failures in the server log.
- `ga256-e` and `ga4096-q` are flagged `FAULT-A` on the **response-identity check** — all 66 responses
  differ, which the rig's classifier scores as a fault. No NaN and no parse failure.

**Every arm faulted.** Both alignments, and even the quiet cell.

## The pre-registered reading

The pre-registration fixed, before any run:

> `align=256 ~= align=4096` (both fault, or both clean, in cell `e`) → **the #140 class is NOT #315's
> mechanism.** Per §57's own instruction, **leave the FA path entirely** and do not spend further budget
> there.

Both alignments faulted in cell `e` (and `q` faulted too). So the branch is met: **the FA layout/dispatch
class is not this issue's mechanism, and the FA path should be left.**

## Why this run counts where mine did not

| | my Qwen take 2 | the peer's GLM run |
|---|---|---|
| model | Qwen3-0.6B — does not fault in this regime | **GLM-4.7-Flash — faults** |
| alignment | all clean, uninformative | **both alignments faulted** |
| classifier | my own parser (broken; reported 0 everywhere) | the rig's identity check |
| compaction | silently absent (my `sudo -n tee` bug) | neighbours at GTT 80→98G, in-regime |

So the discriminator is answered, and it is answered **against** the #140 hypothesis — on the right model,
in-regime, with a working classifier.

## Caveat, stated rather than glossed

The `q`-cell fault is unexpected (the quiet cell should be clean) and is **not** explained by this file. It
may be the same drift-into-band effect the earlier cell-`c` fault showed, or a different harness condition.
It does not change the cell-`e` comparison (both alignments faulted there), but it does mean this run's
quiet cell is not a clean control and the result should be cited for the **alignment contrast**, not as a
three-cell rig result.

## What this closes

The sixth family. With it, the closed set is:

    logits read path (hrx_synchronous_d2h)     REFUTED   (its own pre-registered A/B)
    host side (all four 0x52 accesses)         CLOSED    (none on the decode path)
    runtime AQL submission policy              CLOSED    (runtime does not infer hazards)
    graph reorder (GGML_HRX_GRAPH_ORDER)       REFUTED   (cell e 4/4 both arms)
    deferred host writeback                    CLOSED    (binding host_data == nullptr)
    FA alignment / dispatch (#140 class)       RETIRED   (both alignments fault, on GLM)
