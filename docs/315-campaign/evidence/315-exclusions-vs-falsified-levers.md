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

# Verification-contract check: the seven exclusions against the recorded falsified levers

The objective's verification contract requires: *"Re-read the issue's falsified-levers list and confirm the
claimed mechanism is not one of them; if it is, the fix is rejected."* Done explicitly here rather than
assumed, because two of this session's exclusions are **adjacent** to recorded levers and the difference
matters.

## The recorded falsified levers

From `KNOWN-ISSUES.md`'s lever table:

1. memory coherence is the carrier — `HOST_COHERENT` made unconditional (`315accept`, 9 clean + 1 fault / 10)
2. consumer-side ordering is the carrier — `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC=1` (0/4 vs 1/4, p = 1.00)
3. ordering the instrument's own reads — `ggml_backend_synchronize` before each read (p = 1.00)
4. forced AQL execution barrier on every packet — `HRX_FORCE_BARRIERS=1` (3/4 vs 1/4)
5. flash attention is the carrier — `-fa off` (1/4 vs 2/4, fault persists)
6. `HRX_COMMAND_BUFFER_MODE=pm4` — unusable (all fix arms failed at `req=1`)
7. the expert/partition tables go stale — static, both rebuilt by producer dispatches
8. the fused path loses `route_stride` — static, already fixed in every tree that faults
9. the engine's MoE expert-streaming cache — static, Vulkan-only

## This session's seven exclusions, mapped

| this session's exclusion | relation to the recorded list |
|---|---|
| logits **read path** (`hrx_synchronous_d2h` at `buffer_get`) | **new** — a specific code site, not on the recorded list |
| **host side** (all four host accesses to the `0x52` mapping) | **adjacent to lever 1** — lever 1 tested the *coherence attribute*; this enumerated the *accesses* and showed none is on the decode path. Complementary, not a re-proposal, and it *explains* lever 1's null: an attribute cannot repair a read that never happens per token. |
| runtime **AQL submission policy** | **adjacent to lever 4** — lever 4 forced a barrier on every packet (a change to *output*); this read the *policy* and found the runtime does not infer hazards at all. Different question, same subsystem. |
| **graph reorder** (`GGML_HRX_GRAPH_ORDER`) | **new** — not on the recorded list |
| **deferred host writeback** | **new** — not on the recorded list |
| **memory-pool quarantine/reuse** | **new** — not on the recorded list |
| **FA alignment/dispatch** (`GGML_HRX_FA_PARTIAL_ALIGN`) | **adjacent to lever 5** — lever 5 disabled FA entirely (`-fa off`); this varied the *partial alignment* with FA on, which changes which FA kernel is selected. Related but distinct: a reader could reasonably ask "isn't that the same lever?", and the answer is no, with the distinction stated. |

## The contract's question, answered

*"confirm the claimed mechanism is not one of them"* — the claims this session makes are **exclusions**, not
mechanisms, and **none of them is a recorded lever**:

- five are straightforwardly new (read path, graph reorder, writeback, pool reuse, and the AQL-policy read);
- two are **adjacent** to recorded levers (host side ↔ lever 1; FA alignment ↔ lever 5) and are distinguished
  above, with the reason each is a different test rather than a re-run.

And the one *fix* claim this session makes is a **non**-claim: no forward-only fix is landable, so there is no
proposed fix for the contract's "if it is, the fix is rejected" clause to bite on.

## Also worth stating

Lever 5 (`-fa off`) is the closest thing on the record to my FA-alignment exclusion, and the two together are
now a coherent pair: **disabling FA entirely does not stop the fault (lever 5), and neither does keeping FA on
while changing which FA kernel the dispatcher selects by lowering the partial alignment (this session)**. If
the fault lived in the FA path at all, at least one of those two should have moved the rate.
