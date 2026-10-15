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

# RETRACTION: there is no fixed "32 KV entry" boundary. The divergence step varies (7, 49, …) — and the detector has a NaN confound

The `315-MECHANISM-faceb-diverges-at-first-32-entry-attention-step.md` finding was **over-fit to one arm**, and I
am withdrawing the part of it that generalised.

## What was claimed, and what the population says

The claim was that the divergence occurs at logits index 7 — 25 prompt tokens + 7 = exactly **32 KV entries** —
and that this is "the first clean power-of-two boundary", i.e. the first point at which the fault *can* appear.

Running the same divergence detector over **all 54 arms** that carry `hrx-logits-fp`:

| divergence index | arms | absolute position |
|---|---|---|
| 1 | 1 | 26 |
| 7 | 1 | 32 |
| 16 | 1 | 41 |
| 49 | 1 | 74 |

**Four events, four different indices.** Index 7 is not special, "32 KV entries" is not a boundary, and the
reasoning that connected it to the record's other 32s (`1984 = 31×64`, `2048 = 32×64`) was pattern-matching on a
coincidence. **Retracted.**

## And a confound in the detector itself

Any NaN in the logits changes the hash, so a hash-based divergence detector fires on **NaN** as well as on the
degeneration Face B describes. Splitting the four events by whether the arm's logits contain any `nan > 0`:

| arm | index | `nan>0` in arm | what it actually is |
|---|---|---|---|
| `315inp2-r1e` | 7 | **0** | **genuine non-NaN divergence** (the Face-B case) |
| `315inp2-r2e` | 49 | **0** | **genuine non-NaN divergence** |
| `315lout4-r2e` | 1 | 1 | NaN-driven — hash change explained by the NaN |
| `315nan-p1l` | 16 | 1 | NaN-driven — hash change explained by the NaN |

So there are **two** genuine non-NaN divergence events on the record, both in eviction arms, at indices **7 and
49** — different steps of different prompts. **The detector must be gated on `nan == 0` to speak to Face B at
all**, and that gating has now been done rather than assumed.

## What survives, and it is still the strongest thing on the record

1. **Identical input, intermittent outcome.** In `315inp2-r1e` prompt A was answered four times in one process;
   three runs are **bit-identical** in the logits, the fourth diverges. That is a real state difference for
   identical input, and it is the fault.
2. **The divergence is upstream of sampling.** The *logits hash* differs, so the logits values differ; the
   sampler did not introduce it.
3. **The fault is finite.** All 423 logits readings in that arm are `nan=0`. Face B involves no NaN.
4. **Logits are deterministic across arms**, so a hash difference is signal, not noise — verified by matching
   the arm's step-0..3 hashes against an independently-taken sample from a different arm.
5. **Face B's HTTP 500 is the server's own response** to a degenerate generation
   (`"The model produced output that does not match the expected Content-only format"`), with an empty body —
   which is why `utf8_ok`/`json_ok` are true on that row. **Invalid UTF-8 is impossible here:** the garbage
   contains U+FFFD, a valid codepoint.
6. **The corruption step varies**, which is *also* informative: it is not a fixed structural boundary, it is a
   **per-step probabilistic event** — the signature of a race in the decode loop, not of a fixed-size region.

## What this leaves

The mechanism is a **per-decode-step intermittent, finite, non-NaN divergence that occurs in eviction arms**.
The step at which it appears varies, so no boundary-based test can name it; what can is a **per-step paired
read** at the divergent step. That is exactly what `patch-input-f32-sync-twin.py` enables: for the call whose
logits hash diverges, compare the plain-memcpy and synchronised reads of the tensors it depends on. If the two
reads differ there, the fault is a visibility/ordering defect; if they agree, it is genuine device state and the
race is on the device side.

Two honest notes on this directory: this is the **second** claim I have had to retract after extending a
single-arm observation to the population (the first was "the onset is one row / layer 41"), and the pattern is
the same — a sharp result in one arm that does **not** survive being counted. Both retractions are recorded
rather than quietly edited. And the mechanism remains unnamed.
