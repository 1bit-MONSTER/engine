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

# Face B is not one defect — it is two populations, and its majority is not a fault. The two faces never co-occur.

All of this is from the existing archive plus the runtime source. **No GPU was used.** This is a
reclassification of the record, and it is decision-relevant: the acceptance criterion "zero faults on both
faces" as currently worded cannot be measured, because the Face-B label is a mixture whose larger part is a
benign format failure.

## The record's instrument inventory (first time enumerated)

| tag | records | what it is |
|---|---|---|
| `[hrx-ids-fp]` | 3,412,028 | unsynchronized host `memcpy` of the MoE id tensor |
| `[hrx-ids-sync]` | 1,160,376 | same tensor via `hrx_synchronous_d2h` |
| `hrx-copy site=` / `hrx-copy-LOGITS` | 164,552 each | the logits copy probe |
| `[hrx-tensor-fp]` | 134,180 | device-side tensor fingerprint (call/name/op/nan/of/ne0/first/row/hash) |
| **`[hrx-logits-fp]`** | **104,720** | **host-side logits fingerprint (call/n_vocab/hash/nan/sentinel)** |
| `[hrx-finite-fp]` | 1,698 | hash over finite values only |
| `[hrx-buf]` | 351 | allocation records |
| `[hrx-dl-fp]` | 146 | download fingerprint with probe-vs-copy hash comparison |
| `[hrx-sentinel]` / `[hrx-producer]` | 90 / 24 | sentinel and producer probes |

## The real markers, and the ones that do not exist

Grepping for the objective's language, over every arm log:

    "HTTP 500"          0 arms
    "Invalid UTF"       0 arms      (consistent with 315-invalid-utf8-never-observed.md)
    HSA_STATUS_ERROR   12 arms      <- Face A
    MEMORY_FAULT        3 arms      <- Face A (subset)
    "Content-only output" 77 arms   <- the operative Face-B marker

The Face-B marker is `common_chat_peg_parse: unparsed Content-only output: <text>`, and the `<text>` is the
model's actual output. Reading it is what splits the population.

## Face B is two populations of about equal size

    GARBAGE-text arms : 39
    prose-text arms   : 38

- **`prose` (38 arms)** — coherent English: `"We'll assume the reader has a basic understanding of"`, verbatim,
  in roughly twenty separate arms. **That is a normal model answer.** The harness asked for structured JSON,
  the model answered in prose, and the PEG parser rejected it. **This is a format-compliance failure, not
  corruption** — and it cannot be a memory fault, because the text is coherent.
- **`GARBAGE` (39 arms)** — genuine corruption: token soup (`"over  of of - ore of),         •1- •, •,''3 or • •3 "`),
  repetition loops (`"gaps gaps gaps gaps gaps gaps gaps gaps gaps"`), replacement characters
  (`"the}:}:}甚;"`, `"一个}: �甚22}$2 ;"`), and the telling **coherent-then-dissolve** form:
  `"We'll assume thet ( size（, ((A(A.,,//值*（ (名 (。 de*。*"` — correct for a few tokens, then degrades.

## The two faces are disjoint, and Face B is never NaN

    GARBAGE-text arms : 39   device NaN: 0   face A marker: 0
    prose-text arms   : 38   device NaN: 0   face A marker: 0
    device tensor-fp NaN present at all: 0 of 89

and every one of the 104,720 `[hrx-logits-fp]` records that lands in a Face-B arm reports **`nan=0`**.

**No arm on the record shows both faces.** Face A arms have `empty` output — they crash before emitting text
(all six `v315t-r*` arms, `315probe3-p1c`, …). Face B arms never carry a Face-A marker.

**So "both faces" are not two symptoms of one intermittent event.** They are two disjoint populations, and the
Face-B population is mostly not a hardware fault at all.

## Two record artefacts that must be retired

1. **`oor` and `dup_pairs` in `[hrx-ids-fp]` are not corruption signals.** The probe checks each id against
   `ne0 = node->ne[0]`, and for `ffn_moe_argsort` that is the **top-k count (observed: 4 or 64)**, not the
   expert-table size — so *valid* expert indices are flagged `oor` permanently. The observed `dup_pairs` totals
   confirm it: **977,666,493 duplicate pairs in the 23 CLEAN arms** versus 10,333,698 in the probed Face-B arms.
   A metric that is three orders of magnitude worse in the clean population carries no information. Any reading
   built on `oor`/`dup_pairs` should be withdrawn, including the "clean permutation" framing.
2. **`hrx-ids-fp` vs `hrx-ids-sync` are in the same arm** (3,412,028 vs 1,160,376) and agree on all 253,598
   paired readings — already recorded. But `hrx-tensor-fp`-on-`result_output` (8 arms) and `hrx-logits-fp`
   (56 arms) have an **empty intersection**: no arm carries both, so the device-vs-host logits comparison is
   still not possible on the record. Same disjoint-instrument defect as before.

## What this changes, and what it does not

**Changes.** The objective's Face B — *"invalid-UTF-8 output surfacing as HTTP 500 parse failures"* — describes
a phenomenon with **zero occurrences** on the record: no invalid UTF-8, no HTTP 500. What exists is
`Content-only output`, and ~half of it is the model declining to emit JSON. Any acceptance campaign that counts
all `Content-only` outcomes as faults will count benign format failures as failures and can never reach
"zero faults"; conversely a campaign that excludes them silently narrows the criterion. **This needs
`/goal-tweak` before a ≥60-arm campaign can be built on it** — the decision is the user's, not mine.

**Does not change.** Face A is real and well-marked (12 arms, `HSA_STATUS_ERROR`; 3 of them `MEMORY_FAULT`),
it has `nan=0`-free `empty` output, and it remains the thing to explain. The 256-byte granularity measurements
sit on hidden-state tensors and are unaffected. Nothing here names Face A's mechanism.

**And the one new lead for the genuine half of Face B.** The `GARBAGE` arms show finite logits and
*coherent-then-dissolve* text, which is the signature of a corruption that enters **partway through
generation** — the prompt is answered correctly for several tokens and then degrades. That is a property of the
**decode loop state** (KV cache / hidden state carried across steps), not of a single forward pass, and it is
consistent with the record's own PR #90 finding in the input direction. That is where the next mechanism hunt
should look, not at the prefill path where the 256-byte readings were taken.
