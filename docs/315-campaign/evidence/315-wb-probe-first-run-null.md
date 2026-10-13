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

# First writeback probe run: ZERO records — and my own tally output was wrong

The stalled lead's first measurement ran at 15:16:24 on an idle box. It produced **no evidence**, and the
script that reported it produced a **false positive** that must be corrected before either is believed.

## What the probe reported, and why it was wrong

The script's own output included:

    === rec_eq_sync tally ===
          1 rec_eq_sync=0

which reads as "the queued source changed while queued" — the exact hazard this lead is about. **It is an
artifact of my own script.** The tally line was produced by `grep -o "rec_eq_sync=[01]" "$L"`, and the
literal string `rec_eq_sync=0` also appears in the script's **own echo header**
(`=== sample records (rec_eq_sync=0 means the source CHANGED while queued) ===`), which had been written
into the same `$L` file with `tee -a`. So the grep matched the header, not a record.

Verified: `grep -c "\[hrx-wb-fp\]"` on the same log returns **0**. There are no writeback records at all.

Recorded plainly because the failure mode is exactly the one this issue keeps punishing: **a tally that
could not have been anything but positive**, in a log the tally's own text was written into.

## The real result: the path was not exercised

`graphs reused = 7` on each of the three decodes. The three requests were **identical** (same prompt, same
`max_tokens`, `temperature=0`), so llama.cpp reused the recorded graph and `download_prepared_host_staging`
was not re-entered. No downloads ⇒ no queued writebacks ⇒ no records. The writeback path is only exercised
when the graph is (re)constructed and its host-staging bindings are materialized.

So this run does **not** show the deferred writeback is unused. It shows my probe never asked the question.
The honest status of the lead is unchanged: **still unmeasured**.

## What the next probe must do differently

1. **Vary the input across requests** so the graph is rebuilt rather than reused — different prompt lengths
   and different content, which is what the real detector does (`315-detector.py` issues 8 identical greedy
   requests *separated by different-length distractors*, then 50 mixed-length requests). A probe of three
   identical one-word prompts is the one shape guaranteed to hit the cached graph.
2. **Count only anchored record lines** (`^\[hrx-wb-fp\]` or `grep -c "\[hrx-wb-fp\]"`), and never grep a
   file the script's own headers are written into. The header text and the record text must not share a
   pattern.
3. **Assert the instrument fired at least once** before reporting any tally, so a silent path is reported
   as "not exercised" rather than as a zero tally that reads like a clean result.

All three are fixed in the revised probe (`~/wt/315wb-probe2.sh`).

## Process note

This is the second time in this goal that a measurement's *reporting* was the defect rather than the
measurement (the earlier one was the `pgrep` self-match scare). Both were caught before the number was
cited, and both are recorded in place rather than quietly dropped.
