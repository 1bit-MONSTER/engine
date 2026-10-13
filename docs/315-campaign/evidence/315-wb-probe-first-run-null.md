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

---

# Take 2 also produced zero records — and this time the probe was sound, so the result is real

The corrected probe (`~/wt/315wb-probe2.sh`) fixed take 1's three defects and ran on an idle box. It issued
**10 varying prompt lengths** so the recorded graph could not be reused:

    10 requests issued, 10 decodes completed, 10 "graphs reused" events
    hrx-wb-fp records: 0
    host staging / download language in the server log: 0 occurrences

**Zero writeback records from a sound probe.** Unlike take 1, this is a real measurement: the graph was
rebuilt on every request, so `download_prepared_host_staging` was re-entered each time, and it still queued
nothing.

## The instrumentation is genuinely reachable

Checked before drawing any conclusion, because "instrument silent" and "instrument absent" are different
findings:

    strings libggml-hrx.so | grep hrx-wb-fp          -> 1   (the log format string is compiled in)
    nm -C libggml-hrx.so | grep fingerprint_writeback_log -> 1   (the function exists)
    nm -C libggml-hrx.so | grep add_host_writeback        -> 1   (the queue entry point exists)

So the path is compiled in and callable; it simply is not taken.

## What this means, at the strength it supports

On this workload — GLM-4.7-Flash decode via `llama-server`, ~10 varying requests, no regime — the host
staging **download** path is not entered, so `pending_host_writebacks_` is never populated and the deferred
window never opens.

That is a **negative for the path under these conditions**, and it is weaker than it sounds:

- The gate is `staging.download = access.write && !binding.graph_input` **with** `binding.host_data != nullptr`.
  The recorded trace (`host_visible=1 direct_host_binding=0 memory_type=0x52`) says the logits *are*
  host-visible, but that trace was taken on a **different build and configuration** (`GGML_HRX_BUFFER_TRACE`
  on `315instr3`/`315fix` trees), and whether the same holds in `315coherent-build` under this exact load
  has not been confirmed.
- No regime was involved. Whether the path is entered under GTT pressure, with neighbours, or with the
  eviction forcer is untested — the same gap that made the `315align` run uninformative.

So the honest state of this lead is: **not reached in the one configuration tested, not yet tested in the
configuration that matters.** The next step is to run the buffer trace (`GGML_HRX_BUFFER_TRACE=1` alongside
`GGML_HRX_WRITEBACK_FINGERPRINT=1`) in the *coherent* build to establish whether the logits are host-visible
there at all — if they are not, this lead closes by construction rather than by absence of output.
