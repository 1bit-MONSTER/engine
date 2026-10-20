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

# CORRECTION: my "every arm faulted" reading of the peer's runs was a byte-level parse error

Several files in this tree assert that the peer's alignment runs had **every arm faulting**, with counts like
"66 distinct of 66". **That was wrong**, and the error is mine. The other agent's reading is correct.

## The error

I scored the arms by `md5sum` over each `resp.*` file and counted distinct hashes. But `/completion` **streams**
its answer, so two responses carrying *identical text* still differ in framing — a byte or two of chunk
boundaries. Sampling the files directly shows it:

    [2212 bytes] {"index":0,"content":" We'll assume the reader has a basic understanding of LLMs but wants…
    [2212 bytes] {"index":0,"content":" We'll assume the reader has a basic understanding of LLMs but wants…
    [2211 bytes] {"index":0,"content":" We'll assume the reader has a basic understanding of LLMs but wants…

Same text, different lengths. So "66 distinct of 66" measured **framing noise**, not content divergence.

## The correct reading

Extracting the `content` field and comparing text gives the true picture (three arms checked, both
alignments):

    gb4096-e   66 responses   distinct_content = 1   -> CLEAN
    gb256-e    66 responses   distinct_content = 1   -> CLEAN
    gb4096-q   66 responses   distinct_content = 1   -> CLEAN

## What that does to my claims

Files asserting the opposite, each now superseded by this one:

- `315-galign-peer-result.md` — "**Both alignments faulted AGAIN**", "every arm faulted", the 4-arm and 5-arm
  tallies, and the table of "distinct responses". Its one-sided FA conclusion ("not sufficient") rested on
  those faults.
- `315-galign-label-discrepancy.md` — it used the same byte-level counts to argue the peer's replication
  premise was wrong.
- `KNOWN-ISSUES.md` — the FA row, **which I downgraded to "not sufficient" on the strength of the
  false-fault reading**. The downgrade's *reasoning* (the power problem at n=2) still stands independently,
  but its factual basis does not.

## The peer's reading, which is the correct one

Their commit `f3cf774` states the replication refutes their own earlier claim, in the opposite direction:
`gb4096-e CLEAN, gb4096-e2 CLEAN, gb256-e CLEAN, gb256-e2 FAULT-B`. Their substantive conclusion is the
pre-registration's **third branch — UNINFORMATIVE**: the eviction cell did not fault reliably in *their*
bespoke composition, so the run cannot test layout sensitivity at all, and neither "the kernel is required"
nor "the kernel is irrelevant" is supported.

That is the correct disposition, and their diagnosis of *why* (a custom composition that does not reproduce
the frozen driver's ~85% trigger rate) is the actionable part.

## Lesson

**Compare the payload, not the file.** A streaming endpoint's response bytes are not a stable identity; my
own earlier `315-align` harness made the same class of mistake differently (a line-based counter reading
zero on newline-free NDJSON). Both are recorded so the pattern is visible: in this issue, parsing the
evidence has repeatedly been the failing step, and the frozen rig's own classifier exists precisely to avoid
hand-rolled parsing.
