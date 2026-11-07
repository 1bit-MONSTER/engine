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
# The eight tickets that produced the recorded tau=0.6 window

These are the in-tree inputs for the draft route's **default** chain gate: `Options::tau = 0.6`
(`npu/mtp/draft.h:162`). Every acceptance row published before them was `tau = 0`, so the regime a serve gets by
default had no measured acceptance, hit rate or tokens-per-pass figure — and because the gate stops the chain after the
first low-confidence draft, a `tau = 0.6` ticket offers depths 1-3 where a `tau = 0` one offers 1-4.

The window: 8 prompts x 64 tokens, `k = 4`, `tau = 0.6`, `--live --match`, one prompt per `npu-hold` ticket on
2026-11-06 (16:42-18:06 AST, array exclusive, `amdxdna` errors 0 in every arm). Greedy text was byte-identical to
plain decode on 8 of 8 prompts with 0 stream mismatches, and prompt 0 was measured twice (16:29 and 16:42) with
identical counts (31/35) — the only repeat in the set.

**Provenance.** Each `.txt` is the lines the parser reads, extracted verbatim from its ticket with
`~/wt/mtp-window/extract-fixture.sh` (the same `grep` the `mtp_clause55` README documents, which re-extracts those three
fixtures byte-identically):

```sh
pat="forward ready:|^\[ *[0-9]+s\] == [0-9]+ prompts x |prompt [0-9]+:|depth [0-9]+: |match p[0-9]+:|== live vs offline at the SAME|== ticket "
grep -E "$pat" ticket-tau06-p0-try1-20261106-164254.log > p0.txt   # 15 lines, prompt 0
# ... one per prompt: 165612 (p1), 170407 (p2), 172013 (p3), 172718 (p4), 173228 (p5), 174605 (p6), 175540 (p7)
```

The `== 8 prompts x 64 tokens, k=4 tau=0.60, hidden=post-norm` line is carried **inside** each file: it is the window's
identity, and it is what lets `tools/mtp_depth_pool.py` refuse to pool these tickets with a `tau = 0` one (engine#1277).

`tests/mtp_depth_pool_test.py` pools these eight and fails if any of the six `27b-mtp-tau06-*` leaderboard records stops
agreeing with them, or if a depth's counts drift by one.
