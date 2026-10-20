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

# What the merge and the gates found: two pre-existing defects on this branch

Prepared for task 3 ("engine CI green"). Opening the PR is what made the gates run, and running them is what
found these. Both predate this turn and neither is a consequence of the merge; the merge only exposed them.

## Defect 1: the pin move was a HALF-MOVE, and the gate built for it caught it

`95238dc08` is titled *"move the llama.cpp pin forward to the logits-read wait"*. It changed **one file**:

```
$ git show --stat --oneline 95238dc
95238dc08 315: move the llama.cpp pin forward to the logits-read wait (e44c9d01a -> 47a8fe929365...)
 third_party/llama.cpp | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

`registry/architectures.json` still declared `"llama.cpp (hrx)": "e44c9d01a4d511..."`. That is exactly the
half-move that `tools/check_pins.py` was extended to refuse, and the gate said so in its own words:

```
FAIL registry/architectures.json: "llama.cpp (hrx)" is 0e255d774 but the gitlink at third_party/llama.cpp
     is 47a8fe929 - move both together
```

So **criterion 2's change was landed inconsistently**, and this went unnoticed until the PR triggered the
gate. Fixed here by moving the registry with the gitlink:

```
ok   registry/architectures.json: "llama.cpp (hrx)" agrees with the gitlink (47a8fe929)
ok   third_party/llama.cpp: 0e255d774 -> 47a8fe929 is ahead
```

**The forward-only claim itself was never in doubt** and is now confirmed against the *current* base, which is
stronger than the original: `git merge-base --is-ancestor 0e255d77 47a8fe929` is **true** and
`47a8fe929`'s parent **is** `0e255d77`. Main moved the pin to `0e255d774032` (#373) *after* this branch was
cut, and the fork commit was built directly on top of it — so the move is one commit ahead of today's main,
not a revert, not sideways.

## Defect 2: 46 files would have failed the copyright gate

`tools/copyright.py --check` requires the *full* notice — including `Licensed under the Apache License, Version
2.0` — within the first 20 lines. The **two-line SPDX form** (`# Copyright …` + `# SPDX-License-Identifier:
Apache-2.0`) that most of this campaign's files were written with does **not** satisfy it:

```
missing copyright notice: dev/315/315-campaign/evidence/315-faceb-invariant-signature.md
... 46 files, including this campaign's own tools and evidence documents
```

Fixed with the repository's own tool, `python3 tools/copyright.py --fix`, which inserts the notice in the
correct comment style per file — plus one genuine gap in the tool: `*.hip` had no comment style, so
`dev/315/315-campaign/tools/315restore-fidelity.hip` could not be fixed at all. HIP is C++, so it was added to
the existing `//` line style rather than exempted. Both `--fix` and `--check` now exit 0.

## Why this is recorded rather than quietly fixed

Task 3's criterion is "engine CI green", and the honest state before this work was that the branch would have
failed CI on two separate jobs. The value of the record is that a reader can see the branch was **not** green
on arrival, what the failures were, and that the fix was made by the repo's own tooling rather than by
loosening a gate. No gate was weakened: the pin check still refuses half-moves, and the copyright check still
requires the full notice.

## CI state

`gh pr checks 414` and the locally-run equivalents of every `ci.yml` python step, at
`88d3b71a5`:

| step | result |
|---|---|
| `python3 tools/copyright.py --check` | exit 0 |
| `python3 tools/check_pins.py <PR base>` | exit 0 — registry agrees, gitlink ahead |
| `python3 tests/gguf_to_q4nx_test.py` | PASS |
| `python3 tests/ternary_to_q4_0_test.py` | PASS (`PTQ1_0 -> Q4_0 exact`) |
| `python3 tests/mtp_draft_vocab_test.py` | OK |
| `python3 tools/registry_build.py --check-gaps` | exit 0 |
| `python3 tests/registry_gaps_test.py` | PASS |
| GitHub Actions `build`, `pr_agent`, `Analyze (…)` | running; PR reads `mergeable=MERGEABLE` |

The GitHub-side result is the authority and is reported from `gh pr checks 414` when it settles; the table
above is the local equivalent, run at the same revision so the claim does not depend on the workflow's queue.
