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
<!-- Copyright 2026 bong-water-water-bong
     SPDX-License-Identifier: Apache-2.0

     engine#315 evidence: the engine's own CI steps, run locally at the current pin on 2026-10-09 so
     task-6 has a baseline and so the gates that will judge the eventual fix are known to work in this
     environment. Stored as Markdown because the repository's CI requires a copyright notice on every
     file. -->

# engine CI baseline at pin `e44c9d01a`

Run on `ryzen` in the worktree `~/wt315-prereg` (`1bit/315-preregistration`), 2026-10-09, at
`third_party/llama.cpp = e44c9d01a4d5110cecaf45472a717f46cd1abbf9`. Every step below is a step of
`.github/workflows/ci.yml` or a `ctest` test that `ci.yml`'s final `ctest` runs, executed directly
rather than through the workflow.

| step | command | result |
|---|---|---|
| copyright | `python3 tools/copyright.py --check` | **exit 0** (it failed before `b777bdc`: three `evidence/*.txt` files had no comment style and the copied probe script had no notice) |
| pin gate | `python3 tools/check_pins.py 6d5e470` | **exit 0** - `third_party/llama.cpp: 26330cc20 -> e44c9d01a is ahead` (plus hrx-system, xdna-driver, laya) |
| pin gate, longer span | `python3 tools/check_pins.py beec5e6` | **exit 0** - `third_party/llama.cpp: cf4dfd801 -> e44c9d01a is ahead` |
| gguf -> Q4NX | `python3 tests/gguf_to_q4nx_test.py` | PASS |
| ternary -> Q4_0 | `python3 tests/ternary_to_q4_0_test.py` | `PASS: PTQ1_0 -> Q4_0 exact, other tensors and metadata kept` |
| MTP draft vocab | `python3 tests/mtp_draft_vocab_test.py` | `OK` |
| registry gaps (ctest `registry_gaps`) | `python3 tools/registry_build.py --check-gaps` | exit 0 - `none of the 15 reviewed significant classes is mapped without a reason` |
| registry gaps fire (ctest `registry_gaps_fire`) | `python3 tests/registry_gaps_test.py` | PASS (2 checks) |
| build (`onebit` + the 14 test targets, gcc-15) + `ctest` | not run | nothing to regress yet: there is no fix revision to compare against, and this is the CPU-only half of the workflow |

## What the pin gate does and does not cover

`tools/check_pins.py` compares **submodule gitlinks** between a base commit and `HEAD` and asks GitHub's
compare API whether the new commit is ahead of the old one. It is the machine enforcement of the
objective's "must not be a revert" clause, and it works here (rows 2-3).

What it does **not** cover: the hrx revision declared in `registry/architectures.json`
(`"llama.cpp (hrx)": "e44c9d01a4d5..."`). `tests/registry_gaps_test.py` reads that file, but it checks
class coverage, not the pin, and nothing in `tools/` or `tests/` ties the registry value to the
gitlink. So a change that bumps one and not the other passes CI while leaving the tree inconsistent.
Task-7 has to move both, by hand, and cannot rely on the gate to catch a half-move.

## The pin lineage, and why the fix cannot be "move the pin"

`git log -- third_party/llama.cpp`:

| commit | move | title |
|---|---|---|
| `4aa4ac2` | e57beb9721af | HRX bump: HIP plumbing hook, HRX fixes, Flash-Next |
| `11d6e2f` | 522dab4789ce | long-prompt fix; hrx-system moves to our fork (Loom NaN backport) |
| `c18b319` | f2099e9b7 | AMD's core with our ggml-hrx |
| `beec5e6` | f2099e9b7 -> cf4dfd801 | **Avoid subgroup votes in HRX decode-split flash attention** (#342) |
| `6d5e470` | cf4dfd801 -> 26330cc20 | **pin llama.cpp to the shared-memory FA tile visibility (engine#314/315)** (#347) |
| `805d30f` | 26330cc20 -> e44c9d01a | AMD's tested pair |

`6d5e470` is already a #315-motivated pin move, aimed at exactly the flash-attention visibility that
sections 26-27 of `315-mechanism.md` now falsify as the locus. So "bump the pin to the next AMD pair"
is not a fix for this issue - it has been tried and the fault survived it (section 22 finding 2 records
that the coherence and ordering levers failed; section 26 kills the FA masked-row lever outright).
The eventual change has to be a new fork commit that is a **descendant of `e44c9d01a`** and carries
whatever the mechanism work ends up naming.
