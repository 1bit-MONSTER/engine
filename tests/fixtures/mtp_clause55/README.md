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
# The three tickets that produced the recorded 8x64 window

These are the only in-tree inputs that let `tools/mtp_depth_pool.py --selftest` check its **known answer**: the
per-depth live-vs-offline table the brief is judged on (8 prompts x 64 tokens, k=4, tau=0,
`+0.00 / +0.55 / +0.09 / +1.08 pp` at depths 1-4, 75.14 / 63.24 / 73.26 / 84.13 %). Without them the check needs the
original ticket logs, which live in one home directory on one box — so CI could break the tool without noticing.

**Provenance, so these are not a restatement of the answer.** The three `.txt` files are the lines the parser reads,
extracted verbatim from the `npu-hold` tickets of 2026-11-05 (the same tickets whose numbers the lane recorded):

```sh
pat="forward ready:|prompt [0-9]+:|depth [0-9]+: |match p[0-9]+:|== live vs offline at the SAME|== ticket "
grep -E "$pat" ticket-clause55-p0-20261105-115131.log > p0.txt    # 15 lines, prompt 0
grep -E "$pat" ticket-clause55-p1-4-20261105-115551.log > p1-4.txt  # 35 lines, prompts 1-4
grep -E "$pat" ticket-clause55-p5-7-20261105-120713.log > p5-7.txt  # 28 lines, prompts 5-7
```

**Why `.txt` and not `.log`**: `.gitignore:19` is `*.log`, so the first cut of this directory committed the README and
silently left all three tickets behind — the test then passed on the machine that had them lying around and failed in CI
with `tests/fixtures/mtp_clause55 is incomplete`. A fixture that can be dropped by a rename or a re-clone is worse than
one with a different suffix, so the files carry `.txt` and `tests/mtp_depth_pool_test.py` asserts with `git ls-files`
that all three are **tracked**, which fails locally instead of only in CI.

Everything the tool ignores is dropped — the engine chatter, the ELF loads, the per-prompt `per-depth pN:` compact
lines, the phase split. Nothing the tool reads is edited: the `[  Ns]` elapsed stamps, the `[live joint …]` brackets
and the `match pN:` headers are exactly as the tickets printed them, and the pool still has to reproduce the recorded
counts from them (that is what makes it a known answer rather than a fixture written to agree with the code).

`tests/mtp_depth_pool_test.py` runs `--selftest` over these three and fails if the pooled cells drift by one count.
