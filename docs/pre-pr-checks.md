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
# Before you ask for a merge

Six commands, all host-side, none needing a device or a build. Every one of them answers a question that
`mergeable=true`, a green CI tick and a careful read of your own diff do not.

Run them from the repository root on the branch you are about to hand over.

| command | red means |
|---|---|
| `python3 tools/copyright.py --check` | a file has no notice, or has two |
| `python3 tools/ci/tests_referenced_check.py` | a test, tool self-test or python test exists that nothing runs |
| `python3 tools/ci/ci_targets_check.py --check` | a CI row names a target or path that does not exist |
| `python3 tools/ci/registration_loss_check.py --check --base origin/main` | your branch would drop rows `main` has |
| `python3 tools/ci/tmp_path_check.py --check` | a new `/tmp` or `/dev/shm` path, where builds and dumps belong under `~/wt` or `~/.cache/scratch-*` |
| `python3 tools/ci/submodule_pin_check.py --check` | a submodule's checkout is not the commit the tree pins |

Two more, for the moments they are for:

| command | when |
|---|---|
| `python3 tools/ci/conflict_triage.py . <their-branch>` | before resolving a merge with many conflicted files: it classifies them by kind (content, add/add, mod/del, rename) from the index, which text tools cannot see in a file whose conflict has no markers |
| `python3 tools/ci/hrx_f16_narrowing.py --check --root <tree with the .loom corpus>` | when the change could touch f32→f16 narrowing; it lists every body that narrows |

## Why the merge-base one is not the same as "it merges"

A PR's diff is computed against its **merge base**. If `main` moved under you, every row another lane landed shows in your
diff as a removal, and a resolution that takes your side deletes it. Measured three times on 2026-11-11/12: `main-whisper-tool`
(#2183) was five commits behind and its diff removed 48 of `main`'s 429 registrations, two rows of which had been merged the
same day; #2091's naive resolution would have dropped eight `policy_verdict` references; and one of this lane's own branches
carried an already-merged commit for the same reason.

`registration_loss_check` separates the three states, so its message is actionable rather than alarming:

- **rows deleted by your branch** (present at the merge base, absent at your head) — a merge honours the deletion; say in the
  commit message that you meant it;
- **stale and conflicting** — the risky combination: the conflict sits in a file full of other lanes' rows, and taking your
  side drops them; rebase and keep every row `main` has;
- **stale and clean** — behind, not destructive: the merge keeps `main`'s rows. Rebase when convenient.

## These are not CI rows

Every check above runs in CI except `registration_loss_check` and `conflict_triage`, which compare two revisions — a build only
ever has one. That is why they are here: the merge-base question can only be asked before the merge.

The self-tests of all of these *are* ctest rows (`tools/ci/*.py --self-test`), so a check that stops working turns the build red
even though its own verdict is a manual one.
