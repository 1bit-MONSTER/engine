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

Eight commands, all host-side, none needing a device or a build. Every one of them answers a question that
`mergeable=true`, a green CI tick and a careful read of your own diff do not.

Run them from the repository root on the branch you are about to hand over.

**`scripts/pre-pr-checks.sh` runs the first six**, one line each (`--base <ref>` for the registration check, `--list` to see the
commands; a check that cannot run says `skip` and why, and an unresolvable `--base` is a usage error rather than a skip). The
seventh is an **author's check**, true or false exactly once — when you add a page — so it is not in the runner, and the section
below says why.

| command | red means |
|---|---|
| `python3 tools/copyright.py --check` | a file has no notice, or has two |
| `python3 tools/ci/notice_covers_deps.py --check` | a dependency that is fetched and compiled into the binary is not named in `NOTICE` |
| `python3 tools/ci/tests_referenced_check.py` | a test, tool self-test or python test exists that nothing runs |
| `python3 tools/ci/ci_targets_check.py --check` | a CI row names a target or path that does not exist |
| `python3 tools/ci/registration_loss_check.py --check --base origin/main` | your branch would drop rows `main` has |
| `python3 tools/ci/tmp_path_check.py --check` | a new `/tmp` or `/dev/shm` path, where builds and dumps belong under `~/wt` or `~/.cache/scratch-*` |
| `python3 tools/ci/submodule_pin_check.py --check` | a submodule's checkout is not the commit the tree pins |
| `git grep -l -F "$(basename <your new page>)" -- '*.md'` | nothing links to the page you just added — measured tonight: 45 docs, one with no inbound link, and it was the page whose whole subject is pre-merge checks |

Two more, for the moments they are for:

| command | when |
|---|---|
| `python3 tools/ci/conflict_triage.py . <their-branch>` | before resolving a merge with many conflicted files: it classifies them by kind (content, add/add, mod/del, rename) from the index, which text tools cannot see in a file whose conflict has no markers |
| `python3 tools/ci/hrx_f16_narrowing.py --check --root <tree with the .loom corpus>` | when the change could touch f32→f16 narrowing; it lists every body that narrows |

## A page nothing links to is a page nobody runs

`docs/pre-pr-checks.md` had no inbound link from any other `.md` for two hours after it merged: it was reachable only by knowing
its name. A new page is a claim that someone should read it, and the only mechanical half of that claim is a link — so before you
open the PR, grep for your own page's name and see someone else's file come back:

    git grep -l -F "$(basename docs/your-new-page.md)" -- '*.md'

Empty output means the page exists for whoever already knows about it, which is nobody.

**Why this one is not in `scripts/pre-pr-checks.sh`**: that runner is a loop over `python3` tools that carry their own self-tests,
and a repo-wide form of this check is a new tool for an invariant that measures **1 today — this page** — and 0 once engine#2350
adds its inbound link. That is the strongest argument for it being an author's step rather than a gate: **the number it reports is a
property of the tree at that moment, not of your change** — it flips from 1 to 0 when a *different* PR merges, which is not a
statement about anything the author did. What the author's own command answers is: does the page I am adding have a way in. Prefer the link from the page a reader is
already on when they need yours — here `CONTRIBUTING.md`, under the rules the page operationalises — rather than from an index
nobody opens.

## A private pin (`config/*.config`) is the one bump CI cannot check

Two of these files pin a **private** repository; they are the ones whose bumps need evidence:

| file | `repo=` | private? |
|---|---|---|
| `config/npu-runtime.config` | `1bit-MONSTER/1bit-MONSTER` | **yes** (`gh api repos/… --jq .private` → `true`) |
| `config/gpu-kernels.config` | `1bit-MONSTER/gpu-kernels` | **yes** |

The other two are **not** private pins, and assuming otherwise sends a reader looking for evidence that does not exist:
`config/llama-mirror-hold.config` holds a **bare sha** — which commit the mirror is held at, a decision rather than a fetched
pin — and `config/hrx-toolchain.config` names a **public** index (`index=https://nightly.repo.amd.com/rocm/whl-next/`, fetched by
`scripts/fetch-rocm-sdk.sh`).
`ONEBIT_GPU_PRIVATE` or `gpu-kernels` — so a green `build` on a pin bump proves the engine configures and builds *without* that
dependency and says nothing about the commit the pin names. `tools/check_pins.py` does not help either: it compares **submodule
gitlinks**, not `config/*.config` shas.

Three commands are the evidence, and they are cheap enough to run before merge (a fetched private pin was 3.4 MB and 1.4 s):

```bash
git ls-remote https://github.com/1bit-MONSTER/<repo>.git refs/heads/main     # note the tip
git fetch <repo> <tip> && git merge-base --is-ancestor <pinned> <tip>        # rc 0: the pin is still on the branch
bash scripts/fetch-private.sh config/<the>.config ~/.cache/scratch-<lane>/pin # rc 0, HEAD == the pinned sha,
                                                                             # both require= paths present
git -C ~/.cache/scratch-<lane>/pin diff --stat <the previous pin>..HEAD       # what the range actually contains
```

**The first check is reachability, not equality, and the difference is the whole point.** A pin is *normally behind* the tip —
`main` moves after the pin is taken — so "the pinned sha IS the advertised tip" flags almost every correct pin. Measured on the
gpu-kernels bump merged on 2026-11-13: the tip was `b0ed206` and the *previous, healthy* pin was `eb0f3839`, two commits behind and
still an ancestor, which `fetch-private.sh` checks out without complaint. What the check is *for* is a pin that no longer exists —
a force-push — so the test is `merge-base --is-ancestor`, and a pin that equals the tip is worth a second look for the opposite
reason (the branch moved after review).

Run the first two **from the branch that changes the pin**: `fetch-private.sh` reads the config out of the *working tree*, so the
same command in a checkout of an older commit fetches the older pin and returns **rc 0** — indistinguishable output, wrong answer.

That third command is the one a reviewer needs. A pin range is not "the fixes I named": measured on engine#2316,
`fd69590c..b0ed206` was **ten commits**, of which two were the f16-range fixes the PR closed, one was #572's routed GEMM, and the
rest were **two new ~400-line kernels, a new refusal path and an id renumbering**. A body that says "three fixes" asks a reviewer
to accept eight things they were not told about.

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

## The one that is not a command: build the target that contains the file you edited

Nothing above compiles anything, and a green `build` tick does not answer this question either -- it answers "the targets CI built
compile", which is a different set, built by a different runner, from a different checkout.

    engine#2327, 2026-11-13: the image refusal added a 400 body to app/forward_serve.cpp and it was one brace short. The PR's unit
    test built and passed -- it covered `npu::message_carries_an_image()`, the *predicate*. The PR merged, `main` did not compile,
    and another lane fixed it in 273f6e9df ("serve: main does not compile at the #2327 image refusal") before I knew.

The trap is that the test which seemed to be the evidence links a different translation unit from the one that broke. Ask which
target compiles the file, not whether the file looks right:

    cmake --build build-<cfg> --target <that target> -j 6

`forward_serve.cpp` is in the serve binary, not in the helper's test. If you cannot name the target, `grep` `CMakeLists.txt` for
the file name -- and if the answer is "the whole `1bit` binary", that is the answer, not a reason to skip it.

## These are not CI rows

Every check above runs in CI except `registration_loss_check` and `conflict_triage`, which compare two revisions — a build only
ever has one. That is why they are here: the merge-base question can only be asked before the merge.

The self-tests of all of these *are* ctest rows (`tools/ci/*.py --self-test`), so a check that stops working turns the build red
even though its own verdict is a manual one.
