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

# Which binary is which, and the trap that nearly mixed them up

Recorded because the campaign's validity depends on the two builds being what they are claimed to be, and
the first attempt at the fixed build silently produced an **unfixed** one.

## The trap

`git worktree add -f <path> <rev>` was run in a script whose failure would have been fatal — but the
command's output was piped through `tail`, so the pipeline's exit status was `tail`'s and `set -e` did not
fire. The worktree path already existed, the add failed, and the script went on to **configure and build the
pre-existing tree**, which was at `e44c9d01a` with no fix in it. The build succeeded and reported a clean
binary. The only thing that caught it was a `grep` for the fix marker in the source, which printed nothing
where it should have printed `1`.

This is the same class as the trap the issue already records ("Stale incremental builds — several earlier
measurements on this issue were retracted as build artifacts"). It is recorded here rather than quietly
redone.

**The rule that follows:** never accept a build's revision from the build's own exit status. Assert the
marker in the **source** before configuring, and assert the instrument markers in the **binary** after
building. Both are done in the replacement build.

## What each artefact actually is

| path | revision | fix present | purpose |
|---|---|---|---|
| `~/wt/315logwait-build` | `47a8fe929` (the landed fix) | **yes — asserted before the build** | **cell F** (fix), clean, no instruments |
| `~/wt/315fixlog-build` | `e44c9d01a` | **no** (`grep` for the fix marker returns 0) | **cell C** (control), clean, no instruments — it is the accidental first build, and it happens to be exactly the control the campaign needs |
| `~/wt/315fp-build` | `e44c9d01a` | no | the **instrumented** diagnostic build (twin + read-twice control + wb-read probe). **Not for the campaign:** the pre-registration says instruments off in both cells |
| `~/wt/315coherent-build` | `llama-315coherent` | no | lever-1 cell; unrelated to this campaign |

Both campaign cells are instrumented-**free**: binary markers `hrx-f32-sync` and `hrx-wb-read` both read `0`.
