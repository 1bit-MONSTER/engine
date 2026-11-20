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

# The checks in `tools/ci`: what they defend, and how to run them

**40 single-purpose checkers live in `tools/ci/`, and nothing listed them.** A lane that has just been burned by a
class of mistake — a lost commit, a removed diagnostic, a route nobody registered — has no way to discover that a
check for it already exists, and writes a forty-first. This page is the index. It names the *questions*, not the
files, so it does not rot when one is added.

## Run them

```sh
for f in tools/ci/*.py; do printf '%s: ' "$(basename "$f")"; python3 "$f" --self-test >/dev/null 2>&1 && echo ok || echo FAIL; done
```

Every one of them carries `--self-test` with hermetic fixtures, and `tools/ci/tool_selftest_wired_check.py` fails when
a tool that has a self-test is referenced nowhere — so a check whose arms stopped running says so.

**Exit codes are the same shape across the family, and the second one is the point:**

| code | meaning |
|---|---|
| 0 | checked, nothing found |
| 1 | the thing the check exists for was found, named |
| 2 | **cannot tell** — the input was absent, unreadable, or in an unexpected shape |

A `2` is never a pass. It exists because the recurring bug in this tree has been a check reporting `0` about a
population it never examined (`ctest_skips` on a shallow clone, `guard_loss_check` counting refusals only, a fault
census counting lines instead of events).

## The questions, by family

**Is the thing we shipped the thing we reviewed?** `merged_tail_check` (a commit pushed after its PR merged),
`pr_carry_check`, `pr_verdict`, `pr_review_queue`, `branch_stale_check`, `moving_ref_check`. The rule they serve: *a PR
merges at its reviewed head; a commit pushed after the merge is silently lost.*

**Does a guard still guard?** `guard_loss_check` (a removed refusal or diagnostic), `conflict_marker_check`,
`dupdef_check`, `doc_section_loss_check`, `registration_loss_check`, `registry_loss_check`, `tests_referenced_check`,
`tool_selftest_wired_check`. These are loss checks: they compare what *was* there against what is, and refuse to call a
shrunken population clean.

**Is the census measuring the right population?** `census_counters_check`, `stale_expectations_check`,
`fork_hrx_rows`, `ctest_skips`, `ctest_known_fail`, `hip_f16_audit`, `hrx_f16_narrowing`. The distinction they encode:
a *mention* is not a *read*, a *line* is not an *event*, and a *baseline entry* is not a *skip that is fine*.

**Are the routes, fields and launches consistent?** `serve_routes_check`, `warmup_route_check`, `server_launch_check`,
`npu_decl_check`, `ci_targets_check`, `cmake_configure_check`. Warmed paths must be registered routes; fields a
request sends must be fields something reads; a test that starts a server must take the port lock.

**Pins, files and containers.** `config_pin_check`, `submodule_pin_check`, `file_create_check`, `notice_covers_deps`,
`hrx_build_scope`, `kernel_id_selftest`, `token_rate_floor`, `medians`, `lane_syntax_check`, `tmp_path_check`,
`tool_help_check`, `conflict_triage`.

## Adding the forty-first

Before writing one, grep this directory for the question you are about to answer — that is what this page is for. If
the question is genuinely new, the two things the family requires are a `--self-test` with a **fixture that fails**, and
a `2` for every input it cannot interpret. A check that has never been shown to fail is decoration.
