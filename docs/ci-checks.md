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

**A directory of single-purpose checkers lives in `tools/ci/` — 44 of them on 2026-11-20 — and nothing listed them.** A
lane that has just been burned by a class of mistake — a lost commit, a removed diagnostic, a route nobody registered —
has no way to discover that a check for it already exists, and writes one more. This page is the index. It names the
*questions*, not the files, so it does not rot when one is added.

## Run them

The count is deliberately a dated observation, not a maintained number: `ls tools/ci/*.py | wc -l` is the authority, and
the loop below enumerates the real set on every run. A literal here went stale within the hour it merged.

```sh
for f in tools/ci/*.py; do printf '%s: ' "$(basename "$f")"; python3 "$f" --self-test >/dev/null 2>&1 && echo ok || echo FAIL; done
```

Every one of them carries `--self-test` with hermetic fixtures, and `tools/ci/tool_selftest_wired_check.py` fails when a
tool that has a self-test is referenced nowhere — so a check whose arms stopped running says so instead of looking fine.

## Exit codes are the family's contract

| code | meaning |
|---|---|
| 0 | checked, nothing found |
| 1 | the thing the check exists for was found, named |
| 2 | **cannot tell** — the input was absent, unreadable, or in an unexpected shape |

A `2` is never a pass. It exists because the recurring bug in this tree is a check reporting `0` about a population it
never examined: a fault census counting lines instead of events, a loss check counting refusals while diagnostics went
missing, a skip that a baseline made acceptable. Note that `SKIP_RETURN_CODE 77` does **not** convert a `2` into a skip
(`77` is a skip); a ctest row that wants "cannot tell → skip" needs an explicit mapping.

**A row that did not run is neither a pass nor a failure**, and the exit code alone does not always say so. The fork's
HRX test runner (`tests/testing_suite.h`) answers `77` for a device case, or a whole run of them, that could not see a
device — but a run whose host cases ran and whose device cases did not still exits `0`, and the hand-written `test-hrx-*`
rows exit `1`. What all of them share is one line on their output, `test skipped: no device available`, which is also
the fork's `FAIL_REGULAR_EXPRESSION` (engine#2393). So `ctest_known_fail` reads ctest's per-row outputs
(`--outputs <build>/Testing/Temporary/LastTest.log`) and counts those rows in a column of their own: they are not
known failures, not new failures and not "now passing", and the row `fork_ctest_known_fail` **fails** on them, because
its subject is the hardware. `ONEBIT_FORK_CTEST_NO_DEVICE=skip` is the explicit skip (`77`) for a box that is allowed to
have no device. Do not add `SKIP_RETURN_CODE 77` to a row that relies on that regex: on ctest 4.2.3 the skip code wins,
and the red row becomes a green one (engine#2912).

## The questions, by family

**Is the thing we shipped the thing we reviewed?** `merged_tail_check` (a commit pushed after its PR merged),
`pr_carry_check`, `pr_verdict`, `pr_review_queue`, `branch_stale_check`, `moving_ref_check`, `duplicate_fixes_check`. The
rule they serve: *a PR merges at its reviewed head; a commit pushed after the merge is silently lost.*

**Does a guard still guard?** `guard_loss_check` (a removed refusal **or diagnostic**), `doc_section_loss_check`,
`conflict_marker_check`, `dupdef_check`, `registration_loss_check`, `registry_loss_check`, `tests_referenced_check`,
`file_create_check`, `tool_selftest_wired_check`. These compare what *was* present against what is, and refuse to call a
shrunken population clean.

**Is the census measuring the right population?** `census_counters_check`, `stale_expectations_check`, `fork_hrx_rows`,
`fa_split_expectation_check`, `ctest_skips`, `ctest_known_fail`, `hip_f16_audit`, `hrx_f16_narrowing`,
`serve_field_census`. The distinctions they encode: a *mention* is not a *read*, a *line* is not an *event*, and a
*baseline entry* is not a *skip that is fine*.

**Are routes, fields and launches consistent?** `serve_routes_check`, `warmup_route_check`, `server_launch_check`,
`npu_decl_check`, `ci_targets_check`, `cmake_configure_check`. Warmed paths must be registered routes; fields a request
sends must be fields something reads; a test that starts a server must take the port lock, and the port census must see
the port whether it is a `freeport` call, a `--port` literal or a shell default.

**Pins, files and containers.** `config_pin_check`, `submodule_pin_check`, `notice_covers_deps`, `hrx_build_scope`,
`kernel_id_selftest`, `token_rate_floor`, `medians`, `lane_syntax_check`, `tmp_path_check`, `tool_help_check`,
`conflict_triage`.

## Adding the next one

Grep this directory for the question you are about to answer — that is what this page is for. If the question is
genuinely new, the family asks for three things: a `--self-test` whose fixture **fails** before the fix, a `2` for every
input it cannot interpret, and a line in the message saying **what it inspected and what it could not**. A check that has
never been shown to fail is decoration; a check that cannot say "I could not look" is a green light nobody earned.
