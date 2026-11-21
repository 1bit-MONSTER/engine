#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# engine#3131: the two runner modes a CI row depends on are a CONTRACT, so they get arms rather than prose.
#
#   --tree-only     must drop EXACTLY the rows that read the open-PR queue. A row that reads the whole queue running in
#                   a gating job reds every PR for a condition no author can fix; that is #3131's own reason for the
#                   mode, and a renamed row is how it silently stops being safe (the list names rows, so a rename just
#                   runs the row).
#   --base-optional must turn an unresolvable --base into cannot-tell for the four rows that need one, and NOT into a
#                   pass: the summary has to count them, or a run with no base at all would read all-green.
#
# Three of these arms (the four names, the count, the summary line) are the ask on engine#3138's review -- the flag had
# measurements but no arm. The adapted shape comes from #3137's tests/pre_pr_checks_tree_only_test.sh, which could not be
# taken verbatim: it pins a DIFFERENT --tree-only (rows printed as `skip … tree-only: this row reads the open-PR set`),
# and main's merged one drops them and counts them ("2 open-PR row(s) not run (--tree-only)"), so two of its arms would
# fail here.
#
# Deliberately not covered: the resolvable-base control. `pre_pr_checks_tree` (the CI row) runs this script with
# `--base origin/main --base-optional --tree-only`, so a flag that broke the ordinary path would red that row; a third
# 20-second run here would only duplicate it.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 1
fail=0
check() { if eval "$2"; then printf 'ok   %s\n' "$1"; else printf 'FAIL %s\n' "$1"; fail=1; fi; }

bogus="origin/this-ref-does-not-exist"
base_rows="tools/ci/registration_loss_check.py tools/ci/registry_loss_check.py tools/ci/branch_stale_check.py tools/ci/guard_loss_check.py"
open_rows="tools/ci/duplicate_fixes_check.py tools/ci/pr_branch_stale_check.py"

# A checkout with no history is the case the flag exists for, and a ref that cannot resolve is the same event: neither
# the tools nor the runner can tell "you mistyped" from "this clone has one ref".
out=$(timeout 900 scripts/pre-pr-checks.sh --base "$bogus" --base-optional --tree-only 2>&1); rc=$?

check "the run completes (rc=0) with no base and --base-optional" '[ $rc -eq 0 ]'
check "the summary counts them as cannot-tell, not as a pass" 'grep -qE "pre-pr-checks: .*, 4 could not be told" <<<"$out"'
check "the summary says how many open-PR rows it did not run" \
      'grep -q "pre-pr-checks: .*2 open-PR row(s) not run (--tree-only)" <<<"$out"'
for row in $base_rows; do
    check "$row is reported cannot-tell, with its own line" \
          "grep -q '^cannot-tell  $row .*--base $bogus does not resolve' <<<\"\$out\""
done
for row in $open_rows; do
    check "$row does not run at all under --tree-only" '! grep -qE "^(ok|FAIL|cannot-tell) +$row" <<<"$out"'
done
for row in $base_rows $open_rows; do
    check "$row is still a row of the runner (a rename would silently readmit it)" \
          "grep -q '\"$row' scripts/pre-pr-checks.sh && [ -f '$row' ]"
done

# Without the flag the same invocation is a USAGE error, which is the behaviour a person relies on: a mistyped ref must
# not be reported as "could not look".
out2=$(timeout 900 scripts/pre-pr-checks.sh --base "$bogus" --tree-only 2>&1); rc2=$?
check "without --base-optional an unresolvable base is still rc 2" '[ $rc2 -eq 2 ]'
check "...and the message says how to run it deliberately" 'grep -q -- "--base-optional" <<<"$out2"'
# the --tree-only skip lines ARE printed before the base gate (they are announced first), so "nothing ran" means no row
# was executed and no summary was printed -- not that the output was empty.
check "...and no row ran, and no summary was printed" \
      '! grep -qE "^(ok|FAIL|cannot-tell) +tools/" <<<"$out2" && ! grep -qE "^pre-pr-checks: (all|[0-9]+ failed)" <<<"$out2"'

if [ $fail -eq 0 ]; then echo "pre_pr_checks_modes: ok"; fi
exit $fail
