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
# engine#3138 review (cl-1077, whose #3137 the orchestrator closed as this PR's duplicate): the run path is a DIFFERENT
# branch from the `--list` path the self-test asserts, so pin the announcement itself and, above all, that the number of
# rows ANNOUNCED is the number the summary counts. A skip that is printed but not counted is the failure this script
# exists to prevent, and it is invisible to `--list` -- which is why the count below is compared against the lines rather
# than against a literal.
announced=$(printf '%s\n' "$out" | grep -cE '^skip +(tools/ci/duplicate_fixes_check|tools/ci/pr_branch_stale_check)\.py +--tree-only: this row reads the open-PR set')
noted=$(printf '%s\n' "$out" | sed -n 's/.*, \([0-9][0-9]*\) open-PR row(s) not run (--tree-only).*/\1/p')
check "each open-PR row is announced with the tree-only reason" '[ "$announced" -eq 2 ]'
check "every announced row is counted in the summary (printed == counted)" '[ -n "$noted" ] && [ "$announced" -eq "$noted" ]'
check "the open-PR rows are not claimed as ordinary skips" \
      '! grep -qE "^skip +(tools/ci/(duplicate_fixes_check|pr_branch_stale_check)\.py) +(nothing compared|SKIPPED)" <<<"$out"'
# engine#3142 review (cl-review's mutant): the arms above are all about the OPEN-PR rows, so a `continue` that swallows
# the open-PR row AND the next one leaves every one of them green -- it was measured: 7 green where the tree gives 8,
# announced == noted still true, rc 0. So pin the other direction too: every row `--list --tree-only` names must appear
# in the run with a verdict, which is a statement about the rows that SHOULD have run rather than the two that should not.
listed=$(scripts/pre-pr-checks.sh --list --tree-only | awk '{print $2}' | sort -u)
seen=$(printf '%s\n' "$out" | awk '/^(ok|FAIL|cannot-tell|skip) +tools\// {print $2}' | sort -u)
missing=$(comm -23 <(printf '%s\n' "$listed") <(printf '%s\n' "$seen") | tr '\n' ' ')
check "every row --list --tree-only names actually ran (a swallowed row is not a clean run)" '[ -z "$missing" ]'
# ...and the set comparison above is SELF-CONSISTENT with the code under test: a `continue` that swallows a row drops it
# from `--list` too, so `listed` and `seen` shrink together and the two arms stay green. Measured: that mutant (cl-review's,
# applied verbatim) made this file report `ok` while one check had vanished from the run. So the expected SIZE is a
# literal, taken from the repository the way the runner's own self-test takes 17/15: a dropped row cannot hide behind a
# second reading of the same broken function.
check "--list --tree-only still names 15 rows (a literal, not a second reading of this code)" \
      '[ "$(printf "%s\n" "$listed" | grep -c .)" -eq 15 ]'
extra=$(comm -13 <(printf '%s\n' "$listed") <(printf '%s\n' "$seen") | sort -u | tr '\n' ' ')
want_extra=$(printf '%s\n' $open_rows | sort -u | tr '\n' ' ')
check "...and the only names outside --list are the two open-PR rows it announced" '[ "$extra" = "$want_extra" ]'
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
