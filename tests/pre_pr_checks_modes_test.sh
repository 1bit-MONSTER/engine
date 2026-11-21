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
# 4 is the four `--base` rows; 5 is those plus `loom_pin_check`, whose answer is a fact about the MACHINE rather than about this
# runner (it is cannot-tell when no local clone carries the Loom pin, green when one does -- and it says which). Accepting both
# keeps this arm's teeth on the four rows it is actually about.
check "the summary counts them as cannot-tell, not as a pass" 'grep -qE "pre-pr-checks: .*, (4|5) could not be told" <<<"$out"'
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
check "--list --tree-only still names 16 rows (a literal, not a second reading of this code)" \
      '[ "$(printf "%s\n" "$listed" | grep -c .)" -eq 16 ]'
# ...and the TOTAL, which is the arm #3145 arrived at independently 24 seconds after this branch (crediting it here
# because its reasoning is the better statement of the same property, and it catches one thing mine does not): a filter
# that drops a row it never announced changes this number and nothing else; a row that prints TWO lines changes it too.
# `--list` WITHOUT `--tree-only` is the correct baseline precisely because it never enters the filter under test -- that
# is what makes the expectation independent rather than a second reading of the same broken function.
listed_total=$(bash scripts/pre-pr-checks.sh --list | wc -l)
check "every listed row produces exactly one line (no filter swallowed or doubled a row)" \
      '[ "$(printf "%s\n" "$out" | grep -cE "^(ok|FAIL|skip|cannot-tell) ")" -eq "$listed_total" ]'
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

# engine#3150 review: the gpu-hold --strict-warnings decision is computed FROM $base, so `--fetch-base` has to run
# before it or the one run that made the base available still decides that arm against a ref that did not exist yet --
# measured in a faithful shallow checkout: the OFF line printed *before* the fetch and the arm stayed off on exactly the
# PRs that touch `scripts/gpu-hold`. A move, so the arm is about ORDER, and it is a source-order check on purpose: the
# behavioural version needs a clone of this repository with a `main` branch on its remote, which a unit test cannot have
# without the network. (A local `--depth 1` clone of a worktree does NOT substitute: measured, its remote has no `main`
# branch, so `--unshallow` can never produce `origin/main` and the arm would pass for the wrong reason.)
fetch_line=$(grep -n 'fetch-base: %s does not resolve in this shallow clone' scripts/pre-pr-checks.sh | head -1 | cut -d: -f1)
touches_line=$(grep -n '^if ! touches=' scripts/pre-pr-checks.sh | head -1 | cut -d: -f1)
check "--fetch-base is placed before the strict-warnings decision (order is the fix)" \
      '[ -n "$fetch_line" ] && [ -n "$touches_line" ] && [ "$fetch_line" -lt "$touches_line" ]'
# ...and with a base that resolves, the flag changes nothing: the control for the control.
check "--fetch-base on a full clone with a real base prints no fetch line" \
      '! grep -q "fetch-base:" <<<"$(bash scripts/pre-pr-checks.sh --base origin/main --base-optional --fetch-base --list --tree-only 2>&1)"'

if [ $fail -eq 0 ]; then echo "pre_pr_checks_modes: ok"; fi
exit $fail
