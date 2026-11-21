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
# engine#3131: `--tree-only` must skip EXACTLY the rows that read the open-PR queue, and say so, because this is the
# mode a CI job gates on. Two failures matter and both are checked here: a row that reads the open-PR set running
# anyway (every PR would go red for a condition no author can fix), and a skip that is not counted (the summary would
# read all-green while a row did not run -- the shape this script exists to prevent).
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 1
fail=0
check() { if eval "$2"; then printf 'ok   %s\n' "$1"; else printf 'FAIL %s\n' "$1"; fail=1; fi; }

out=$(timeout 900 scripts/pre-pr-checks.sh --tree-only --base HEAD 2>&1); rc=$?

check "the run completes" '[ $rc -eq 0 ] || [ $rc -eq 1 ]'
check "duplicate_fixes_check is skipped by name" \
      'grep -q "^skip  tools/ci/duplicate_fixes_check.py .*tree-only: this row reads the open-PR set" <<<"$out"'
check "pr_branch_stale_check is skipped by name" \
      'grep -q "^skip  tools/ci/pr_branch_stale_check.py .*tree-only: this row reads the open-PR set" <<<"$out"'
check "neither reads the open-PR set in the run" \
      '! grep -qE "^(ok|FAIL) +tools/ci/(duplicate_fixes_check|pr_branch_stale_check)\.py" <<<"$out"'
check "the skips are counted, not dropped" \
      'grep -qE "pre-pr-checks: .*, [1-9][0-9]* skipped" <<<"$out"'
# A rename is the way this silently stops being safe: the skip list names rows, so a renamed row would just RUN.
for row in tools/ci/duplicate_fixes_check.py tools/ci/pr_branch_stale_check.py; do
    check "$row still exists and is a row" 'grep -q "\"$row" scripts/pre-pr-checks.sh && [ -f "$row" ]'
done

if [ $fail -eq 0 ]; then echo "pre_pr_checks_tree_only: ok"; fi
exit $fail
