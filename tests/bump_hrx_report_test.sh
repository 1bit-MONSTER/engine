#!/bin/bash
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
# bump_hrx_report_test.sh -- arms for scripts/bump-hrx-report.sh (engine#1945).
#
# The three cases a failure report must distinguish, because conflating any two of them is how a report stops
# being read: files in conflict, a failure that is not a conflict, and a report that could not read the log.
#
# usage: bump_hrx_report_test.sh <path-to-bump-hrx-report.sh>
set -uo pipefail

S=${1:?usage: bump_hrx_report_test.sh <path-to-bump-hrx-report.sh>}
[ -x "$S" ] || { echo "not executable: $S"; exit 1; }
fail=0
ok()  { echo "ok   $1"; }
bad() { echo "FAIL $1"; fail=1; }

# 1. A merge conflict: git's two shapes (content and modify/delete), one path repeated as a retry would.
got=$("$S" --run https://example/runs/1 --log - --step "Sync the fork and merge AMD's pin into our line" <<'LOG'
Auto-merging ggml/src/ggml-hrx/ggml-hrx.cpp
CONFLICT (content): Merge conflict in ggml/src/ggml-hrx/ggml-hrx.cpp
CONFLICT (modify/delete): drivers/fake.c deleted in HEAD and modified in abc123
CONFLICT (content): Merge conflict in ggml/src/ggml-hrx/ggml-hrx.cpp
LOG
)
[[ $got == *"ggml/src/ggml-hrx/ggml-hrx.cpp"* ]] && ok "a conflict names the file" || bad "a conflict names the file"
[[ $got == *"drivers/fake.c"* ]] && ok "the modify/delete form is caught too" || bad "the modify/delete form is caught too"
[[ $(printf '%s' "$got" | grep -c 'ggml-hrx/ggml-hrx.cpp') -ge 1 ]] || bad "path present"
n=$(printf '%s\n' "$got" | sed -n '/^```$/,/^```$/p' | grep -c 'ggml/src/ggml-hrx/ggml-hrx.cpp')
[ "$n" = 1 ] && ok "a repeated path is listed once" || bad "a repeated path is listed once (got $n)"
[[ $got == *"Failing step"*"Sync the fork"* ]] && ok "the failing step is in the body" || bad "the failing step is in the body"

# 2. A failure that is NOT a conflict must say so -- an empty list would read as "no problem found".
got=$("$S" --run https://example/runs/2 --log - --step "Merge AMD's hrx-system pin into our line" <<'LOG'
error: could not apply 1a2b3c4... some commit
fatal: Exiting because of an unresolved dispute
LOG
)
[[ $got == *"not a merge conflict"* ]] && ok "a non-conflict failure says so" || bad "a non-conflict failure says so"
[[ $got != *'```'* ]] && ok "and prints no empty file block" || bad "and prints no empty file block"

# 3. No log at all: the body must still render and must not claim there were no conflicts.
got=$("$S" --run https://example/runs/3 --step "Sync")
[[ $got == *"log was not available"* ]] && ok "a missing log is stated, not implied" || bad "a missing log is stated, not implied"
[[ $got != *"not a merge conflict"* ]] && ok "a missing log is not reported as a non-conflict" || bad "a missing log is not reported as a non-conflict"
[[ $got == *"https://example/runs/3"* ]] && ok "the run URL is always present" || bad "the run URL is always present"

# 4. Contract: --run is required, an unknown flag is a usage error, not a silent empty body.
"$S" > /dev/null 2>&1; [ $? = 2 ] && ok "a missing --run exits 2" || bad "a missing --run exits 2"
"$S" --run x --nope > /dev/null 2>&1; [ $? = 2 ] && ok "an unknown argument exits 2" || bad "an unknown argument exits 2"

[ $fail = 0 ] && echo "all bump-hrx-report arms passed"
exit $fail
