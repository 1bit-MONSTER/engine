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
# tests/git_shallow_instruments_test.sh — the shallow-clone guard, exercised by RUNNING the tools (engine#3106).
#
# `git_shallow.py --self-test` pins the decision table; this pins the WIRING, in the environment the guards exist
# for: a real `git clone --depth 1` whose distance reads 1 at the graft and whose merge base is absent, which is
# what a default `actions/checkout` gives every job.
#
# The fixture carries the files these tools parse (registry JSON, docs/*.md, and a tool source with a guard line in
# it), because a thin fixture is how an audit certifies safety it never tested: with no registry to parse,
# `registration_loss_check` bails at "nothing parsed at origin/main" and exits 2 for a reason that has nothing to do
# with shallowness. So every arm below asserts **rc 2 AND that the tool names the shallow clone** — a tool that
# cannot-tell for a different reason fails here on purpose.
#
# usage: bash tests/git_shallow_instruments_test.sh    (exit 0 = every arm passes, 1 = a failure, 77 = not applicable)
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# The fixture's data: real files, copied, not invented.
mkdir -p "$work/R/registry" "$work/R/docs" "$work/R/tools/ci"
cp "$root"/registry/*.json "$work/R/registry/" 2>/dev/null
cp "$root"/registry/*.tsv  "$work/R/registry/" 2>/dev/null
[ -f "$root/docs/npu.md" ] && cp "$root/docs/npu.md" "$work/R/docs/"
cp "$root/tools/ci/guard_loss_check.py" "$work/R/tools/ci/" 2>/dev/null

have=0
for f in registry docs tools; do [ -n "$(ls -A "$work/R/$f" 2>/dev/null)" ] && have=$((have + 1)); done
[ "$have" -eq 3 ] || { echo "skip: the tree does not carry the files these tools parse"; exit 77; }

cd "$work/R" || exit 2
git init -q -b main .
git config user.email t@e
git config user.name t
git add -A
git commit -qm base

# A branch that deletes a doc section and a guard-shaped line: the loss these tools exist to catch.
git checkout -q -b feat
python3 - <<'PY'
import pathlib
p = pathlib.Path("docs/npu.md")
if p.exists():
    lines = p.read_text().splitlines()
    idx = [i for i, l in enumerate(lines) if l.startswith("## ")]
    if len(idx) > 2:
        del lines[idx[1]:idx[2]]
        p.write_text("\n".join(lines) + "\n")
g = pathlib.Path("tools/ci/guard_loss_check.py")
if g.exists():
    lines = g.read_text().splitlines()
    drop = [i for i, l in enumerate(lines) if l.strip().startswith("if ") and "return" in l]
    if drop:
        del lines[drop[0]]
        g.write_text("\n".join(lines) + "\n")
PY
git commit -qam "feat removes a doc section and a guard line"
git checkout -q main
printf 'g\n' > g.txt
git add -A
git commit -qm "main advances by one"

cd "$work" || exit 2
git clone -q --depth 1 --branch feat "file://$work/R" sh || { echo "FAIL: shallow clone failed"; exit 1; }
cd sh || exit 2
git fetch -q --depth 1 origin main:refs/remotes/origin/main 2>/dev/null || git fetch -q --depth 1 origin main

fails=0
pass() { echo "ok   $*"; }
fail() { echo "FAIL $*"; fails=$((fails + 1)); }

# The premise, asserted before any verdict is drawn from it: if this tree is not shallow, every arm below is
# testing nothing.
if [ "$(git rev-parse --is-shallow-repository)" = "true" ]; then pass "the fixture is a shallow clone"
else fail "the fixture is not shallow -- every arm below would be vacuous"; fi
mb=$(git merge-base refs/remotes/origin/main HEAD 2>/dev/null)
if [ -z "$mb" ]; then pass "the fixture has no merge base (the state these guards detect)"
else fail "the fixture has a merge base ($mb): it is not the shape actions/checkout produces"; fi

# arm <name> <expected: "shallow" | "loud"> <tool> [args...]
arm() {
    local name=$1 want=$2 tool=$3; shift 3
    local out rc
    out=$(timeout 180 python3 "$root/tools/ci/$tool" "$@" 2>&1)
    rc=$?
    if [ "$rc" != 2 ]; then
        fail "$name: rc=$rc, want 2 (cannot-tell) -- $(tail -1 <<<"$out")"
    elif [ "$want" = shallow ] && ! grep -qi 'shallow' <<<"$out"; then
        # The tool answered cannot-tell for a reason that is not shallowness -- either the fixture lost its data or
        # the guard is not wired. Both are failures here, and the second is the one this file exists for.
        fail "$name: rc 2 but the reason is not the shallow clone -- $(tail -1 <<<"$out")"
    else
        pass "$name: rc=2$( [ "$want" = shallow ] && echo ', names the shallow clone')"
    fi
}

arm "guard_loss_check refuses to judge a shallow clone"        shallow guard_loss_check.py      --check --base refs/remotes/origin/main
arm "doc_section_loss_check refuses to judge a shallow clone"  shallow doc_section_loss_check.py --base refs/remotes/origin/main
arm "registration_loss_check refuses to judge a shallow clone" shallow registration_loss_check.py --check --base refs/remotes/origin/main

# pr_carry_check has no guard of its own: it is loud because `merge-base` fails, not because it asks. Pinned as
# cannot-tell so the day it starts trusting a number instead, this arm says so.
arm "pr_carry_check cannot tell either (loud, not guarded)" loud pr_carry_check.py . HEAD --base refs/remotes/origin/main

if [ "$fails" -eq 0 ]; then echo "git_shallow_instruments_test: pass"; exit 0; fi
echo "git_shallow_instruments_test: $fails failure(s)"
exit 1
