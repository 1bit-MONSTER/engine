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
#
# tests/pre_pr_checks_ci_test.sh — the CI wrapper's three decisions, each driven through the state it decides on
# (engine#3131). Hermetic: a scratch git repo, git only, ~1 s. The point of the arms is the one that matters in CI —
# a shallow checkout must ask for history rather than silently degrade to eight enforced rows — and the one that must
# NOT happen — a fetch failure turning the row red for an environment the PR did not choose.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
wrapper=$root/tests/pre_pr_checks_ci.sh
[ -x "$wrapper" ] || { echo "FAIL wrapper missing: $wrapper"; exit 1; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
pass=0; fail=0
check() { if eval "$2"; then echo "ok   $1"; pass=$((pass + 1)); else echo "FAIL $1"; fail=$((fail + 1)); fi; }

# A source repo with history, and a shallow clone of it: the shape `actions/checkout` produces.
mkdir -p "$work/src" && cd "$work/src" || exit 2
git init -q -b main . && git config user.email t@e && git config user.name t
for i in 1 2 3; do echo "$i" > "f$i"; git add -A; git commit -qm "c$i"; done
cd "$work" || exit 2
git clone -q --depth 1 "file://$work/src" shallow 2>/dev/null
cd "$work/shallow" || exit 2
git config user.email t@e && git config user.name t
check "the fixture really is shallow" '[ "$(git rev-parse --is-shallow-repository)" = true ]'

# 1. shallow + a reachable remote: history is fetched, so the wrapper asks for the full run (no --base-optional).
out=$(bash "$wrapper" --plan --runner /bin/true 2>&1)
check "a shallow checkout is unshallowed before the checks run" 'grep -q "would run: git fetch --unshallow" <<<"$out"'
check "...and the planned argv has no --base-optional (the base rows will run)" '! grep -q -- "--base-optional" <<<"$(grep "^argv:" <<<"$out")"'

# 2. shallow + NO remote: the fetch cannot happen, so the wrapper degrades with a reason instead of going red.
git remote remove origin
out=$(bash "$wrapper" --plan --runner /bin/true 2>&1)
check "an un-unshallowable checkout falls back to --base-optional" 'grep -q -- "--base-optional" <<<"$(grep "^argv:" <<<"$out")"'
check "...and says why, so the row's cannot-tells are explained where they appear" 'grep -q "does not resolve" <<<"$out"'

# 3. a complete clone WITH its remote ref: nothing is fetched and nothing is degraded.
git -C "$work/src" update-ref refs/remotes/origin/main "$(git -C "$work/src" rev-parse main)"
cd "$work/src" || exit 2
out=$(bash "$wrapper" --plan --runner /bin/true 2>&1)
check "a full clone plans the plain run" '[ "$(grep "^argv:" <<<"$out")" = "argv: --base origin/main --tree-only" ]'
check "...and prints no unshallow line" '! grep -q "unshallow" <<<"$out"'

echo "pre_pr_checks_ci: $pass ok, $fail failed"
[ "$fail" -eq 0 ]
