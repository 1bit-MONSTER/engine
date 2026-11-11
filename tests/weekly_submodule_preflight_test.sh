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
# weekly_submodule_preflight_test.sh: the release preflight BLOCKS on a build tree that is not the pinned tree and only
# WARNS when there is no checkout to compare against (engine#2129). Both halves are the real code: the function is cut out
# of scripts/weekly.sh and run against stub checkers, so no submodules, no network and no build.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fails=0
ok()  { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1"; fails=$((fails + 1)); }

sed -n '/^preflight_submodules() {$/,/^}$/p' "$root/scripts/weekly.sh" > "$tmp/fn.sh"
grep -q 'check-submodules.sh exit' "$tmp/fn.sh" || { bad "could not cut preflight_submodules() out of scripts/weekly.sh"; echo "weekly_submodule_preflight: FAIL"; exit 1; }
grep -q 'preflight_submodules "' "$root/scripts/weekly.sh" || bad "scripts/weekly.sh's preflight no longer calls preflight_submodules"
say() { echo "$*"; }
# shellcheck disable=SC1091
. "$tmp/fn.sh"

# A stub checker: prints what the real one prints and exits with the code under test.
stub() {  # <exit code> <line...>
    local rc=$1; shift
    printf '#!/usr/bin/env bash\nprintf "%%s\\n" %s\nexit %s\n' "$(printf '%q ' "$@")" "$rc" > "$tmp/stub-$rc"
    chmod +x "$tmp/stub-$rc"; echo "$tmp/stub-$rc"
}
row() {  # <name> <want return> <want text> <rc> [lines...]
    local name=$1 want=$2 text=$3 rc=$4; shift 4
    local check; check=$(stub "$rc" "$@")
    local out; out=$(preflight_submodules "$check" "$tmp" 2>&1); local got=$?
    if [ "$got" = "$want" ] && printf '%s\n' "$out" | grep -q -- "$text"; then ok "preflight: $name -> return $got"
    else bad "preflight: $name: want return $want with '$text', got $got: $out"; fi
}

row "a tree at its pins passes" 0 "preflight: " 0 "ok   third_party/sub 1234567890" "check-submodules: every checkout is its pin and clean"
row "a checkout that disagrees stops the run" 1 "not the pinned tree" 1 "::error::check-submodules: third_party/x is not the pin: tree a, checkout b"
row "an unchecked-out clone only warns" 0 "WARNING, not blocking" 3 "::warning::check-submodules: third_party/x is not checked out"
row "no .gitmodules only warns" 0 "WARNING, not blocking" 2 "check-submodules: no .gitmodules under . -- nothing to inspect, which is not a pass"
row "a missing checker does not stop the run" 0 "" 0

# The stated asymmetry, in one assertion: the two "cannot tell" codes must not return 1.
for rc in 2 3; do
    check=$(stub "$rc" "::warning::check-submodules: cannot tell")
    preflight_submodules "$check" "$tmp" >/dev/null 2>&1
    [ $? = 0 ] && ok "exit $rc is not a block" || bad "exit $rc should not block the release"
done

if [ "$fails" != 0 ]; then echo "weekly_submodule_preflight: $fails failure(s)"; exit 1; fi
echo "weekly_submodule_preflight: ok"
