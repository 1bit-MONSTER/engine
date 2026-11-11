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
# weekly_submodule_preflight_test.sh: the release preflight BLOCKS when a submodule is not the commit the tree pins and
# only WARNS when there is no checkout to compare against (engine#2129). The function is cut out of scripts/weekly.sh and
# run against stub checkers, so no submodules, no network and no build.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fail=0
ok()  { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1"; fail=1; }

sed -n '/^preflight_submodules() {$/,/^}$/p' "$root/scripts/weekly.sh" > "$tmp/fn.sh"
grep -q 'submodule_pin_check.py exit' "$tmp/fn.sh" || { bad "could not cut preflight_submodules() out of scripts/weekly.sh"; echo "weekly_submodule_preflight: FAIL"; exit 1; }
grep -q 'preflight_submodules "' "$root/scripts/weekly.sh" || bad "scripts/weekly.sh's preflight no longer calls preflight_submodules"
say() { echo "$*"; }
# shellcheck disable=SC1091
. "$tmp/fn.sh"

# A stub standing in for tools/ci/submodule_pin_check.py. It is a Python file because the function runs the real tool as
# `python3 <path>`: printing what the tool prints and exiting with the code under test is the whole stub.
stub() {  # <exit code> <line>
    {
        printf 'import sys\nprint('
        printf '%s' "$2" | sed 's/.*/"&"/'
        printf ')\nsys.exit(%s)\n' "$1"
    } > "$tmp/stub.py"
    echo "$tmp/stub.py"
}
row() {  # <name> <want return> <want text> <rc> <line>
    local name=$1 want=$2 text=$3 rc=$4 line=$5
    local check out got=0
    check=$(stub "$rc" "$line")
    out=$(preflight_submodules "$check" "$tmp" 2>&1) || got=$?
    if [ "$got" = "$want" ] && printf '%s\n' "$out" | grep -q -- "$text"; then
        ok "preflight: $name -> return $got"
    else
        bad "preflight: $name: want return $want with '$text', got $got: $out"
    fi
}

# A passing run prints nothing at all: the tool's summary line ("N submodule(s) at their pinned commit") is the one
# thing the function drops, because a preflight that always prints a line trains its reader to skip the lines.
check=$(stub 0 "submodule_pin: 7 submodule(s) at their pinned commit")
out=$(preflight_submodules "$check" "$tmp" 2>&1)
if [ -z "$out" ]; then ok "a tree whose submodules are their pins passes silently"; else bad "a passing run should print nothing, got: $out"; fi
row "a submodule at a different commit BLOCKS the run" 1 "not at the commit this tree pins" 1 "submodule_pin: third_party/x is checked out at abc, which is NOT the commit this tree pins"
row "no checkout to compare only WARNS" 0 "WARNING, not blocking" 2 "submodule_pin: no submodules reported - cannot tell, and that is not a pass"

# A dirty submodule at its pin is reported and does NOT block: a deliberate local patch is a real thing a box does.
row "a dirty submodule at its pin is reported, not blocking" 0 "changed path" 0 "submodule_pin: third_party/x is at the pinned abc but its work tree has 301 changed path(s)"

# A checker that is not there at all (an older tree): the row is skipped, not failed.
preflight_submodules "$tmp/no-such-check.py" "$tmp" >/dev/null 2>&1 && ok "a missing checker does not stop the run" \
    || bad "a missing checker should not stop the run"

# The asymmetry, in one assertion: exit 2 must not block, exit 1 must.
check=$(stub 2 "submodule_pin: cannot tell")
if preflight_submodules "$check" "$tmp" >/dev/null 2>&1; then ok "exit 2 is not a block"; else bad "exit 2 should not block the release"; fi

if [ "$fail" != 0 ]; then echo "weekly_submodule_preflight: FAIL"; exit 1; fi
echo "weekly_submodule_preflight: ok"
