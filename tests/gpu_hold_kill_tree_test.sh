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
# gpu-hold ends the hold's whole descendant tree (engine#1620), and must still do it when the ROOT is the
# process that dies first (engine#1624). That is the common case: a ticket's payload is a `bash -c`, which
# exits on SIGTERM, and its children are then reparented - so a tree recomputed *after* the root is gone
# returns only the corpse, and a descendant that ignored TERM outlives its ticket. This test runs the
# script's own `descendants`/`kill_tree` (extracted with sed, not copied - a copy would not catch the real
# function regressing) against exactly that shape, and fails if anything survives.
#
# No GPU, no ticket, ~5 s: the processes here are `sleep`s under two nested `bash -c`s.
set -uo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
src=$repo/scripts/gpu-hold
fails=0
pass() { printf 'ok   %s\n' "$1"; }
fail() { printf 'FAIL %s\n' "$1"; fails=$((fails + 1)); }

[ -f "$src" ] || { fail "scripts/gpu-hold not found at $src"; printf '\ngpu_hold_kill_tree_test: %d failure(s)\n' "$fails"; exit 1; }

# the functions under test, taken from the script itself
extract() {
    sed -n '/^descendants()/,/^}/p;/^alive()/,/^}/p;/^kill_tree()/,/^}/p' "$src"
}
if ! bash -n <(extract); then
    fail "the extracted functions do not parse (has scripts/gpu-hold changed shape?)"
    printf '\ngpu_hold_kill_tree_test: %d failure(s)\n' "$fails"; exit 1
fi

child=$(mktemp)
cat > "$child" <<'CHILD'
#!/bin/bash
# a descendant that handles nothing and starts a TERM-ignoring grandchild: runner -> server
trap '' TERM
sleep 120 &
wait
CHILD
chmod +x "$child"

run_case() {   # -> prints the surviving pids of one hold's tree, empty when kill_tree did its job
    # the ROOT is a plain `bash -c`, so IT dies on TERM and the child is reparented - the #1624 shape
    bash -c "$child & wait" &
    local root=$! tree survivors p
    sleep 1
    tree=$(bash -c "eval \"\$(sed -n '/^descendants()/,/^}/p' $src)\"; descendants $root" | tr '\n' ' ')
    bash -c "eval \"\$(sed -n '/^descendants()/,/^}/p;/^alive()/,/^}/p;/^kill_tree()/,/^}/p' $src)\"; kill_tree $root 3" 2> /dev/null
    sleep 2
    survivors=
    for p in $tree; do kill -0 "$p" 2> /dev/null && survivors="$survivors $p"; done
    for p in $tree; do kill -9 "$p" 2> /dev/null; done
    printf '%s' "${survivors# }"
}

# three runs: a one-shot pass could be a race, and the bug was timing-shaped (the root had to die first)
for i in 1 2 3; do
    left=$(run_case)
    if [ -z "$left" ]; then pass "run $i: every process in the hold's tree is gone"
    else fail "run $i: killed the root and left $left alive - a TERM-ignoring descendant outlived its ticket"; fi
done

rm -f "$child"
printf '\ngpu_hold_kill_tree_test: %d failure(s)\n' "$fails"
[ "$fails" = 0 ]
