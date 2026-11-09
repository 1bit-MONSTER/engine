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

# tests/serve_kill_tree_test.sh — ending a serve ends the process that holds the memory.
#
# `1bit serve` is a wrapper: the model, and the GTT it holds, live in the `llama-server` it spawns as a child. A teardown
# that kills only the pid the shell started reaps the wrapper and leaves the child — the orphan shape from 11-08, one
# level down, produced by a test's own failure path (engine #1653 review, and engine #1624 for the same conclusion in
# scripts/gpu-hold).
#
# This drives `serve_kill_tree` from tests/serve_port.sh against a synthetic wrapper, so it needs no model and no device.
# The child ignores TERM, which is the case the KILL pass exists for and the one a `pkill -P` teardown misses.
#
# Exit: 0 = the tree ended, 1 = a descendant survived, 77 = cannot run here.
set -u

HELPER=$(dirname "$0")/serve_port.sh
[ -r "$HELPER" ] || { echo "skip: no $HELPER"; exit 77; }
# shellcheck source=tests/serve_port.sh
. "$HELPER"
command -v serve_kill_tree > /dev/null || { echo "skip: serve_kill_tree is not in the helper"; exit 77; }

tmp=$(mktemp -d)
pids=$tmp/pids
cleanup() {
    for p in $(cat "$pids" 2> /dev/null); do kill -9 "$p" 2> /dev/null; done
    rm -rf "$tmp"
}
trap cleanup EXIT

cat > "$tmp/inner.sh" << 'INNER'
#!/bin/sh
trap '' TERM                 # the process the KILL pass exists for
echo $$ >> "$PIDS"
for _ in $(seq 1 300); do sleep 1; done
INNER
cat > "$tmp/wrapper.sh" << 'WRAPPER'
#!/bin/sh
PIDS="$PIDS" "$1" &          # stands in for `1bit serve` spawning llama-server
wait
WRAPPER
chmod +x "$tmp/inner.sh" "$tmp/wrapper.sh"
: > "$pids"

PIDS="$pids" "$tmp/wrapper.sh" "$tmp/inner.sh" &
root=$!
sleep 1

if [ ! -s "$pids" ]; then
    echo "FAIL the wrapper never started its child (inconclusive, not a pass)"
    exit 1
fi
child=$(tail -n 1 "$pids")
kill -0 "$root" 2> /dev/null || { echo "FAIL the wrapper is not running"; exit 1; }
kill -0 "$child" 2> /dev/null || { echo "FAIL the child is not running"; exit 1; }

serve_kill_tree "$root" 2

# A SIGKILLed process is a zombie until it is reaped, and `kill -0` is true for a zombie — so poll for it to disappear
# rather than checking once, or the test reports a survivor that is already dead and holds nothing. The property that
# matters is "not running", not "not signalled".
gone() {  # <pid> -> true once the pid is gone or already reaped
    local i=0
    while [ "$i" -lt 20 ]; do
        kill -0 "$1" 2> /dev/null || return 0
        # a zombie is dead for our purposes: it holds no memory and cannot be killed further
        [ "$(sed -n 's/.*) \([A-Z]\).*/\1/p' "/proc/$1/stat" 2> /dev/null)" = Z ] && return 0
        sleep 0.1
        i=$((i + 1))
    done
    return 1
}

fails=0
if ! gone "$root"; then echo "FAIL the wrapper ($root) survived"; fails=$((fails + 1)); fi
if ! gone "$child"; then
    echo "FAIL the descendant ($child) survived — this is the process that would keep the GTT"
    fails=$((fails + 1))
fi
[ "$fails" = 0 ] && echo "ok   the whole tree ended, including a descendant that ignores TERM"
exit "$fails"
