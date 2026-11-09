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
# gpu_hold_kill_tree.sh — the ticket's teardown must reach a descendant that ignores TERM (engine#1624).
#
# The fleet's orphans were all two or three levels deep (ticket → shell → runner → server) and the ones worth
# catching are exactly the ones that *do* handle TERM but slowly, or trap it while finishing work. #1620 made
# the teardown walk the descendant tree instead of one level; this checks the two ways that walk still missed:
#   * it recomputed the tree for the KILL pass, from a root that was already dead and whose children had been
#     reparented — so the KILL pass had nothing to kill;
#   * the payload dies on TERM, `wait` returns, and the exit path killed the watchdog *mid-escalation*, so on
#     the max-hold path the KILL never ran at all.
#
# usage: tests/gpu_hold_kill_tree.sh [path/to/gpu-hold]      (default: scripts/gpu-hold)
#
# Device-free: it uses a private queue directory and token file, so it never touches the fleet's GPU queue or
# the box's real token. ~10 s, no GPU work. Exit 0 on pass, 1 on a survivor.
set -u
hold=${1:-scripts/gpu-hold}
hold=$(readlink -f "$hold")
[ -x "$hold" ] || { echo "not executable: $hold" >&2; exit 2; }

tmp=$(mktemp -d "${TMPDIR:-/tmp}/gpu-hold-kill-tree.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
queue=$tmp/queue; mkdir -p "$queue"
token="marker-$RANDOM$RANDOM"

# the shape from #1624: the payload starts a child that TRAPS TERM and waits, so TERM kills the payload and
# the child is reparented while still running
cat > "$tmp/child.sh" <<'EOF'
#!/bin/sh
trap '' TERM
sleep 300
EOF
cat > "$tmp/payload.sh" <<'EOF'
#!/bin/bash
"$1" "$2" & wait
EOF
chmod +x "$tmp/child.sh" "$tmp/payload.sh"

GPU_QUEUE_DIR=$queue GPU_TOKEN_FILE=$queue/TOKEN GPU_HOLD_RELEASE_WAIT=0 \
    "$hold" -t 0.05 -l gpu-hold-kill-tree-test -- "$tmp/payload.sh" "$tmp/child.sh" "$token" \
    > "$tmp/hold.log" 2>&1
rc=$?
# the max hold must still report 124 (timeout-style), whatever the teardown did
[ "$rc" = 124 ] || { echo "FAIL: hold exited $rc, expected 124"; sed 's/^/  /' "$tmp/hold.log"; exit 1; }

survivors=""
while read -r pid args; do
    case "$args" in
        *"$token"*) survivors="$survivors $pid" ;;
    esac
done < <(ps -eo pid=,args=)
if [ -n "$survivors" ]; then
    echo "FAIL: a TERM-ignoring descendant outlived its ticket:$survivors (engine#1624)"
    for pid in $survivors; do kill -9 "$pid" 2>/dev/null; done
    sed 's/^/  /' "$tmp/hold.log"
    exit 1
fi
# and the ticket must be gone, not left holding the token
[ -f "$queue/TOKEN" ] && { echo "FAIL: token left behind at $queue/TOKEN"; exit 1; }
echo "PASS: a TERM-ignoring descendant does not outlive its ticket"
