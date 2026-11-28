#!/bin/sh
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
# tests/gpu_step_reap_test.sh - scripts/gpu-step-reap reaps everything a step started. No device: a "server" is a
# sleeper that holds a file open, forked into the background by the step, orphaned when the step's shell exits.
set -u
here=$(cd "$(dirname "$0")/.." && pwd); R=$here/scripts/gpu-step-reap
t=$(mktemp -d); trap 'for p in $(pgrep -f "gpu-step-reap-sleeper-$$"); do pkill -KILL -P "$p"; kill -KILL "$p"; done 2> /dev/null; rm -rf "$t"' EXIT
bad=0; ok() { echo "ok   $*"; }; no() { echo "FAIL $*"; bad=$((bad + 1)); }
# Match the SLEEPER ITSELF ("/bin/sh <dir>/gpu-step-reap-sleeper-<pid>-<x> <secs>"), anchored: the helper's own command
# line also contains the marker (it is in the step it was given), and an unanchored `pgrep -f` saw "alive" before the
# helper had even set its trap - on a slow CI runner the TERM then hit the default handler (rc 143, no reap line).
alive() { pgrep -f "^/bin/sh $t/gpu-step-reap-sleeper-$$-$1 " > /dev/null 2>&1; }
for x in a b c d e; do printf '#!/bin/sh\nsleep "$1"\n:\n' > "$t/gpu-step-reap-sleeper-$$-$x"; chmod +x "$t/gpu-step-reap-sleeper-$$-$x"; done   # named sleepers pgrep can find (a shell + its sleep child)

# 1. the exit code passes through
HOLD_REAP_GRACE_S=1 sh "$R" 30 sh -c 'exit 7'; [ $? = 7 ] && ok "the step's exit code is returned" || no "exit code not returned"

# 2. a step that forks a long-running child and exits: the child is reaped, the step's rc kept
HOLD_REAP_GRACE_S=1 sh "$R" 30 sh -c "$t/gpu-step-reap-sleeper-$$-a 300 & exit 0" 2> "$t/e2"; rc=$?
sleep 0.5
! alive a && [ $rc = 0 ] && grep -q "left [0-9]* process" "$t/e2" && ok "an orphaned child of the step is reaped (and named)" \
    || no "orphan survived (rc $rc): $(cat "$t/e2")"

# 3. a grandchild that ignores TERM is KILLed after the grace
HOLD_REAP_GRACE_S=1 sh "$R" 30 sh -c "sh -c 'trap \"\" TERM; $t/gpu-step-reap-sleeper-$$-b 300' & exit 0" 2> /dev/null
sleep 0.5
! alive b && ok "a grandchild that ignores TERM is killed after the grace" || no "TERM-ignoring grandchild survived"

# 4. a step that times out: 124, and its child reaped too
HOLD_REAP_GRACE_S=1 sh "$R" 1 sh -c "$t/gpu-step-reap-sleeper-$$-c 300 & sleep 30" 2> /dev/null; rc=$?
sleep 0.5
[ $rc = 124 ] && ! alive c && ok "a timed-out step returns 124 and leaves nothing" || no "timeout: rc $rc, child alive: $(alive c && echo yes || echo no)"

# 5. a process OUTSIDE the step's session (another job of the same user) is not touched
$t/gpu-step-reap-sleeper-$$-d 300 > /dev/null 2>&1 < /dev/null &
HOLD_REAP_GRACE_S=1 sh "$R" 30 true
alive d && ok "a process outside the step's session is left alone" || no "an unrelated process was killed"
pkill -KILL -f "gpu-step-reap-sleeper-$$-d"; pkill -KILL -P "$!" 2> /dev/null; sleep 0.2   # the marker shell and its sleep child

# 6. the HELPER is signalled mid-step (a CI cancel): it reaps the step's session before exiting 143 (review cl-release).
#    `exec` so $! IS the helper (with `VAR=.. cmd 2>f &` some shells' $! is a wrapper subshell, and the TERM would kill
#    that instead), and a slow runner gets up to 20 s for the step's child to appear before the signal.
( exec env HOLD_REAP_GRACE_S=1 sh "$R" 60 sh -c "$t/gpu-step-reap-sleeper-$$-e 300 & sleep 60" 2> "$t/e6" ) & hp=$!
n=0; while ! alive e && [ $n -lt 200 ]; do sleep 0.1; n=$((n + 1)); done
if ! alive e; then no "signalled helper: setup - the step's child never started within 20 s"
else
    sleep 0.3   # the helper is in its wait loop with the trap set
    kill -TERM "$hp"; wait "$hp"; rc=$?
    n=0; while alive e && [ $n -lt 100 ]; do sleep 0.1; n=$((n + 1)); done   # the reap's grace is 1 s
    [ $rc = 143 ] && ! alive e && ! kill -0 "$hp" 2> /dev/null && grep -q "signalled; reaping" "$t/e6" \
        && ok "a signalled helper reaps the step's session, then exits 143" \
        || no "signalled helper: rc $rc, child alive: $(alive e && echo yes || echo no), helper alive: $(kill -0 "$hp" 2> /dev/null && echo yes || echo no), stderr: $(cat "$t/e6")"
fi

# 7 + 8. the /dev/kfd check counts only what the STEP started (cl-pragent, 2026-11-28: hw-gate run 46087950806's ops
#        step passed 3611/3611 and failed 125, because another lane's CPU-only llama-perplexity of the same account
#        opened /dev/kfd the second the step ended). A plain file stands in for the device node.
: > "$t/fake-kfd"
holder() { sh -c 'exec 3< "$1"; sleep 30' gpu-step-reap-holder "$t/fake-kfd"; }
# 7. a stranger of the same uid holding the "device" during the step: not the step's, rc unchanged
holder & sp=$!; sleep 0.3
GPU_STEP_REAP_KFD_PATH=$t/fake-kfd HOLD_REAP_GRACE_S=1 sh "$R" 30 true 2> "$t/e7"; rc=$?
[ $rc = 0 ] && ! grep -q "kfd" "$t/e7" && ok "a same-uid stranger holding /dev/kfd is not blamed on the step (rc 0)" || no "stranger: rc $rc $(cat "$t/e7")"
kill "$sp" 2> /dev/null; pkill -KILL -f "gpu-step-reap-holder" 2> /dev/null; sleep 0.2
# 8. a child the step started that left its session with setsid and still holds the "device": 125
GPU_STEP_REAP_KFD_PATH=$t/fake-kfd HOLD_REAP_GRACE_S=1 sh "$R" 30 sh -c "setsid sh -c 'exec 3< \"\$1\"; sleep 30' gpu-step-reap-holder '$t/fake-kfd' & sleep 0.5" 2> "$t/e8"; rc=$?
[ $rc = 125 ] && grep -q "the step started" "$t/e8" && ok "a setsid escapee the step started that holds /dev/kfd -> 125" || no "escapee: rc $rc $(cat "$t/e8")"
pkill -KILL -f "gpu-step-reap-holder" 2> /dev/null; sleep 0.2

echo "gpu_step_reap_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
