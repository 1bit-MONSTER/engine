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
# tests/gpu_hold_tree_test.sh — a hold that ends must end everything it started.
#
# Why: three orphans in two days were all GRANDCHILDREN of a ticket — a shell or python wrapper starts the
# real server one level down, the hold's `pkill -P` reached only the wrapper, and the server outlived the
# ticket (a 14 h llama-server holding 51 GiB of GTT; an NPU forward-serve 70 minutes past a 6-minute ticket).
# This drives the real scripts/gpu-hold with an unlabelled private queue, a wrapper that spawns a grandchild,
# and asserts the grandchild is gone when the hold expires.
#
# Exit: 0 = the tree died with the ticket, 1 = something survived, 77 = cannot run here (no setsid/ps).
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
hold=$here/scripts/gpu-hold
[ -x "$hold" ] || { echo "skip: no $hold"; exit 77; }
command -v setsid > /dev/null 2>&1 || { echo "skip: no setsid"; exit 77; }

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
q=$scratch/queue; mkdir -p "$q"

# THREE levels: hold → wrapper → mid → the work. Every orphan had at least this depth (a ticket's shell
# starts a runner, the runner starts the server; thermal-run's scope starts ninja, ninja starts the
# compiler), and `pkill -P` on the wrapper reaches only `mid` — the survivor is one level further down.
cat > "$scratch/mid.sh" <<'MID'
#!/bin/bash
( exec -a gpu-hold-grandchild sleep 300 ) &
echo $! > "$1/deep.pid"
sleep 300
MID
cat > "$scratch/wrapper.sh" <<'WRAP'
#!/bin/bash
bash "$1/mid.sh" "$1" &
echo $! > "$1/mid.pid"
wait
WRAP
chmod +x "$scratch/mid.sh" "$scratch/wrapper.sh"

GPU_QUEUE_DIR=$q GPU_TOKEN_FILE=$scratch/TOKEN GPU_HOLD_LOG=$scratch/log \
GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=2 \
GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE=$scratch/tctl GPU_HOLD_THERMAL_FLAG=$scratch/x \
    "$hold" -t 0.05 -l selftest-tree -- "$scratch/wrapper.sh" "$scratch" >/dev/null 2>&1
rc=$?

deep=$(cat "$scratch/deep.pid" 2>/dev/null)
if [ -z "$deep" ]; then
    echo "FAIL the wrapper never wrote deep.pid (rc=$rc)"; exit 1
fi
if kill -0 "$deep" 2> /dev/null; then
    echo "FAIL the grandchild ($deep) survived the hold — this is the orphan shape (rc=$rc)"
    kill -9 "$deep" 2> /dev/null
    exit 1
fi
echo "ok   the grandchild was killed with the hold (hold rc=$rc)"

# The case above is the one `pkill -P` failed, and a `sleep` dies on TERM, so it only ever exercises the TERM pass. The
# processes that became 14-hour orphans are the ones that IGNORE TERM, and the trap is that their parent exits on TERM:
# once reparented they are no longer reachable from the ticket, so a KILL pass that re-walks the tree cannot find them.
# This arm is that shape - a TERM-ignoring grandchild whose mid-level parent exits during the grace period.
cat > "$scratch/mid_stubborn.sh" <<'MID'
#!/bin/bash
trap '' TERM                       # the wrapper ignores TERM, like a server that installs its own handler
( trap '' TERM; exec -a gpu-hold-survivor sleep 300 ) &
echo $! > "$1/stubborn.pid"
sleep 300
MID
cat > "$scratch/wrapper_quitter.sh" <<'WRAP'
#!/bin/bash
bash "$1/mid_stubborn.sh" "$1" &
echo $! > "$1/mid.pid"
sleep 300
WRAP
chmod +x "$scratch/mid_stubborn.sh" "$scratch/wrapper_quitter.sh"
rm -f "$scratch/stubborn.pid"

GPU_QUEUE_DIR=$q GPU_TOKEN_FILE=$scratch/TOKEN GPU_HOLD_LOG=$scratch/log \
GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=3 \
GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE=$scratch/tctl GPU_HOLD_THERMAL_FLAG=$scratch/x \
    "$hold" -t 0.05 -l selftest-tree-stubborn -- "$scratch/wrapper_quitter.sh" "$scratch" >/dev/null 2>&1

stubborn=$(cat "$scratch/stubborn.pid" 2>/dev/null)
if [ -z "$stubborn" ]; then
    echo "FAIL the stubborn wrapper never wrote stubborn.pid"; exit 1
fi
if kill -0 "$stubborn" 2> /dev/null; then
    echo "FAIL the TERM-ignoring grandchild ($stubborn) survived the hold - this is the 14-hour-orphan shape"
    kill -9 "$stubborn" 2> /dev/null
    exit 1
fi
echo "ok   the TERM-ignoring grandchild was killed too"


# A SIGNAL to the hold itself. Until now TERM ran only `exit 143`: EXIT then released the token and removed the ticket
# while the payload kept the device, so the next lane was admitted on top of it — the 11-08 orphan through a different
# door (`timeout`, an operator kill, an orchestrator stopping a lane). This arm TERMs the hold mid-hold and asserts both
# halves that matter: the payload's TREE is gone, and the token is released. It is the same shape as the two arms above
# (a grandchild two levels down) so that a fix which only TERMs the wrapper still fails here.
rm -f "$scratch/deep.pid" "$scratch/TOKEN"
mkdir -p "$scratch/sig"
GPU_QUEUE_DIR=$q GPU_TOKEN_FILE=$scratch/TOKEN GPU_HOLD_LOG=$scratch/log \
GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=2 \
GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE=$scratch/tctl GPU_HOLD_THERMAL_FLAG=$scratch/x \
    "$hold" -t 5 -l selftest-signal -- "$scratch/mid.sh" "$scratch" >/dev/null 2>&1 &
holdpid=$!
for i in $(seq 1 200); do
    [ -s "$scratch/deep.pid" ] && [ -f "$scratch/TOKEN" ] && break
    kill -0 "$holdpid" 2> /dev/null || break
    sleep 0.1
done
sigdeep=$(cat "$scratch/deep.pid" 2>/dev/null)
if [ -z "$sigdeep" ] || [ ! -f "$scratch/TOKEN" ]; then
    echo "FAIL the signal arm never got to a held state (deep=${sigdeep:-none} token=$([ -f "$scratch/TOKEN" ] && echo held || echo absent))"
    kill -9 "$holdpid" 2> /dev/null
    exit 1
fi
kill -TERM "$holdpid" 2> /dev/null
wait "$holdpid"
rc=$?
if kill -0 "$sigdeep" 2> /dev/null; then
    echo "FAIL a signal to the hold released the token but left the payload running ($sigdeep) - the #1640 door"
    kill -9 "$sigdeep" 2> /dev/null
    exit 1
fi
echo "ok   a signal to the hold ended the payload's tree (hold rc=$rc)"
if [ -f "$scratch/TOKEN" ]; then
    echo "FAIL the token outlived the hold after a signal"
    exit 1
fi
echo "ok   and released the token"

# The two arms #1642 does not have (engine#1640): a signal while the hold is still QUEUED, and the ordering the guard
# depends on. The queued arm is the other side of `[ -n "$child" ]`; the ordering arm exists because #1642's `declare -F
# kill_tree` guard silently skips the kill in the window between `child=$!` and the helper's definition.
cat > "$scratch/mid_signal2.sh" <<'MID'
#!/bin/bash
( exec -a gpu-hold-signal2-child sleep 300 ) &
echo $! > "$1/signal2.pid"
sleep 300
MID
chmod +x "$scratch/mid_signal2.sh"
rm -f "$scratch/signal2.pid" "$scratch/TOKEN"

GPU_QUEUE_DIR=$q GPU_TOKEN_FILE=$scratch/TOKEN GPU_HOLD_LOG=$scratch/log \
GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=2 \
GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE=$scratch/tctl GPU_HOLD_THERMAL_FLAG=$scratch/x \
    "$hold" -t 5 -l selftest-signal2 -- "$scratch/mid_signal2.sh" "$scratch" >/dev/null 2>&1 &
hold2=$!
for _ in $(seq 100); do [ -s "$scratch/signal2.pid" ] && break; sleep 0.1; done
sig2=$(cat "$scratch/signal2.pid" 2>/dev/null)
if [ -z "$sig2" ]; then echo "FAIL the second signalling payload never started"; kill -9 "$hold2" 2>/dev/null; exit 1; fi
kill -TERM "$hold2" 2> /dev/null
wait "$hold2" 2> /dev/null
if kill -0 "$sig2" 2> /dev/null; then
    echo "FAIL the payload ($sig2) survived a TERM to the hold - it holds the device with the token free"
    kill -9 "$sig2" 2> /dev/null; exit 1
fi
if [ -f "$scratch/TOKEN" ]; then echo "FAIL the token outlived a TERM to the hold"; rm -f "$scratch/TOKEN"; exit 1; fi
echo "ok   a TERM to the hold ends the payload and frees the token"

# queued: no payload exists, so the queue entry must go and the token it never held must be left alone
printf 'someone-else %s\n' "$$" > "$scratch/TOKEN"
GPU_QUEUE_DIR=$q GPU_TOKEN_FILE=$scratch/TOKEN GPU_HOLD_LOG=$scratch/log \
GPU_HOLD_POLL=1 GPU_HOLD_WAIT=30 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=2 \
GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE=$scratch/tctl GPU_HOLD_THERMAL_FLAG=$scratch/x \
    "$hold" -t 5 -l selftest-queued -- sleep 300 >/dev/null 2>&1 &
queued=$!
for _ in $(seq 100); do [ -n "$(ls "$q" 2>/dev/null | grep -v '^\.' || true)" ] && break; sleep 0.1; done
kill -TERM "$queued" 2> /dev/null
wait "$queued" 2> /dev/null
left=$(ls "$q" 2>/dev/null | grep -v '^\.' || true)
if [ -n "$left" ]; then echo "FAIL a queued hold left its ticket behind after a TERM: $left"; rm -f "$q"/*; exit 1; fi
if [ ! -f "$scratch/TOKEN" ]; then echo "FAIL a queued hold removed someone else's token"; exit 1; fi
rm -f "$scratch/TOKEN"
echo "ok   a TERM to a queued hold leaves the queue, not the token"

# ordering: the guard only works if the helper exists by the time the trap can run
ktl=$(grep -n '^kill_tree()' "$hold" | head -1 | cut -d: -f1)
lnch=$(grep -n '^child=\$!' "$hold" | head -1 | cut -d: -f1)
if [ -z "$ktl" ] || [ -z "$lnch" ]; then echo "FAIL cannot find kill_tree() or child=! in $hold"; exit 1; fi
if [ "$ktl" -gt "$lnch" ]; then
    echo "FAIL kill_tree is defined at line $ktl, after the payload launch at $lnch - a TERM in that window releases the"
    echo "     token and leaves the payload running, which is what #1642's declare -F guard then skips"
    exit 1
fi
echo "ok   kill_tree is defined before the payload starts ($ktl < $lnch)"

echo "gpu_hold_tree: ok"
