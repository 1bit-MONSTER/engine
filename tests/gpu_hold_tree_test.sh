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

# A THIRD shape, and it is the one nobody's ticket timeout goes through: a signal to `gpu-hold` ITSELF. The trap was
# `trap 'exit 143' TERM INT HUP`, so TERM ran release() - token and ticket removed - and left the payload holding the device:
# the next lane is admitted while the old work still has it. Measured on main 2026-11-09 by pi-verify as
# "hold alive=no, payload alive=yes, token=released", and by this lane the same way before the fix in engine#1640.
# This arm is that measurement: TERM the hold, then assert the payload is gone AND the token was released.
rm -f "$scratch/sig.pid" "$scratch/TOKEN"
cat > "$scratch/sig_payload.sh" <<'SIG'
#!/bin/bash
echo $$ > "$1/sig.pid"
exec -a gpu-hold-signal-payload sleep 300
SIG
chmod +x "$scratch/sig_payload.sh"
GPU_QUEUE_DIR=$q GPU_TOKEN_FILE=$scratch/TOKEN GPU_HOLD_LOG=$scratch/log \
GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=3 \
GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE=$scratch/tctl GPU_HOLD_THERMAL_FLAG=$scratch/x \
    "$hold" -t 1 -l selftest-signal -- "$scratch/sig_payload.sh" "$scratch" >/dev/null 2>&1 &
sig_hold=$!
# wait for it to HOLD (the token names its pid) rather than for a sleep: a queued hold has no payload yet
for _ in $(seq 1 30); do
    grep -q "pid $sig_hold" "$scratch/TOKEN" 2> /dev/null && break
    sleep 0.5
done
sig_payload=$(cat "$scratch/sig.pid" 2> /dev/null)
if [ -z "$sig_payload" ]; then
    kill -9 "$sig_hold" 2> /dev/null
    echo "FAIL the payload never started for the signal arm"; exit 1
fi
kill -TERM "$sig_hold" 2> /dev/null
for _ in $(seq 1 40); do
    kill -0 "$sig_payload" 2> /dev/null || break
    sleep 0.5
done
sig_rc=0
if kill -0 "$sig_payload" 2> /dev/null; then
    echo "FAIL the payload ($sig_payload) survived a TERM to gpu-hold - the token is free and the device is not (engine#1640)"
    kill -9 "$sig_payload" 2> /dev/null
    sig_rc=1
else
    echo "ok   a TERM to gpu-hold ends the payload too"
fi
# The order inside the hold is tree FIRST, token second: releasing the token while the payload still holds the device is the
# defect, so this arm waits for the release instead of assuming it lands in the same instant the payload dies.
for _ in $(seq 1 40); do
    [ -e "$scratch/TOKEN" ] || break
    sleep 0.25
done
if [ -e "$scratch/TOKEN" ]; then
    echo "FAIL the token was not released within 10 s of the signal"; sig_rc=1
else
    echo "ok   ... and the token is released (tree first, token second)"
fi
[ "$sig_rc" = 0 ] || exit 1

echo "gpu_hold_tree: ok"
