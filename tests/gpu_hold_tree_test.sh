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
echo "gpu_hold_tree: ok"
