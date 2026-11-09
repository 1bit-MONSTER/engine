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
# tests/gpu_hold_mem_abort_test.sh — a hold ends itself when the box runs out of memory, BEFORE the kernel picks
# a victim. The failure this prevents is not the run's own death: on 2026-11-08/09 a GPU run took the box to
# 0.7 GiB available and the OOM killer chose the PR-agent LLM, which took the fleet's own brains down.
#
# Usage: gpu_hold_mem_abort_test.sh [path-to-gpu-hold]
#
# The meminfo is FAKE (GPU_HOLD_MEMINFO): the test needs the transition "plenty, then starved", which cannot be
# produced on demand on a shared box. Two assertions, and both matter:
#   * the hold ends WELL BEFORE its own bound — the payload sleeps 600 s and the test allows 120 s, so a script
#     with no memory gate cannot pass by finishing;
#   * the log says why, so a hold that ends for another reason cannot be mistaken for the gate firing.
set -u
hold=${1:-./scripts/gpu-hold}
[ -x "$hold" ] || { echo "not executable: $hold"; exit 2; }

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
q=$scratch/queue; mkdir -p "$q"

awk '{if ($1=="MemAvailable:") print "MemAvailable:   900000000 kB"; else print}' /proc/meminfo > "$scratch/mem-high"
awk '{if ($1=="MemAvailable:") print "MemAvailable:      1048576 kB"; else print}' /proc/meminfo > "$scratch/mem-low"

cat > "$scratch/wrapper.sh" <<'WRAP'
#!/bin/bash
( trap '' TERM; exec -a mem-abort-child sleep 600 ) &
echo $! > "$1/deep.pid"
sleep 600
WRAP
chmod +x "$scratch/wrapper.sh"

[ "$(grep -c '^MemAvailable:' "$scratch/mem-high")" = 1 ] || { echo "FAIL the high fixture has $(grep -c '^MemAvailable:' "$scratch/mem-high") MemAvailable lines, not 1"; exit 1; }
cp "$scratch/mem-high" "$scratch/meminfo"
( sleep 5; cp "$scratch/mem-low" "$scratch/meminfo" ) &

t0=$(date +%s)
GPU_HOLD_MEMINFO="$scratch/meminfo" GPU_HOLD_MEM_ABORT_GIB=10 \
GPU_QUEUE_DIR="$q" GPU_TOKEN_FILE="$scratch/TOKEN" GPU_HOLD_LOG="$scratch/log" \
GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=3 \
GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE="$scratch/tctl" GPU_HOLD_THERMAL_FLAG="$scratch/x" \
    timeout 120 "$hold" -t 10 -l memtest -- "$scratch/wrapper.sh" "$scratch" >/dev/null 2>&1
rc=$?
elapsed=$(( $(date +%s) - t0 ))
deep=$(cat "$scratch/deep.pid" 2> /dev/null)

if [ "$rc" = 124 ]; then
    echo "FAIL the hold ran to the 120 s bound (${elapsed}s): no memory abort"
    [ -n "$deep" ] && kill -9 "$deep" 2> /dev/null
    exit 1
fi
if grep -q "MemAvailable" "$scratch/log" 2> /dev/null && grep -q "ending the hold" "$scratch/log" 2> /dev/null; then
    :
else
    echo "FAIL the hold ended (rc=$rc, ${elapsed}s) without a memory abort in the log:"
    sed 's/^/     /' "$scratch/log" 2> /dev/null | tail -4
    [ -n "$deep" ] && kill -9 "$deep" 2> /dev/null
    exit 1
fi
if [ -n "$deep" ] && kill -0 "$deep" 2> /dev/null; then
    echo "FAIL the child ($deep) survived the memory abort (rc=$rc, ${elapsed}s)"
    kill -9 "$deep" 2> /dev/null
    exit 1
fi
echo "ok   the hold ended on the memory floor after ${elapsed}s (rc=$rc) and the child is gone"

# engine#1652 review: pin the OPT-IN property, which is the one a future edit is most likely to break by giving the
# variable a default. Same starved meminfo, same payload, floor unset -- the hold must run to its own max hold, with
# no abort in the log and the payload alive until then.
cp "$scratch/mem-high" "$scratch/meminfo"
( sleep 5; cp "$scratch/mem-low" "$scratch/meminfo" ) &
t0=$(date +%s)
GPU_HOLD_MEMINFO="$scratch/meminfo" \
GPU_QUEUE_DIR="$q" GPU_TOKEN_FILE="$scratch/TOKEN2" GPU_HOLD_LOG="$scratch/log2" \
GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=3 \
GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE="$scratch/tctl2" GPU_HOLD_THERMAL_FLAG="$scratch/x2" \
    timeout 120 "$hold" -t 0.1 -l off -- "$scratch/wrapper.sh" "$scratch" >/dev/null 2>&1
rc=$?
elapsed=$(( $(date +%s) - t0 ))
deep=$(cat "$scratch/deep.pid" 2> /dev/null)
[ -n "$deep" ] && kill -9 "$deep" 2> /dev/null
if [ "$rc" != 124 ]; then
    echo "FAIL the floor is not opt-in: the hold ended at ${elapsed}s (rc=$rc) with GPU_HOLD_MEM_ABORT_GIB unset"
    exit 1
fi
if grep -q "ending the hold" "$scratch/log2" 2> /dev/null; then
    echo "FAIL the floor is not opt-in: an abort was logged with GPU_HOLD_MEM_ABORT_GIB unset"
    exit 1
fi
echo "ok   the floor is opt-in: with it unset the hold ran to its own max hold (${elapsed}s) and nothing aborted"
echo "gpu_hold_mem_abort: ok"
