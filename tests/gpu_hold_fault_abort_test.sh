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
# tests/gpu_hold_fault_abort_test.sh — a hold ends itself when the kernel log shows a GPU fault DURING the hold
# (CRASH-HUNT 2026-11-15). The 11-13 crash was nine MES REMOVE_QUEUE -> MODE2 resets inside two tickets, each after a
# gfxhub page fault from the ticket's own job; #2480 checks for resets only when the token is taken, so a storm inside
# one ticket ran on until the GPU wedged.
#
# Usage: gpu_hold_fault_abort_test.sh [path-to-gpu-hold]
#
# The kernel log is a FILE (GPU_HOLD_KLOG_FILE): the test needs "clean, then a fault", which cannot be produced on
# demand. Three arms:
#   1 a page fault appears 5 s in: the hold ends WELL before its payload would (600 s), with rc 78, the log says why,
#     and the job's deepest TERM-ignoring descendant is gone;
#   2 a clean log: the payload runs to completion, rc 0 (the watcher does not fire on nothing);
#   3 the same fault with GPU_HOLD_FAULT_ABORT=0: the payload runs to completion, rc 0 (the opt-out works);
#   4 the same fault with GPU_HOLD_FAULT_ABORT UNSET: rc 78, so the DEFAULT is on (arms 1-3 set it explicitly; a
#     mutant that flipped the default to off passed them - review on #2486).
set -u
hold=${1:-./scripts/gpu-hold}
[ -x "$hold" ] || { echo "not executable: $hold"; exit 2; }

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0

cat > "$scratch/wrapper.sh" <<'WRAP'
#!/bin/bash
( trap '' TERM; exec -a fault-abort-child sleep "$2" ) &
echo $! > "$1/deep.pid"
sleep "$2"
WRAP
chmod +x "$scratch/wrapper.sh"

run() {   # $1 arm  $2 payload seconds  $3 inject fault (1/0)  $4 GPU_HOLD_FAULT_ABORT, or "unset" to leave it out
          # $5 the fault line to inject (default: the 11-13 gfxhub page fault)
    local q=$scratch/q$1; mkdir -p "$q"; : > "$scratch/klog$1"; rm -f "$scratch/deep.pid"
    printf '%s\n' "kernel: usb 1-1: new device" > "$scratch/klog$1"
    local line=${5:-"kernel: amdgpu 0000:c5:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:9 pasid:26332)"}
    if [ "$3" = 1 ]; then
        ( sleep 5; printf '%s\n' "$line" >> "$scratch/klog$1" ) &
    fi
    local t0; t0=$(date +%s)
    local abort_env=(GPU_HOLD_FAULT_ABORT="$4"); [ "$4" = unset ] && abort_env=()
    env -u GPU_HOLD_FAULT_ABORT "${abort_env[@]}" GPU_HOLD_KLOG_FILE="$scratch/klog$1" \
    GPU_QUEUE_DIR="$q" GPU_TOKEN_FILE="$scratch/TOKEN$1" GPU_HOLD_LOG="$scratch/log$1" \
    GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=3 GPU_HOLD_RESET_WINDOW_MIN=0 \
    GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE="$scratch/tctl" GPU_HOLD_THERMAL_FLAG="$scratch/x" \
        timeout 120 "$hold" -t 10 -l faulttest$1 -- "$scratch/wrapper.sh" "$scratch" "$2" > /dev/null 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - t0 ))
    deep=$(cat "$scratch/deep.pid" 2> /dev/null)
}

echo 40000 > "$scratch/tctl"   # 40 C, in millidegrees

# ── the TAKE-TIME guard (engine#2480), which had no automated arm until now ───────────────────────────────────────
# It shipped with manual arms only, and the NPU spellings were missing from its regex although `npu-hold` execs this
# script: an NPU ticket's only look at the log before work starts is this check, so `amdxdna` faults have to count.
# Each arm runs the real hold against a klog file with a 10-minute take-time window and the during-hold watch OFF, so
# what is being tested is the refusal and nothing else.
take_run() {   # $1 arm name  $2 the kernel-log line (or "clean")
    local q=$scratch/tq$1; mkdir -p "$q"; rm -f "$scratch/TTOKEN$1"
    : > "$scratch/tklog$1"
    printf '%s\n' "kernel: amdgpu 0000:c5:00.0: [drm] Initialized" > "$scratch/tklog$1"
    [ "$2" = clean ] || printf '%s\n' "$2" >> "$scratch/tklog$1"
    env GPU_HOLD_KLOG_FILE="$scratch/tklog$1" GPU_QUEUE_DIR="$q" GPU_TOKEN_FILE="$scratch/TTOKEN$1" \
        GPU_HOLD_LOG="$scratch/tlog$1" GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=3 \
        GPU_HOLD_RESET_WINDOW_MIN=10 GPU_HOLD_FAULT_ABORT=0 GPU_HOLD_TCTL_START=100 \
        GPU_HOLD_TCTL_FILE="$scratch/tctl" GPU_HOLD_THERMAL_FLAG="$scratch/x" \
        timeout 60 "$hold" -t 1 -l taketest$1 -- true > /dev/null 2>&1
    trc=$?
    [ -e "$scratch/TTOKEN$1" ] && ttok=held || ttok=released
}

take_run amdgpu "kernel: amdgpu 0000:c5:00.0: [drm] *ERROR* amdgpu: GPU reset begin!"
[ "$trc" = 78 ] || { echo "FAIL take-time: an amdgpu reset -> rc $trc, want 78"; fail=1; }
[ "$ttok" = released ] || { echo "FAIL take-time: the token was still held after a refusal"; fail=1; }

# the NPU spelling: this is the line the rules say to stop for, and npu-hold has no other take-time check
take_run npu "kernel: amdxdna 0000:c6:00.1: [drm] *ERROR* command timeout: opcode 0x2"
[ "$trc" = 78 ] || { echo "FAIL take-time: an amdxdna command timeout -> rc $trc, want 78 (the NPU rule)"; fail=1; }
[ "$ttok" = released ] || { echo "FAIL take-time: the token was still held after an NPU refusal"; fail=1; }

take_run powercycle "kernel: amdxdna 0000:c6:00.1: [drm] power-cycle"
[ "$trc" = 78 ] || { echo "FAIL take-time: an amdxdna power-cycle -> rc $trc, want 78"; fail=1; }

# engine#2494 (cl-infra 11-15): the take-time pattern was a separate copy that lacked these two, so the ticket after a
# fault-ended hold was GRANTED on the very log that had just ended it; only the MODE2 line further down the chain refused.
take_run gfxhub "kernel: amdgpu 0000:c5:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:9 pasid:26332)"
[ "$trc" = 78 ] || { echo "FAIL take-time: a gfxhub page fault -> rc $trc, want 78 (engine#2494)"; fail=1; }
[ "$ttok" = released ] || { echo "FAIL take-time: the token was still held after a page-fault refusal"; fail=1; }

take_run mesremove "kernel: amdgpu 0000:c5:00.0: amdgpu: MES failed to respond to msg=REMOVE_QUEUE"
[ "$trc" = 78 ] || { echo "FAIL take-time: an MES REMOVE_QUEUE timeout -> rc $trc, want 78 (engine#2494)"; fail=1; }

take_run clean clean
[ "$trc" = 0 ] || { echo "FAIL take-time: a clean log -> rc $trc, want 0 (a healthy hold must still run)"; fail=1; }


# arm 1: the fault fires
run 1 600 1 1
e1=$elapsed
if [ "$rc" != 78 ]; then echo "FAIL arm 1: rc $rc, want 78 (the fault abort)"; fail=1; fi
if [ "$elapsed" -ge 60 ]; then echo "FAIL arm 1: the hold took ${elapsed}s; the fault was at 5 s and the payload sleeps 600 s"; fail=1; fi
grep -q "ABORT GPU/NPU fault" "$scratch/log1" || { echo "FAIL arm 1: the log does not say the fault ended the hold"; fail=1; }
grep -q "gfxhub" "$scratch/log1" || { echo "FAIL arm 1: the log does not quote the fault line"; fail=1; }
if [ -n "$deep" ] && kill -0 "$deep" 2> /dev/null; then echo "FAIL arm 1: the TERM-ignoring grandchild $deep survived"; kill -9 "$deep"; fail=1; fi
[ -e "$scratch/TOKEN1" ] && { echo "FAIL arm 1: the token is still held"; fail=1; }

# arm 2: a clean log
run 2 12 0 1
if [ "$rc" != 0 ]; then echo "FAIL arm 2: rc $rc with a clean kernel log, want 0"; fail=1; fi
grep -q "ABORT GPU/NPU fault" "$scratch/log2" && { echo "FAIL arm 2: the watcher fired on a clean log"; fail=1; }

# arm 3: the opt-out
run 3 12 1 0
if [ "$rc" != 0 ]; then echo "FAIL arm 3: rc $rc with GPU_HOLD_FAULT_ABORT=0, want 0"; fail=1; fi
grep -q "ABORT GPU/NPU fault" "$scratch/log3" && { echo "FAIL arm 3: the watcher fired with GPU_HOLD_FAULT_ABORT=0"; fail=1; }

# arm 4: the default (variable unset) must be ON
run 4 600 1 unset
if [ "$rc" != 78 ]; then echo "FAIL arm 4: rc $rc with GPU_HOLD_FAULT_ABORT unset, want 78 (the default must be on)"; fail=1; fi
if [ -n "$deep" ] && kill -0 "$deep" 2> /dev/null; then kill -9 "$deep"; fi

# arm 5: an amdxdna fault DURING the hold. #2498 added this spelling at take time only; with one shared pattern the
# during-hold watch now reads it too (the NPU rule: on any amdxdna fault, stop).
run 5 600 1 1 "kernel: amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_error_worker: fault at 0x00000000"
if [ "$rc" != 78 ]; then echo "FAIL arm 5: rc $rc after an amdxdna fault line, want 78"; fail=1; fi
if [ -n "$deep" ] && kill -0 "$deep" 2> /dev/null; then kill -9 "$deep"; fi

[ "$fail" = 0 ] && echo "gpu_hold_fault_abort: ok (fault at 5 s -> rc 78 after ${e1}s; clean -> 0; opt-out -> 0; default on -> 78; take-time page fault/MES -> 78; NPU fault in hold -> 78)"
exit "$fail"
