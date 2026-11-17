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

# engine#2803: the refusal must say WHICH KIND of fault it saw, and who the driver named. It used to say only "fault line(s)",
# and the PR whose hw_gate was refused (2801) spent its time doubting its own pin -- while the log's own "Process <name> pid
# <n>" line, two lines above the reset, named the other lane's suite. Measured on this box: two of three MODE2 resets in one
# night were that suite, and the driver writes the process name WITHOUT any reset keyword, so it is not in the matched set.
q=$scratch/tqclass; mkdir -p "$q"; rm -f "$scratch/TTOKENclass"; : > "$scratch/tklogclass"
printf '%s\n' "kernel: amdgpu 0000:c5:00.0:  Process hrx-ops-test pid 186654 thread hrx-ops-test pid 186654" \
              "kernel: amdgpu 0000:c5:00.0: MES failed to respond to msg=REMOVE_QUEUE" \
              "kernel: amdgpu 0000:c5:00.0: GPU reset(2) succeeded!" > "$scratch/tklogclass"
env GPU_HOLD_KLOG_FILE="$scratch/tklogclass" GPU_QUEUE_DIR="$q" GPU_TOKEN_FILE="$scratch/TTOKENclass" \
    GPU_HOLD_LOG="$scratch/tlogclass" GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=3 \
    GPU_HOLD_RESET_WINDOW_MIN=10 GPU_HOLD_FAULT_ABORT=0 GPU_HOLD_TCTL_START=100 \
    GPU_HOLD_TCTL_FILE="$scratch/tctl" GPU_HOLD_THERMAL_FLAG="$scratch/x" \
    timeout 60 "$hold" -t 1 -l takeclass -- true > /dev/null 2>&1
crc=$?
[ "$crc" = 78 ] || { echo "FAIL take-time class: rc $crc, want 78 (the refusal itself must not change)"; fail=1; }
grep -q "the newest is a GPU reset/hang" "$scratch/tlogclass" \
    || { echo "FAIL take-time class: the refusal does not name the class of the newest line"; fail=1; }
grep -q "the driver last named hrx-ops-test pid 186654" "$scratch/tlogclass" \
    || { echo "FAIL take-time class: the refusal does not name the process the driver named"; fail=1; }

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

# every term of GPU_FAULT_RE gets an arm (#2538 review): without one, a term could drop out of the pattern and every
# arm would stay green. These two were ridden only through other spellings.
take_run ring "kernel: amdgpu 0000:c5:00.0: [drm] *ERROR* ring gfx_0.0.0 timeout, signaled seq=1234, emitted seq=1236"
[ "$trc" = 78 ] || { echo "FAIL take-time: a ring timeout -> rc $trc, want 78"; fail=1; }

take_run hang "kernel: amdgpu 0000:c5:00.0: amdgpu: GPU hang detected on ring gfx"
[ "$trc" = 78 ] || { echo "FAIL take-time: an amdgpu hang -> rc $trc, want 78"; fail=1; }

# the refusal must say HOW OLD the newest line is: "in the last 10 min" cannot tell a wedged device from another lane's
# teardown nine minutes ago, and on 2026-11-15 every amdxdna fault that refused a ticket arrived 1.5-8.5 min AFTER the
# previous ticket released the NPU, with nothing running (engine#2542). A stub journalctl supplies the epoch.
mkdir -p "$scratch/agebin"
printf '#!/bin/bash\nprintf "1700000000 kernel: amdgpu 0000:c5:00.0: [drm] *ERROR* amdgpu: GPU reset begin!\\n"\n' > "$scratch/agebin/journalctl"
chmod +x "$scratch/agebin/journalctl"
q=$scratch/tqage; mkdir -p "$q"; rm -f "$scratch/age-token"
env PATH="$scratch/agebin:$PATH" HOME=$scratch ONEBIT_HOLD_DIR="$scratch/agehold" GPU_HOLD_COOLDOWN=0 \
    GPU_HOLD_WAIT=5 GPU_HOLD_MINUTES=1 GPU_HOLD_THERMAL_FLAG="$scratch/x" GPU_HOLD_RESET_WINDOW_MIN=10 \
    timeout 60 "$hold" -l takeage -- true > "$scratch/tlogage" 2>&1
age_rc=$?
[ "$age_rc" = 78 ] || { echo "FAIL take-time age: rc $age_rc, want 78"; fail=1; }
grep -qE 'newest [0-9]+s ago' "$scratch/tlogage" || { echo "FAIL take-time age: the refusal does not say how old the newest line is"; fail=1; }
grep -qE 'newest [0-9]{9,}s ago' "$scratch/tlogage" && { echo "FAIL take-time age: the age printed in seconds, not an epoch"; fail=1; }

# The age query must exclude the recovery notice too, not just the count. If it does not, a run whose only matching line
# is the notice reports the NOTICE's age while the count says zero real faults - a fresh timestamp attached to a stale
# event, which is worse than no age at all (review of #2550: the skip was added to the age on the strength of an
# argument, not a test). Two epochs: a real fault 600 s old, the notice 10 s old; the age must come from the fault.
mkdir -p "$scratch/agebin2"; t1=$(( $(date +%s) - 600 )); t2=$(( $(date +%s) - 10 ))
printf '#!/bin/bash\nprintf "%s kernel: amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_error_worker: fault at 0x0\\n" %s\nprintf "%s kernel: amdxdna 0000:c6:00.1: aie2_tdr_recover_all: TDR recovery throttled: last power-cycle 45560 ms ago; no context touched\\n" %s\n' "$t1" "$t1" "$t2" "$t2" > "$scratch/agebin2/journalctl"
chmod +x "$scratch/agebin2/journalctl"
rm -f "$scratch/age-token2"
env PATH="$scratch/agebin2:$PATH" HOME=$scratch ONEBIT_HOLD_DIR="$scratch/agehold2" GPU_HOLD_COOLDOWN=0 \
    GPU_HOLD_WAIT=5 GPU_HOLD_MINUTES=1 GPU_HOLD_THERMAL_FLAG="$scratch/x" GPU_HOLD_RESET_WINDOW_MIN=10 \
    timeout 60 "$hold" -l takeage2 -- true > "$scratch/tlogage2" 2>&1
age2_rc=$?
[ "$age2_rc" = 78 ] || { echo "FAIL take-time age/skip: rc $age2_rc, want 78"; fail=1; }
grep -q 'refusing: 1 ' "$scratch/tlogage2" || { echo "FAIL take-time age/skip: the notice was counted as a fault"; fail=1; }
_age2=$(sed -n 's/.*newest \([0-9]\+\)s ago.*/\1/p' "$scratch/tlogage2" | head -1)
if [ -z "$_age2" ]; then echo "FAIL take-time age/skip: the refusal has no age"; fail=1;
elif [ "$_age2" -lt 300 ]; then echo "FAIL take-time age/skip: age ${_age2}s is the RECOVERY NOTICE's, not the fault's (~600s)"; fail=1; fi

take_run clean clean
[ "$trc" = 0 ] || { echo "FAIL take-time: a clean log -> rc $trc, want 0 (a healthy hold must still run)"; fail=1; }


# arm 1: the fault fires
run 1 600 1 1
e1=$elapsed
if [ "$rc" != 78 ]; then echo "FAIL arm 1: rc $rc, want 78 (the fault abort)"; fail=1; fi
if [ "$elapsed" -ge 60 ]; then echo "FAIL arm 1: the hold took ${elapsed}s; the fault was at 5 s and the payload sleeps 600 s"; fail=1; fi
grep -q "ABORT GPU/NPU fault" "$scratch/log1" || { echo "FAIL arm 1: the log does not say the fault ended the hold"; fail=1; }
# engine#2494: the abort records what the ticket HELD before it kills it. After kill_tree the numbers are gone, which is
# why every post-mortem of this class has started by reconstructing them from `ps`. The payload here is a sleep chain
# with no KFD context, so the arm asserts the "none of ours" line -- the variant that also tells an operator whose fault
# it was -- and a dump that never runs cannot produce it.
grep -q "held: no process of this ticket has a KFD context" "$scratch/log1" || {
    echo "FAIL arm 1: the fault abort did not record what the ticket held on the device (engine#2494)"; fail=1; }
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
# ── arm 6: a ticket the max hold cut short says so in a way the NEXT lane can act on (engine#2542) ────────────────
# Measured on this box: the op-18 line and one IO_PAGE_FAULT burst fell 8m29s after a ticket that exited 124, with
# nothing running, and the boot's other two bursts fell 3m26s and 1m36s after other releases. "command exited 124"
# alone leaves the next lane's take-time refusal looking like the next lane's fault. The cut is REAL here -- `-t 0.05`
# is three seconds and the payload sleeps 600 -- so the arm exercises the same path a lane hits, not a faked rc.
q6=$scratch/q6; mkdir -p "$q6"; : > "$scratch/klog6"; rm -f "$scratch/deep.pid"
printf '%s\n' "kernel: usb 1-1: new device" > "$scratch/klog6"
env GPU_HOLD_FAULT_ABORT=1 GPU_HOLD_KLOG_FILE="$scratch/klog6" \
    GPU_QUEUE_DIR="$q6" GPU_TOKEN_FILE="$scratch/TOKEN6" GPU_HOLD_LOG="$scratch/log6" \
    GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=3 GPU_HOLD_RESET_WINDOW_MIN=0 \
    GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE="$scratch/tctl" GPU_HOLD_THERMAL_FLAG="$scratch/x" \
        timeout 60 "$hold" -t 0.05 -l faulttest6 -- "$scratch/wrapper.sh" "$scratch" 600 > /dev/null 2>&1
rc=$?
if [ "$rc" != 124 ]; then echo "FAIL arm 6: rc $rc for a payload cut by the max hold, want 124"; fail=1; fi
grep -q "the max hold cut it short" "$scratch/log6" || { echo "FAIL arm 6: a 124 release does not name the max hold"; fail=1; }
grep -q "may appear AFTER this release" "$scratch/log6" || { echo "FAIL arm 6: a 124 release does not warn about deferred work"; fail=1; }

if [ -n "$deep" ] && kill -0 "$deep" 2> /dev/null; then kill -9 "$deep"; fi

if [ -n "$deep" ] && kill -0 "$deep" 2> /dev/null; then kill -9 "$deep"; fi

# arm 5: an amdxdna fault DURING the hold. #2498 added this spelling at take time only; with one shared pattern the
# during-hold watch now reads it too (the NPU rule: on any amdxdna fault, stop).
# engine#2542/11-29Z: the driver's RECOVERY notices contain "power-cycle" and "aie2_tdr_recover_all", so they match
# `amdxdna.*power.?cycle` -- and one aborted a real hold. Measured 2026-11-15 (07:29:38 local, GPU-HOLD.log :134):
#   gpu-hold[402117] ::error::ABORT GPU/NPU fault ... (rc 78) after 493s
#   | amdxdna ... aie2_tdr_recover_all: TDR recovery throttled: last power-cycle 45560 ms ago (< 60000 ms),
#   |          retrying on the next watchdog tick; no context touched
# The line says NO CONTEXT TOUCHED: the fault it refers to was 45 s old, and the hold it ended belonged to another
# lane's job. A bookkeeping line must not abort a hold; the events it summarises must still abort.
run 6 600 1 1 "kernel: amdxdna 0000:c6:00.1: [drm] aie2_tdr_recover_all: TDR recovery throttled: last power-cycle 45560 ms ago (< 60000 ms), retrying on the next watchdog tick; no context touched"
if [ "$rc" = 78 ]; then echo "FAIL arm 6: a recovery-throttle notice (\"no context touched\") aborted the hold"; fail=1; fi
run 7 600 1 1 "kernel: amdxdna 0000:c6:00.1: [drm] aie2_tdr_recover_all: Array stopped dispatching, power-cycling NPU to recover"
if [ "$rc" != 78 ]; then echo "FAIL arm 7: a real power-cycle event did not abort (rc $rc)"; fail=1; fi

run 5 600 1 1 "kernel: amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_error_worker: fault at 0x00000000"
if [ "$rc" != 78 ]; then echo "FAIL arm 5: rc $rc after an amdxdna fault line, want 78"; fail=1; fi
grep -q "newest line is from the NPU" "$scratch/log5" || { echo "FAIL arm 5: the abort does not name the NPU as the faulting device"; fail=1; }
grep -q "newest line is from the GPU" "$scratch/log1" || { echo "FAIL arm 1: the abort does not name the GPU as the faulting device"; fail=1; }
if [ -n "$deep" ] && kill -0 "$deep" 2> /dev/null; then kill -9 "$deep"; fi

[ "$fail" = 0 ] && echo "gpu_hold_fault_abort: ok (fault at 5 s -> rc 78 after ${e1}s; clean -> 0; opt-out -> 0; default on; 124 named -> 78; take-time page fault/MES/ring/hang -> 78; NPU fault in hold -> 78; a recovery notice does NOT abort; a real power-cycling event does)"
exit "$fail"
