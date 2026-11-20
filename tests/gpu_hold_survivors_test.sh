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
# gpu_hold_survivors_test.sh: the half of engine#1639's fix that CAN be tested on a CPU-only box.
#
# The defect is a payload root that survives SIGKILL — a process wedged in uninterruptible sleep, which user space cannot
# create and SIGKILL cannot interrupt. So the bound itself cannot be driven here, and a test that pretended to would be a
# test that cannot fail. What CAN be driven is the other half the fix adds: when a holder does release with a survivor, it
# records the survivor in $QUEUE/.gpu-hold-survivors, and the NEXT holder must warn about it and then clear the record —
# otherwise every later lane inherits a device that may still hold memory (engine#1639, the 51-GiB orphan class) with no
# indication at all.
#
# Two arms, both sandboxed (own queue, own token, fake Tctl; the real ~/GPU-QUEUE and ~/GPU-TOKEN are untouched):
#   1. a pre-seeded survivors record is warned about, verbatim, and emptied;
#   2. no record -> no warning (a warning nobody earned is noise, and noise is how a real one gets ignored).
#
# usage: tests/gpu_hold_survivors_test.sh [path-to-gpu-hold]     (default: the one in this tree)
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
hold=${1:-$here/../scripts/gpu-hold}
[ -x "$hold" ] || { echo "skip: no executable $hold" >&2; exit 77; }
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
# The take-time refusal reads the kernel log and refuses a ticket after a recent fault line (#2480/#2498). On a box
# whose NPU is faulting every few minutes (#2542) that refuses EVERY ticket, so this test would fail with rc 78 for a
# reason that is not the code -- measured 2026-11-16 16:4xZ. The fixture is the same override the fault-abort test
# uses: a log with no fault lines in it, so the refusal runs for real and finds nothing.
printf 'kernel: gpu-hold test fixture: no fault lines in this log\n' > $work/klog
export GPU_HOLD_KLOG_FILE=$work/klog
: > "$work/tctl"; printf '200\n' > "$work/tctl"     # fake Tctl: no thermal gate

fails=0
check() { if [ "$2" = "$3" ]; then printf '  ok   %s\n' "$1"; else printf '  FAIL %s (got %s, want %s)\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi; }

run_hold() {  # <dir> ; payload is /bin/true
    local dir=$1
    mkdir -p "$dir/q"
    GPU_QUEUE_DIR="$dir/q" GPU_TOKEN_FILE="$dir/token" GPU_HOLD_LOG="$dir/log" \
    GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=2 \
    GPU_HOLD_TCTL_START=999 GPU_HOLD_TCTL_FILE="$work/tctl" GPU_HOLD_THERMAL_FLAG="$dir/none" \
        "$hold" -t 1 -l survivors -- /bin/true > "$dir/out" 2>&1
}

# --- arm 1: a pre-seeded record is warned about and cleared -------------------------------------
a="$work/arm1"; mkdir -p "$a/q"
printf 'pid=999999 start=Thu Nov  9 10:00:00 2026 cmd=npu-rt wedged\n' > "$a/q/.gpu-hold-survivors"
run_hold "$a"; rc1=$?
check "arm1: the hold itself succeeds" "$rc1" "0"
warned=$(grep -c 'previous holder left' "$a/log" 2>/dev/null || true)
check "arm1: the next holder warns about the recorded survivor" "$warned" "1"
named=$(grep -c 'pid=999999' "$a/log" 2>/dev/null || true)
check "arm1: the warning names the recorded survivor" "$named" "1"
after=$(wc -c < "$a/q/.gpu-hold-survivors" 2>/dev/null || echo missing)
check "arm1: the record is cleared, so it cannot warn twice" "$after" "0"

# --- arm 2: no record -> no warning --------------------------------------------------------------
b="$work/arm2"; mkdir -p "$b/q"
run_hold "$b"; rc2=$?
check "arm2: the hold succeeds" "$rc2" "0"
noise=$(grep -c 'previous holder left' "$b/log" 2>/dev/null || true)
check "arm2: a holder with no record warns about nothing" "$noise" "0"

# engine#2917: the environment sweep reads /proc/<pid>/environ after a separate `-r` test, and a pid can pass `-r` and
# be reused by another user's process before the open. A granted ticket was lost one second into its hold that way, the
# command never ran, and the teardown line read clean. So an unreadable environ must be a COUNTED SKIP, a plain
# non-match must not be counted, and neither may print a shell error or change the sweep's exit status.
#
# Driven directly: the helper takes a PATH, so the three states need no /proc and no device. The unreadable case uses a
# DIRECTORY, which is the deterministic stand-in -- it passes `-r` and fails the read with EISDIR, the same shape as the
# EACCES the race produces (verified separately against /proc/1/environ, which is unreadable to this user).
src=$(mktemp); d2=$(mktemp -d)
sed -n '/^env_has_ticket()/,/^}/p' "$hold" > "$src"
printf 'A=1\0GPU_HOLD_TICKET=gpu-hold lane pid 42 2026-11-20T00:00:00-04:00\0B=2\0' > "$d2/good"
printf 'A=1\0B=2\0' > "$d2/none"
printf 'A=1\0GPU_HOLD_TICKET_SUFFIX=x\0B=2\0' > "$d2/prefixonly"
mkdir -p "$d2/environ"
eht() { bash -c 'set -u; source "$1"; env_has_ticket "$2" "$3"' _ "$src" "$1" "$2"; }
eht "$d2/good" "GPU_HOLD_TICKET=gpu-hold lane pid 42 2026-11-20T00:00:00-04:00"; check "arm3: the exact ticket line is found" "$?" "0"
eht "$d2/none" "GPU_HOLD_TICKET=x"; check "arm3: no such variable is a plain non-match (not unreadable)" "$?" "1"
eht "$d2/prefixonly" "GPU_HOLD_TICKET=x"; check "arm3: a longer variable with the same prefix is not a match" "$?" "1"
errs=$( { eht "$d2/environ" "GPU_HOLD_TICKET=x" 1; } 2>&1 )
check "arm3: an unreadable environ is its OWN state" "$?" "2"
check "arm3: and it prints nothing (no shell 'Permission denied' leak)" "${errs:-}" ""
rm -f "$src"; rm -rf "$d2"

[ "$fails" = 0 ] && { echo "gpu_hold_survivors: ok"; exit 0; }
echo "gpu_hold_survivors: $fails failure(s)"; exit 1
