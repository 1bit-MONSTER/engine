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

[ "$fails" = 0 ] && { echo "gpu_hold_survivors: ok"; exit 0; }
echo "gpu_hold_survivors: $fails failure(s)"; exit 1
