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
# gpu_hold_device_scope_test.sh -- engine#2961: one device's fault must not refuse the other device's ticket.
#
# The defect: gpu-hold reads ONE fault regex over every device on the box, so an `amdxdna` (NPU) fault refused two GPU
# tickets for ten minutes on 2026-11-20 -- the PR-agent LLM's own ticket and an orchestrator-granted GPU window -- and
# neither of them was touching the NPU. The ticket declares its device by the queue it is queued on (npu-hold execs
# gpu-hold with $HOLD/npu-queue), so the fix scopes both consumers of that regex by it.
#
# The kernel log is a FILE (GPU_HOLD_KLOG_FILE), the same override the during-hold test uses: "a fault two minutes ago"
# cannot be produced on demand. Four arms, and the two that must NOT refuse are the ones this issue is about:
#   1 NPU fault in the log, GPU queue   -> the ticket is GRANTED (rc 0) and the log notes the other device's fault
#   2 NPU fault in the log, NPU queue   -> refused (rc 78), as before
#   3 GPU reset in the log, GPU queue   -> refused (rc 78), as before
#   4 GPU reset in the log, NPU queue   -> granted (rc 0)
set -u
hold=${1:-./scripts/gpu-hold}
[ -x "$hold" ] || { echo "not executable: $hold"; exit 2; }

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT INT TERM HUP
fails=0
chk() { if [ "$2" = "$3" ]; then printf 'ok   %s\n' "$1"; else printf 'FAIL %s: got %s, want %s\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi; }

printf '#!/bin/sh\nexit 0\n' > "$scratch/payload.sh"
chmod +x "$scratch/payload.sh"
printf '75000\n' > "$scratch/tctl"

# One run of the hold against a synthetic kernel log and a chosen queue. No device, no token, no journal.
run_arm() {   # <tag> <klog line> <queue-dir> ; echoes the rc
    klog="$scratch/klog.$1"
    printf '%s\n' "$2" > "$klog"
    q="$scratch/$3"
    mkdir -p "$q"
    env -u GPU_HOLD_FAULT_ABORT GPU_HOLD_KLOG_FILE="$klog" GPU_QUEUE_DIR="$q" \
        GPU_TOKEN_FILE="$scratch/token.$1" GPU_HOLD_LOG="$scratch/log.$1" \
        GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_COOLDOWN=0 \
        GPU_HOLD_MINUTES=1 GPU_HOLD_RESET_WINDOW_MIN=10 \
        GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE="$scratch/tctl" GPU_HOLD_THERMAL_FLAG="$scratch/x" \
        timeout 60 "$hold" -t 1 -l scopetest$1 -- "$scratch/payload.sh" > "$scratch/out.$1" 2>&1
    echo $?
}

npu_fault='[Fri Nov 20 12:59:43 2026] amdxdna 0000:c6:00.1: aie2_tdr_dead_fw_recover_all: Array stopped dispatching, power-cycling NPU to recover'
gpu_reset='[Fri Nov 20 12:59:43 2026] amdgpu 0000:c5:00.0: [gfxhub] page fault (src_id:0 ring:0 vmid:1 pasid:32769)'

chk "an NPU fault does not refuse a GPU ticket (engine#2961)" "$(run_arm a "$npu_fault" gpu-queue)" 0
chk "and the GPU ticket's log says the NPU fault was seen and did not refuse it" \
    "$(grep -c 'do not refuse this GPU ticket' "$scratch/log.a" 2> /dev/null)" 1
chk "an NPU fault still refuses an NPU ticket" "$(run_arm b "$npu_fault" npu-queue)" 78
chk "a GPU reset still refuses a GPU ticket" "$(run_arm c "$gpu_reset" gpu-queue)" 78
chk "a GPU reset does not refuse an NPU ticket" "$(run_arm d "$gpu_reset" npu-queue)" 0
chk "a clean log grants either ticket" "$(run_arm e '[Fri Nov 20 12:59:43 2026] usb 1-1: new device' gpu-queue)" 0

if [ "$fails" = 0 ]; then echo "gpu_hold_device_scope: all arms ok"; else echo "gpu_hold_device_scope: $fails failure(s)"; fi
exit $([ "$fails" = 0 ] && echo 0 || echo 1)
