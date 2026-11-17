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
# gpu_hold_queue_state_test.sh: a lane learns at QUEUE TIME that its take will be refused (engine#2494).
#
# Measured 2026-11-16: since the shared GPU_FAULT_RE was deployed, an NPU line sat inside the take-time window for 27 %
# of the day while the GPU logged nothing, so a GPU lane queues, waits, and loses the position to a refusal it could
# have predicted. The take-time refusal is unchanged and remains the authority; this is the same pattern, read one step
# earlier, and it must never decide anything.
#
# Arms:
#   1. a fault line in the window -> the warning appears at queue time, names the device, and the ticket is NOT refused
#      by it (the payload runs once the queue turns);
#   2. no fault line -> no warning (the arm that keeps the warning meaningful);
#   3. an NPU line is named as the NPU, because that is the case a GPU lane cannot recognise from its own job;
#   4. GPU_HOLD_QUEUE_STATE_OFF=1 silences it, for a lane that has read it once too often.
#
# No device: a private queue, a fixture kernel log, and a payload of `true`.
set -uo pipefail
HOLD=${1:?usage: gpu_hold_queue_state_test.sh <path-to-gpu-hold>}
[ -f "$HOLD" ] || { echo "  SKIP no gpu-hold at $HOLD"; exit 77; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ghqs.XXXXXX") || exit 77
trap 'rm -rf "$tmp"' EXIT
fails=0

run() {   # run <tag> <klog-content> [extra env...]
    local tag=$1 content=$2; shift 2
    mkdir -p "$tmp/$tag/queue"
    printf '%s' "$content" > "$tmp/$tag/klog"
    env -u GPU_HOLD_QUEUE_STATE_OFF "$@" HOME="$tmp/$tag" ONEBIT_HOLD_DIR="$tmp/$tag" \
        GPU_HOLD_KLOG_FILE="$tmp/$tag/klog" GPU_HOLD_COOLDOWN=0 GPU_HOLD_RELEASE_WAIT=0 \
        GPU_HOLD_TCTL_START=999 GPU_HOLD_THERMAL_FLAG="$tmp/none" \
        timeout 60 bash "$HOLD" -l "qs-$tag" -t 0.1 -w 20 true 2>&1
}

gpu_line='kernel: amdgpu 0000:c5:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:9 pasid:26332)
'
npu_line='kernel: amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0002 address=0x7fa381807000 flags=0x0077]
'

# 1 + 3. a fault line -> the warning, naming the device, and the payload still runs
out=$(run npu "$npu_line")
if printf '%s\n' "$out" | grep -q "queued while the kernel log carries a NPU fault line"; then
    echo "  PASS arm1/3: an NPU fault in the window is reported at queue time, as NPU"
else
    echo "  FAIL arm1/3: no queue-time warning for an NPU fault line; gpu-hold said:"
    printf '%s\n' "$out" | tail -4 | sed 's/^/      /'
    fails=$((fails + 1))
fi
# arm1b: the warning is ADVISORY. Same fault at queue time, cleared before the take: the ticket must be granted and the
# payload must run -- which is exactly what a lane needs, because the warning fires when the window is read at queue
# time and the refusal reads it again at take.
mkdir -p "$tmp/adv/queue"
printf '%s' "$npu_line" > "$tmp/adv/klog"
# a holder has to own the token first, or the take happens in the same second as the warning and there is no gap in which
# the window can clear -- which is itself the useful case: this arm needs a QUEUE, not an empty box.
sleep 600 & holder=$!
printf 'gpu-hold another-lane pid %s 2026-11-16T00:00:00-04:00\n' "$holder" > "$tmp/adv/gpu-token"
( sleep 2; printf 'kernel: the burst has passed\n' > "$tmp/adv/klog"; sleep 3; kill "$holder" 2> /dev/null ) &
out_adv=$(env HOME="$tmp/adv" ONEBIT_HOLD_DIR="$tmp/adv" GPU_HOLD_KLOG_FILE="$tmp/adv/klog" \
    GPU_HOLD_COOLDOWN=0 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_TCTL_START=999 GPU_HOLD_THERMAL_FLAG="$tmp/none" \
    timeout 60 bash "$HOLD" -l qs-adv -t 0.1 -w 20 true 2>&1)
if printf '%s\n' "$out_adv" | grep -q "queued while the kernel log carries" &&
   printf '%s\n' "$out_adv" | grep -q "command exited 0"; then
    echo "  PASS arm1b: warned at queue time, then GRANTED once the window cleared -- advisory, not a decision"
else
    echo "  FAIL arm1b: want a queue-time warning AND a completed payload; gpu-hold said:"
    printf '%s\n' "$out_adv" | tail -4 | sed 's/^/      /'
    fails=$((fails + 1))
fi
outg=$(run gpu "$gpu_line")
printf '%s\n' "$outg" | grep -q "carries a GPU fault line" \
    && echo "  PASS arm3: a gfxhub line is reported as GPU" \
    || { echo "  FAIL arm3: a gfxhub line was not reported as GPU"; fails=$((fails + 1)); }

# 2. no fault line -> no warning
out2=$(run clean 'kernel: usb 1-1: new device
')
if printf '%s\n' "$out2" | grep -q "queued while the kernel log carries"; then
    echo "  FAIL arm2: the warning fired on a clean window -- it would then be noise every lane ignores"
    fails=$((fails + 1))
else
    echo "  PASS arm2: no warning on a clean window"
fi

# 4. the opt-out
out4=$(run off "$npu_line" GPU_HOLD_QUEUE_STATE_OFF=1)
if printf '%s\n' "$out4" | grep -q "queued while the kernel log carries"; then
    echo "  FAIL arm4: GPU_HOLD_QUEUE_STATE_OFF=1 did not silence it"
    fails=$((fails + 1))
else
    echo "  PASS arm4: GPU_HOLD_QUEUE_STATE_OFF=1 silences it"
fi

# 5. a fault that arrives DURING the wait is reported too -- the case the deploy actually produced: after the new copy
# went live, all four take-time refusals came after waits of 232-296 s and were triggered by a fault that appeared
# mid-wait, with the queue-time warning (correctly) silent at second zero. Logged once, and it must say it is a wait.
mkdir -p "$tmp/mid/queue"
printf 'kernel: clean at queue time\n' > "$tmp/mid/klog"
sleep 600 & holder5=$!
printf 'gpu-hold another-lane pid %s 2026-11-16T00:00:00-04:00\n' "$holder5" > "$tmp/mid/gpu-token"
( sleep 3; printf 'kernel: amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0002 address=0x1 flags=0x7]\n' >> "$tmp/mid/klog" ) &
( sleep 12; kill "$holder5" 2> /dev/null ) &
out5=$(env HOME="$tmp/mid" ONEBIT_HOLD_DIR="$tmp/mid" GPU_HOLD_KLOG_FILE="$tmp/mid/klog" \
    GPU_HOLD_COOLDOWN=0 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_TCTL_START=999 GPU_HOLD_THERMAL_FLAG="$tmp/none" \
    timeout 60 bash "$HOLD" -l qs-mid -t 0.1 -w 30 true 2>&1)
n_mid=$(printf '%s\n' "$out5" | grep -c "still waiting after .* carries a NPU fault line")
n_queue=$(printf '%s\n' "$out5" | grep -c "queued while the kernel log carries")
if [ "$n_mid" = 1 ] && [ "$n_queue" = 0 ]; then
    echo "  PASS arm5: a fault arriving during the wait is reported once, as a wait"
else
    echo "  FAIL arm5: want exactly one 'still waiting after Ns ... NPU fault' and no queue-time line; got mid=$n_mid queue=$n_queue"
    printf '%s\n' "$out5" | grep -E "warning|refusing" | head -3 | sed 's/^/      /'
    fails=$((fails + 1))
fi

if [ "$fails" = 0 ]; then echo "gpu_hold_queue_state: 5 arms pass"; exit 0; fi
echo "gpu_hold_queue_state: $fails failure(s)"
exit 1
