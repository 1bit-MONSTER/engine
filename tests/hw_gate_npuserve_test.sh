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
# hw_gate_npuserve_test.sh: the hardware gate's `npuserve` phase (engine#2106), end to end, with no device.
#
# scripts/hw-gate.sh --phases npuserve is run against the stand-in serve that tools/npu_serve_smoke.py writes
# (--write-stand-in): a process that prints a route line and answers with the timings it is told to. Three runs:
#   a fast-route stand-in at 90 tok/s  -> the phase is PASS and the gate exits 0
#   a slow-route stand-in at 1.9 tok/s -> the phase is FAIL, names route and rate, and the gate exits 1
#   nothing configured                 -> the phase is SKIP and the gate exits 0
# The gate is kept off the fleet's GPU and NPU queues (HW_GATE_GPU_HOLD=0, HW_GATE_TOKEN=none, HW_GATE_NPU_HOLD=0), its
# lock and output live in a directory made here, and HW_GATE_NPU_DEV points the "is there an NPU" probe at a file
# that exists, so the phase runs on a machine with none.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d "$PWD/hw_gate_npuserve_test.XXXXXX") || exit 1      # the working directory, not /tmp
trap 'rm -rf "$work"' EXIT
fail=0
check() { if eval "$1"; then echo "ok    $2"; else echo "FAIL  $2"; fail=1; fi; }

python3 "$root/tools/npu_serve_smoke.py" --write-stand-in "$work/fake1bit" || { echo "FAIL  cannot write the stand-in"; exit 1; }
touch "$work/npu-present"

gate() {  # gate <name> [VAR=value ...]: runs the phase, leaves its text in $work/<name>.txt and its exit in $rc
    local name=$1; shift
    env HW_GATE_GPU_HOLD=0 HW_GATE_TOKEN=none HW_GATE_NPU_HOLD=0 HW_GATE_NPU_DEV="$work/npu-present" \
        HW_GATE_LOCK="$work/lock" HW_GATE_OUT="$work/out-$name" "$@" \
        timeout 120 "$root/scripts/hw-gate.sh" --phases npuserve > "$work/$name.txt" 2>&1
    rc=$?
}
port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

gate fast FAKE_MODE=fast FAKE_TPS=90 HW_GATE_NPU_SERVE_BIN="$work/fake1bit" HW_GATE_NPU_SERVE_MODEL=model HW_GATE_NPU_SERVE_PORT="$(port)"
check '[ "$rc" = 0 ]' "a fast-route serve at 90 tok/s: the gate exits 0 (got $rc)"
check 'grep -q "^| npuserve | PASS |" "$work/fast.txt"' "... and the summary row is PASS"
check 'grep -q "rate ok median 90.0 tok/s" "$work/fast.txt"' "... with the measured rate in the row"

gate slow FAKE_MODE=slow FAKE_TPS=1.9 HW_GATE_NPU_SERVE_BIN="$work/fake1bit" HW_GATE_NPU_SERVE_MODEL=model HW_GATE_NPU_SERVE_PORT="$(port)"
check '[ "$rc" = 1 ]' "a slow-route serve at 1.9 tok/s: the gate exits 1 (got $rc)"
check 'grep -q "^| npuserve | FAIL |" "$work/slow.txt"' "... and the summary row is FAIL"
check 'grep -q "route FAIL the serve log names the per-op forward" "$work/slow.txt"' "... naming the route"
check 'grep -q "rate FAIL median 1.9 tok/s" "$work/slow.txt"' "... and the rate"
check 'grep -q "content ok" "$work/slow.txt"' "... while the answers themselves were fine (the reason the phase exists)"

# a fast route under a floor it does not reach: the floor is the gate's own setting
gate floor FAKE_MODE=fast FAKE_TPS=90 HW_GATE_NPU_SERVE_FLOOR=120 HW_GATE_NPU_SERVE_BIN="$work/fake1bit" HW_GATE_NPU_SERVE_MODEL=model HW_GATE_NPU_SERVE_PORT="$(port)"
check '[ "$rc" = 1 ] && grep -q "rate FAIL median 90.0 tok/s (floor 120.0" "$work/floor.txt"' "HW_GATE_NPU_SERVE_FLOOR=120 fails a 90 tok/s serve on the rate alone"

gate dead FAKE_MODE=dead HW_GATE_NPU_SERVE_BIN="$work/fake1bit" HW_GATE_NPU_SERVE_MODEL=model HW_GATE_NPU_SERVE_PORT="$(port)"
check '[ "$rc" = 1 ] && grep -q "^| npuserve | FAIL |" "$work/dead.txt" && grep -q "never became ready" "$work/dead.txt"' "a serve that dies at start is FAIL, and the row says it never became ready"

gate skip
check '[ "$rc" = 0 ] && grep -q "^| npuserve | SKIP |" "$work/skip.txt"' "with no binary and model configured the phase is SKIP and the gate exits 0"

rm -f "$work/npu-present"
gate nodev HW_GATE_NPU_SERVE_BIN="$work/fake1bit" HW_GATE_NPU_SERVE_MODEL=model
check '[ "$rc" = 0 ] && grep -q "^| npuserve | SKIP |" "$work/nodev.txt" && grep -q "no NPU" "$work/nodev.txt"' "with no NPU the phase is SKIP even when configured"

check '! pgrep -f "$work/fake1bit" > /dev/null' "no stand-in serve is left running"
[ "$fail" = 0 ] && echo "hw_gate_npuserve: all checks passed"
exit "$fail"
