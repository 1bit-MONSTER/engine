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
# npu_kernel_bitx_wait_test.sh -- the serve budget and the failure message of scripts/npu-kernel-bitx.sh (engine#3194).
#
# Device-free: a stub stands in for `1bit`, under a fake ticket file, with a dummy container. The stub does exactly what a
# 14B host-GEMM arm does while it is being killed by the old 60 s budget -- prints the warming line and never binds --
# so both halves of the fix are checked without an NPU:
#   * the message names the BUDGET it expired on, and
#   * the message carries the child's own last line, which the old grep threw away, and
#   * BITX_SERVE_WAIT_S is honoured (a 1 s budget expires in about a second) and validated (0 and non-numeric refuse).
# usage: tests/npu_kernel_bitx_wait_test.sh <scripts/npu-kernel-bitx.sh>
set -u
script=${1:?usage: npu_kernel_bitx_wait_test.sh path/to/npu-kernel-bitx.sh}
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
mkdir -p "$work/kernels" "$work/out"
: > "$work/container.q4nx"
cat > "$work/stub-1bit" <<'STUB'
#!/bin/bash
echo "1bit forward-serve: warming (weight cache; first request would otherwise pay it)"
sleep 120
STUB
chmod +x "$work/stub-1bit"
printf 'fake-ticket-line\n' > "$work/token"

run() {   # run <budget> -> stderr in $work/err, rc in $rc
    local budget=$1
    BITX_OUT="$work/out" ONEBIT_BIN="$work/stub-1bit" NPU_TOKEN_FILE="$work/token" \
    NPU_HOLD_TICKET="fake-ticket-line" BITX_SERVE_WAIT_S="$budget" \
    timeout 60 bash "$script" "$work/container.q4nx" "$work/kernels" '<1>' > "$work/err" 2>&1
    rc=$?
}

start=$(date +%s); run 1; elapsed=$(( $(date +%s) - start ))

check "a serve that never binds fails the run" '[ "$rc" -ne 0 ]'
check "the message names the budget it expired on" 'grep -q "did not come up within 1s" "$work/err"'
check "the message carries the child's own last line (the reason the old grep threw away)" \
      'grep -q "warming (weight cache" "$work/err"'
check "…and it is labelled as the last line of the log, not as a matched error" \
      'grep -q "last line of .*\.log: 1bit forward-serve: warming" "$work/err"'
check "a 1 s budget expires in about a second, not in 60" '[ "$elapsed" -le 10 ]'

# The script aggregates a failed arm into its own exit code (1), so a usage error inside serve() reads as a
# failing run with the reason on stderr rather than as 2 -- assert both halves, not a code the script does not promise.
run 0
check "a zero budget is refused, and says so" '[ "$rc" -ne 0 ] && grep -q "must be a positive integer of seconds (got .0.)" "$work/err"'
run 60x
check "a non-numeric budget is refused, and says so" '[ "$rc" -ne 0 ] && grep -q "must be a positive integer of seconds (got .60x.)" "$work/err"'

if [ "$fail" = 0 ]; then echo "npu_kernel_bitx_wait_test: pass"; exit 0; fi
echo "npu_kernel_bitx_wait_test: $fail failure(s)"; exit 1
