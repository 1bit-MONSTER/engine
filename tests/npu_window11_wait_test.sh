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
# npu_window11_wait_test.sh -- the readiness arm's budget and message in scripts/npu-window11.sh (engine#3200).
#
# Device-free: a stub stands in for `1bit`, so nothing reaches the NPU and no npu-hold ticket is needed. The script
# has TWO ways to fail to reach /health and only one of them used to say why:
#   * the serve DIES                  -> "serve died; see <log>" + tail (already worked, and pinned here so the fix
#                                        cannot have taken that arm's diagnostics away),
#   * the serve STAYS ALIVE, never binds -> what a 14B host-GEMM arm looks like while warming its weight cache. This
#                                        printed "serve never became ready" and nothing else: no budget, no log name,
#                                        no tail.
# usage: tests/npu_window11_wait_test.sh <scripts/npu-window11.sh>
set -u
script=${1:?usage: npu_window11_wait_test.sh path/to/npu-window11.sh}
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
mkdir -p "$work/kernels"
: > "$work/container.q4nx"

# Stub 1: alive, prints the warm-up line, never binds -- the case the issue is about.
cat > "$work/never-binds" <<'STUB'
#!/bin/bash
echo "1bit forward-serve: warming (weight cache; first request would otherwise pay it)"
sleep 120
STUB
# Stub 2: dies with a reason, the arm that already reported properly.
cat > "$work/dies" <<'STUB'
#!/bin/bash
echo "init_elf failed: bad magic"
exit 1
STUB
chmod +x "$work/never-binds" "$work/dies"

run() {   # run <stub> <budget> -> stderr in $work/err, rc in $rc
    local stub=$1 budget=$2
    ONEBIT_BIN="$work/$stub" NPU_WINDOW11_SERVE_WAIT_S="$budget" \
    timeout 60 bash "$script" "$work/container.q4nx" "$work/kernels" \
        --label t --out "$work/o.log" > "$work/err" 2>&1
    rc=$?
}

start=$(date +%s); run never-binds 1; elapsed=$(( $(date +%s) - start ))

check "a serve that never binds fails the run" '[ "$rc" -ne 0 ]'
check "the message names the budget it expired on" 'grep -q "never became ready within 1s" "$work/err"'
check "…the log it never wrote, by name" 'grep -q "see .*o\.log\.serve" "$work/err"'
check "…and the child's own last line, which is the whole diagnosis" \
      'grep -q "warming (weight cache" "$work/err"'
check "a 1 s budget expires in about a second, not in 600" '[ "$elapsed" -le 10 ]'

# The other arm: it already printed the log, so pin it -- a fix to the silent arm must not quietly cost this one.
run dies 10
check "a serve that DIES still names its log and tails it" \
      '[ "$rc" -ne 0 ] && grep -q "serve died; see .*o\.log\.serve" "$work/err" && grep -q "init_elf failed" "$work/err"'

# Validation happens BEFORE the spawn, so a bad value must not leave a serve log behind at all.
rm -f "$work/o.log.serve"
run never-binds 0
check "a zero budget is refused, and says so" '[ "$rc" -eq 2 ] && grep -q "must be a positive integer of seconds (got .0.)" "$work/err"'
check "…and nothing was spawned for it" '[ ! -f "$work/o.log.serve" ]'
run never-binds 60x
check "a non-numeric budget is refused, and says so" '[ "$rc" -eq 2 ] && grep -q "must be a positive integer of seconds (got .60x.)" "$work/err"'

if [ "$fail" = 0 ]; then echo "npu_window11_wait_test: pass"; exit 0; fi
echo "npu_window11_wait_test: $fail failure(s)"; exit 1
