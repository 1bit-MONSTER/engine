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
# npu_window11_wait_test.sh -- both readiness-failure arms of scripts/npu-window11.sh (engine#3196).
#
# Device-free: a stub 1bit stands in for the serve, so no NPU and no container. Two worlds, because the script has two
# ways to fail to reach /health and only one of them used to say why:
#   * the serve DIES            -> "serve died; see <log>" plus the log's tail (this arm already worked),
#   * the serve stays ALIVE and never binds -> "never became ready within Ns" plus the log's tail (this is the arm that
#     reported nothing; a warm-killed 14B arm looks exactly like this).
# usage: tests/npu_window11_wait_test.sh <scripts/npu-window11.sh>
set -u
script=${1:?usage: npu_window11_wait_test.sh path/to/npu-window11.sh}
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
mkdir -p "$work/kernels"; : > "$work/model.q4nx"; : > "$work/kernels/base"
printf 'warming (weight cache; first request would otherwise pay it)\n' > "$work/stub-warm"

# world A: alive, never binds. world B: dies at once.
cat > "$work/stub-alive" <<STUB
#!/bin/bash
cat "$work/stub-warm"
sleep 120
STUB
cat > "$work/stub-dead" <<'STUB'
#!/bin/bash
echo "init_elf failed: no such kernel"
exit 3
STUB
# dies in the GAP between the loop's last `kill -0` and the final curl: prints the warming line, outlives the first
# liveness check, then exits. Without this world the suite cannot see a readiness arm that ASSERTS liveness instead of
# checking it (engine#3200 review, second round).
cat > "$work/stub-dies-late" <<'STUB'
#!/bin/bash
echo "warming (weight cache; first request would otherwise pay it)"
sleep 0.5
exit 1
STUB
chmod +x "$work/stub-alive" "$work/stub-dead" "$work/stub-dies-late"

# A FREE PORT, passed with --port, because the script assigns PORT=8110 itself and ignores the environment: a test that
# let it default would curl whatever else listens on 8110 and, if that answered /health, would post the fixture prompts
# at a stranger's server. That is the cross-lane interference a gate must never cause.
free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

run() {  # run <stub> <budget> ; stderr+stdout in $work/err, rc in $rc
    local stub=$1 budget=$2 port; port=$(free_port)
    ONEBIT_BIN="$work/$stub" NPU_WINDOW11_SERVE_WAIT_S="$budget" \
    timeout 60 bash "$script" "$work/model.q4nx" "$work/kernels" --label t --out "$work/o.log" --port "$port" \
        > "$work/err" 2>&1
    rc=$?
}

start=$(date +%s); run stub-alive 1; elapsed=$(( $(date +%s) - start ))
check "a serve that never binds fails the run" '[ "$rc" -ne 0 ]'
# The alive world must take the READINESS arm, not the death arm: an outer `timeout` that kills the process
# group makes the old script report "serve died", which is how two of these arms passed on the old code for
# the wrong reason when this test was first written. Asserting the absence of that line is what makes the
# two worlds tell themselves apart.
check "…and does not report a death it did not see" '! grep -q "serve died" "$work/err"'
check "…and says the elapsed time and the budget it expired on" 'grep -q "never became ready after 1s (budget 1s)" "$work/err"' 
check "…and names the serve log" 'grep -q "see .*o\.log\.serve" "$work/err"'
# engine#3200 review (#3203): the interpretation is the diagnosis -- the whole defect was a warming arm being read as a
# dead one, so the message says which of the two it is rather than leaving the reader to work it out from the log.
check "…and says this is the warming window, not a dead server" 'grep -q "warming window, not a dead server" "$work/err"'
check "…and names the knob to raise" 'grep -q "raise NPU_WINDOW11_SERVE_WAIT_S" "$work/err"' 
check "…and prints that log's tail, which carries the reason" 'grep -q "warming (weight cache" "$work/err"'
check "a 1 s budget expires in about a second, not in 600" '[ "$elapsed" -le 10 ]'

run stub-dead 600
check "a serve that dies still reports the death with its log tail (the arm that already worked)" \
      '[ "$rc" -ne 0 ] && grep -q "serve died" "$work/err" && grep -q "init_elf failed" "$work/err"'

# The gap: the stub above outlives the first liveness check and dies before the last one, so this is the world in which
# the old final arm printed "the process is alive, so this is the warming window" about a process that had exited.
run stub-dies-late 1
check "a serve that dies just before the last check reports the DEATH, not the warming window" \
      '[ "$rc" -ne 0 ] && grep -q "serve died" "$work/err" && ! grep -q "the process is alive" "$work/err"'

# The budget is validated BEFORE the spawn (engine#3200 review): with a bad value nothing should have been started, and
# the observable of that is that the serve log was never created. A validation placed after `SPID=$!` still exits 2, but
# a 14B forward-serve has by then begun its weight load -- the exit code is right and the work happened anyway.
# The budget is validated above EVERY side effect, so the observable is that the script's own log was never created. A
# check *after* the spawn cannot be observed this way -- the parent exits in microseconds and the child is killed before
# it opens its redirect, so "no serve log" is true either way and the arm would pass for the wrong reason. That is how
# the first version of this arm was written, and it was removed rather than shipped.
rm -f "$work/o.log"
run stub-alive 0
check "a zero budget is refused, and says so" \
      '[ "$rc" -ne 0 ] && grep -q "must be a positive integer of seconds (got .0.)" "$work/err"'
check "…and no side effect happened: the script's own log was never created" '[ ! -e "$work/o.log" ]'

if [ "$fail" = 0 ]; then echo "npu_window11_wait_test: pass"; exit 0; fi
echo "npu_window11_wait_test: $fail failure(s)"; exit 1
