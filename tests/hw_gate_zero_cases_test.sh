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
# A phase that tested NOTHING must not leave the gate green (review of engine#2523; engine#2522's rule).
#
# The pin's `test-backend-ops` refuses to report a pass over an empty selection: it exits 1 and prints
# "error: no test case ran (...) : refusing to report a pass". The gate turns that into `ops SKIP` with the selection
# that produced it -- correct, because it is a filter problem rather than a kernel result -- but a SKIP with no
# `infra=1` leaves the verdict reading PASS, so a run that measured nothing is indistinguishable from a clean one.
#
# No GPU, no model, no compiler: a fake --bin whose test-backend-ops prints that line, an empty model dir, and
# HW_GATE_TOKEN=none so no ticket is taken. The arm is the verdict and the exit code, which is what CI reads.
set -u
root=${1:-$(cd "$(dirname "$0")/.." && pwd)}   # the family's convention: ctest passes ${CMAKE_SOURCE_DIR}
fail=0
scratch=$(mktemp -d "${TMPDIR:-$HOME/.cache}/hwgate-zc.XXXXXX") || exit 1
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/bin"
for b in llama-server test-backend-ops llama-perplexity; do
    {
        echo '#!/bin/bash'
        echo 'if [ "$(basename "$0")" = test-backend-ops ]; then'
        echo '    echo "error: no test case ran (check -b/-o/-p and --test-file): refusing to report a pass"'
        echo '    exit 1'
        echo 'fi'
        echo 'exit 0'
    } > "$scratch/bin/$b"
    chmod +x "$scratch/bin/$b"
done
# thermal-run's contract is "run the rest of the argv"; /bin/true exits 0 and the step then reports nothing at all,
# which is how my first version of this arm turned a passing test into rc 1 with no rows.
printf '#!/bin/sh\nexec "$@"\n' > "$scratch/thermal.sh"
chmod +x "$scratch/thermal.sh"
mkdir -p "$scratch/run"
# Every input this arm depends on is pinned, and each line is a hang that was measured on CI (e2e build 43752965195,
# 120.02 s, rc 124): the step loop retries rc 76 up to HW_GATE_MEM_RETRIES times, and a runner whose MemAvailable sits
# under the 12 GB floor never reaches the zero-case path at all -- the arm then reported rc 124 instead of the verdict
# it was written for. thermal-run is not on a CI runner either, so a real one would have been 127. The queue is not
# what this test is about: HW_GATE_TOKEN=none says so, and the memory floor is 1 GB so it cannot wait.
env HW_GATE_MODELS="$scratch" HW_GATE_TOKEN=none HW_GATE_PRIVATE=0 HW_GATE_COOL_C=1000 \
    HW_GATE_MEM_FLOOR=1 HW_GATE_MEM_RETRIES=0 HW_GATE_THERMAL="$scratch/thermal.sh" \
    HW_GATE_CODER_URL=http://127.0.0.1:9 HW_GATE_LOCK="$scratch/npu.lock" \
    HW_GATE_STORE="$scratch/run/store" HW_GATE_OUT="$scratch/run/out" GITHUB_STEP_SUMMARY= \
    timeout 90 "$root/scripts/hw-gate.sh" --phases ops --bin "$scratch/bin" > "$scratch/run/stdout" 2>&1
rc=$?
out=$scratch/run/stdout

grep -q '| ops | SKIP |' "$out" || { echo "FAIL: a zero-case ops run did not produce an ops SKIP row"; fail=1; }
grep -q 'INCOMPLETE' "$out" || { echo "FAIL: a run that tested nothing did not read as INCOMPLETE"; fail=1; }
grep -q '## HRX hardware gate: PASS' "$out" && { echo "FAIL: a run that tested nothing read as PASS"; fail=1; }
[ "$rc" = 2 ] || { echo "FAIL: rc $rc for a zero-case phase, want 2 (infra, not green)"; fail=1; }
# A fixture that fails without saying why costs a CI cycle each time (measured: 8 minutes to learn "rc 124"). The gate's
# own output is what names the cause, so print it whenever an arm failed.
if [ "$fail" != 0 ]; then
    echo "--- the gate's output follows ---"
    tail -40 "$out" 2> /dev/null || echo "(no output file at $out)"
fi
[ "$fail" = 0 ] && echo "ok   a zero-case ops phase reads INCOMPLETE and exits 2, not PASS"
[ "$fail" = 0 ]
