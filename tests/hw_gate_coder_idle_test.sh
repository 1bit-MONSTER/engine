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
# tests/hw_gate_coder_idle_test.sh -- the gate's "is the coder busy?" probe must be able to say "I cannot tell".
#
# What it pins, and why it is a test at all: `wait_coder_idle` exists so an HRX step never starts while the
# PR-Agent's model is generating. Its parser used to answer `0` on ANY exception, and `curl -s` exits 0 on a 404, so
# a wrong URL, a refused connection, a HTML error page and a genuinely idle coder were all the same answer -- the
# probe could not fail, and the gate proceeded believing the coder was idle. The four inputs below separate them:
# `0` idle, `N` busy, **`-1` cannot tell** (reported, not silently read as idle).
#
# The two functions are extracted from the real script rather than copied, so the test cannot drift from the code;
# if the extraction stops matching, this test fails and says so instead of passing on stale text.
#
# Usage: hw_gate_coder_idle_test.sh [repo root]
set -u
root=${1:-.}
script="$root/scripts/hw-gate.sh"
fails=0
chk() {  # want got what
    if [ "$1" = "$2" ]; then echo "ok   $3"; else echo "FAIL $3: want '$1', got '$2'"; fails=$((fails + 1)); fi
}

json_fn=$(awk '/^coder_busy_from_json\(\)/,/^}/' "$script")
idle_fn=$(awk '/^wait_coder_idle\(\)/,/^}/' "$script")
if [ -z "$json_fn" ] || [ -z "$idle_fn" ]; then
    echo "FAIL cannot extract coder_busy_from_json/wait_coder_idle from $script -- update the extraction" >&2
    exit 1
fi
eval "$json_fn"
eval "$idle_fn"

chk 0 "$(printf '[]' | coder_busy_from_json)" "an empty slots list is idle"
chk 1 "$(printf '[{"is_processing":true},{"is_processing":false}]' | coder_busy_from_json)" "one processing slot is busy"
chk -1 "$(printf '<html>404 Not Found</html>' | coder_busy_from_json)" "an HTML error page is NOT idle"
chk -1 "$(printf '' | coder_busy_from_json)" "an empty body is NOT idle"
chk -1 "$(printf '{"error":{"message":"nf"}}' | coder_busy_from_json)" "a JSON error object is NOT idle"

# End to end through wait_coder_idle: a refused connection must return (bounded, it is not the gate's job to hang on a
# dead coder) but it must SAY it could not tell, and it must get there without sleeping through CODER_WAIT.
notes=""
note() { notes="$notes$*|"; }
CODER_URL="http://127.0.0.1:9"      # discard port: nothing listens, so curl -sf exits 7
CODER_WAIT=900
start=$(date +%s)
wait_coder_idle
elapsed=$(( $(date +%s) - start ))
chk 0 "$?" "a refused connection still lets the gate proceed"
chk yes "$(case "$notes" in *"did not answer"*) echo yes ;; *) echo no ;; esac)" "it warns that it could not ask the coder"
chk yes "$([ "$elapsed" -lt 30 ] && echo yes || echo no)" "and it does not wait out CODER_WAIT to say so (took ${elapsed}s)"

if [ "$fails" != 0 ]; then
    echo "hw_gate_coder_idle: $fails failure(s)"
    exit 1
fi
echo "hw_gate_coder_idle: ok"
