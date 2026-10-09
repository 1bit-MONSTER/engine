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
# engine#315 regression cell (task-6).
#
# Reproduces the recorded baseline from docs/315-preregistration.md 9b, from the SAME source revision as
# the fix cell (~/wt/llama-315fix), because test-backend-ops is absent from the study's builds (their
# recipe sets -DLLAMA_BUILD_TESTS=OFF). Two checks:
#
#   1. `test-backend-ops -o MUL_MAT_ID -b HRX0`, twice, using the study's own wrapper and line format.
#      Recorded baseline: 108/108 passed, fail=0, supported=110, rc=0, stable across repeats.
#   2. dense-model control at ~512 tokens (zaya1-8b, the dense model the study pairs with GLM in
#      ~/wt/hrxsys-correct.sh), asserting valid UTF-8 output and no fault signature on either face.
#      The MoE path is the faulting one, so this is the "is the fix inert on ordinary inference" check.
#
# Takes the box lock, so it cannot collide with the acceptance campaign.
set -uo pipefail

SRC=${SRC:-$HOME/wt/llama-315fix}
B=${B:-$HOME/wt/315fix-tests-build}
FBIN=${FBIN:-$HOME/wt/315fix-build/bin}
M=$HOME/models
O=$HOME/wt/315reg
LOG=$HOME/wt/315fix-tests-build.log
PORT=${PORT:-19401}
PIN=e44c9d01a4d5110cecaf45472a717f46cd1abbf9
export IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
mkdir -p "$O"

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "=== regression cell: source $SRC at $(git -C "$SRC" rev-parse --short HEAD) (pin ${PIN:0:9}) ==="
[ "$(git -C "$SRC" rev-parse HEAD)" = "$PIN" ] || { echo "ABORT: source is not at the pin"; exit 1; }
G=$SRC/ggml/src/ggml-hrx/ggml-hrx.cpp
grep -q 'memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT;' "$G" || { echo "ABORT: the fix is not in the source being validated"; exit 1; }
if grep -A1 'HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;' "$G" | grep -q 'if (direct_host_binding)'; then
  echo "ABORT: coherence still gated - wrong source"; exit 1
fi
say "source assertion: the fix is in effect"

# The op-test needs the box to itself (it drives the GPU hard); take the same lock the campaign uses.
exec 9>"$HOME/.cache/lax-decode/box.lock"
flock -w "${LOCK_WAIT:-1800}" 9 || { echo "ABORT: box lock not acquired"; exit 3; }
say "box lock acquired"

say "=== build with tests ON (the only difference from the campaign cell) ==="
rm -rf "$B"
cmake -S "$SRC" -B "$B" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DGGML_HIP=OFF -DGGML_HRX=ON -DGGML_VULKAN=OFF \
  -DHRX_SOURCE_DIR="$HOME/wt/pin-fork-hrx/third_party/hrx-system" \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DBUILD_TESTING=ON \
  > "$LOG" 2>&1 || { echo "ABORT: cmake failed, see $LOG"; tail -20 "$LOG"; exit 1; }
cmake --build "$B" -j 12 --target test-backend-ops llama-server >> "$LOG" 2>&1 \
  || { echo "ABORT: build failed, see $LOG"; tail -30 "$LOG"; exit 1; }
say "op-test binary: $("$B/bin/test-backend-ops" --help 2>&1 | head -1)"

say "=== check 1: test-backend-ops -o MUL_MAT_ID -b HRX0, two reps (study wrapper, study line format) ==="
TBO_OK=0
for rep in 1 2; do
  timeout 1800 "$B/bin/test-backend-ops" -o MUL_MAT_ID -b HRX0 > "$O/MUL_MAT_ID.$rep.log" 2>&1
  rc=$?
  passed=$(grep -E "tests passed" "$O/MUL_MAT_ID.$rep.log" | head -1)
  fail=$(grep -c FAIL "$O/MUL_MAT_ID.$rep.log")
  ok=$(grep -c OK "$O/MUL_MAT_ID.$rep.log")
  echo "tbo MUL_MAT_ID rep$rep rc=$rc $passed fail=$fail supported=$ok" | tee -a "$O/regression.txt"
  if [ "$rc" = "0" ] && [ "$fail" = "0" ] && echo "$passed" | grep -q "108/108"; then
    TBO_OK=$((TBO_OK + 1))
  fi
done
say "check 1: $TBO_OK/2 reps matched the recorded baseline (108/108, fail=0)"
[ "$TBO_OK" = "2" ] || echo "  NOTE: does not match the recorded baseline - regression NOT cleared"

say "=== check 2: dense-model control at ~512 tokens (zaya1-8b) ==="
DENSE=NOTRUN
if [ -x "$FBIN/llama-server" ] && [ -f "$M/zaya1-8b-Q4_K_M.gguf" ]; then
  "$FBIN/llama-server" -m "$M/zaya1-8b-Q4_K_M.gguf" --host 127.0.0.1 --port "$PORT" \
    -ngl 99 -c 4096 --no-warmup --no-webui -t 8 > "$O/dense-server.log" 2>&1 &
  SPID=$!
  up=0
  for _ in $(seq 1 90); do
    sleep 2
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SPID 2>/dev/null || break
  done
  if [ "$up" = "1" ]; then
    curl -s -m 600 -X POST "http://127.0.0.1:$PORT/completion" \
      -H 'Content-Type: application/json' \
      -d '{"prompt":"Write a short paragraph about the sea.","n_predict":512,"temperature":0.0}' \
      > "$O/dense.json" 2>&1
    DENSE=$(python3 - "$O/dense.json" "$O/dense-server.log" <<'PY'
import json, re, sys
body, log = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(body, errors="replace"))
except Exception as e:
    print(f"BAD-JSON ({e})"); raise SystemExit
c = d.get("content") or ""
txt = open(log, errors="replace").read()
bad = []
if not c: bad.append("empty content")
if "\ufffd" in c: bad.append("U+FFFD in output")
if re.search(r"HRX returned NaN logits", txt): bad.append("Face A: NaN logits")
if re.search(r"HSA_STATUS_ERROR|Queue error", txt): bad.append("Face A: HSA error")
if re.search(r'"code":500|unparsed Content-only', txt): bad.append("Face B: parse/500")
n = len(c.split())
print(f"chars={len(c)} words~{n} utf8_ok={not any('FFFD' in b for b in bad)} "
      f"{'CLEAN' if not bad else 'FAULT: ' + '; '.join(bad)}")
PY
)
    echo "dense control (~512 tokens): $DENSE" | tee -a "$O/regression.txt"
  else
    DENSE="SERVER-DID-NOT-START"
    echo "dense control: $DENSE (see $O/dense-server.log)" | tee -a "$O/regression.txt"
  fi
  kill -TERM $SPID 2>/dev/null; sleep 4; kill -9 $SPID 2>/dev/null
else
  echo "dense control: $DENSE (binary or model missing)" | tee -a "$O/regression.txt"
fi

say "=== summary ==="
tail -4 "$O/regression.txt"
if [ "$TBO_OK" = "2" ] && [ "$DENSE" = "NOTRUN" ]; then say "REGRESSION_DONE tbo=ok dense=not-run"; fi
if [ "$TBO_OK" = "2" ] && [ "$DENSE" != "NOTRUN" ] && ! echo "$DENSE" | grep -q FAULT; then
  say "REGRESSION_DONE tbo=ok dense=clean"
else
  say "REGRESSION_DONE tbo=$TBO_OK/2 dense=$DENSE"
fi
