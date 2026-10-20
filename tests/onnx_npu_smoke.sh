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
# Smoke test for the Linux NPU route of `1bit serve --device onnx` (docs/onnx.md, "Linux NPU"):
# serves an NPU ONNX Runtime GenAI model through ryzenai-server with the environment serve
# builds from ONEBIT_RYZENAI_ROOT, and checks /health, /v1/models, a chat reply, that the child
# holds the NPU open (/dev/accel/accel0 among its descriptors), the decode rate, and that
# stopping serve stops the child. Skips (exit 77) without Ryzen AI Software, the NPU or a model.
#
# usage: tests/onnx_npu_smoke.sh path/to/1bit [model dir]
#   ONEBIT_RYZENAI_ROOT   the Ryzen AI Software venv (deployment/lib inside), required
#   ONEBIT_XRT_ROOT       the XRT prefix (else $XILINX_XRT, else /opt/xilinx/xrt)
#   ONEBIT_ONNX_NPU_MODEL the model directory when none is given
set -uo pipefail

bin=${1:?usage: onnx_npu_smoke.sh path/to/1bit [model dir]}
model=${2:-${ONEBIT_ONNX_NPU_MODEL:-}}
skip() { echo "skip: $1"; exit 77; }
[ -n "${ONEBIT_RYZENAI_ROOT:-}" ] || skip "ONEBIT_RYZENAI_ROOT is not set (the Ryzen AI Software venv)"
[ -d "$ONEBIT_RYZENAI_ROOT/deployment/lib" ] || skip "$ONEBIT_RYZENAI_ROOT has no deployment/lib"
[ -c /dev/accel/accel0 ] || skip "no NPU (/dev/accel/accel0)"
[ -n "$model" ] || skip "no model: pass a directory or set ONEBIT_ONNX_NPU_MODEL"
[ -f "$model/genai_config.json" ] || skip "$model has no genai_config.json"
grep -q '"RyzenAI"' "$model/genai_config.json" || skip "$model does not declare the RyzenAI provider"

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
log=$(mktemp)
"$bin" serve -m "$model" --device onnx --port "$port" --alias npu-smoke >"$log" 2>&1 &
pid=$!
# Never SIGKILL a process that holds the NPU: an abnormal exit with a live hardware context can
# take the kernel down (amdxdna 0.17.0, 2026-10-11). One SIGTERM to serve, and wait.
cleanup() { kill -TERM "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -f "$log"; }
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
api="http://127.0.0.1:$port"

code=000
for _ in $(seq 1 360); do   # model load: up to 3 min
    code=$(curl -s -o /dev/null -w '%{http_code}' "$api/health")
    [ "$code" = 200 ] && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
done
check "/health 200" '[ "$code" = 200 ]'
models=$(curl -s "$api/v1/models")
check "/v1/models names npu-smoke on onnx" '[[ "$models" == *npu-smoke* && "$models" == *'"'"'"device":"onnx"'"'"'* ]]'
child=$(pgrep -P "$pid" | head -1)
check "ryzenai-server is serve's child (pid ${child:-none})" '[ -n "$child" ]'
body='{"model": "npu-smoke", "messages": [{"role": "user", "content": "What is the capital of France? One word."}],
       "temperature": 0, "max_tokens": 32}'
reply=$(curl -s -m 120 "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
content=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["choices"][0]["message"]["content"])' "$reply" 2>/dev/null)
echo "     said: ${content:-<no content: ${reply:0:300}>}"
check "answers Paris" '[[ "$content" == *Paris* ]]'
fds=$(ls -l "/proc/$child/fd" 2>/dev/null | grep -c '/dev/accel/accel0')
check "the child holds /dev/accel/accel0 ($fds descriptor(s))" '[ "${fds:-0}" -gt 0 ]'
body='{"model": "npu-smoke", "messages": [{"role": "user", "content": "Explain in detail how a bicycle works: frame, gears, brakes and tires."}],
       "temperature": 0, "max_tokens": 64}'
start=$(date +%s.%N)
reply=$(curl -s -m 120 "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
wall=$(echo "$(date +%s.%N) - $start" | bc)
tokens=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["usage"]["completion_tokens"])' "$reply" 2>/dev/null)
rate=$(python3 -c 'import sys; t=float(sys.argv[1]); w=float(sys.argv[2]); print(f"{t/w:.1f}" if w > 0 else "0")' "${tokens:-0}" "$wall")
echo "     ${tokens:-0} tokens in ${wall}s: $rate tok/s including prefill"
check "generated tokens (${tokens:-0})" '[ "${tokens:-0}" -ge 16 ]'

# stop: one SIGTERM to serve must take the child with it; wait up to 60 s, and if the child is
# still there, leave it (no SIGKILL on an NPU holder) and record who holds the device
kill -TERM "$pid"
gone=""
for i in $(seq 1 240); do
    if ! kill -0 "$child" 2>/dev/null; then gone=$i; break; fi
    sleep 0.25
done
if [ -n "$gone" ]; then took="$(awk -v n="$gone" 'BEGIN { printf "%.2f s", n / 4 }')"; else took="still alive after 60 s"; fi
check "child exited after SIGTERM to serve ($took)" '[ -n "$gone" ]'
if [ -z "$gone" ]; then
    echo "--- the child is still running (pid $child); left alone; holders of /dev/accel/accel0:"
    lsof /dev/accel/accel0 2>/dev/null || ls -l "/proc/$child/fd" 2>/dev/null | grep accel
fi
wait "$pid" 2>/dev/null
# serve escalates to SIGKILL after ONEBIT_BACKEND_STOP_GRACE_S; on an NPU holder that is the
# amdxdna risk, so an exit that needed it is a failure even though the child is gone
check "serve did not SIGKILL the child" '! grep -q "ignored SIGTERM" "$log"'

if [ $fail -ne 0 ]; then echo "--- serve log (tail)"; tail -30 "$log"; echo FAIL; exit 1; fi
echo PASS
