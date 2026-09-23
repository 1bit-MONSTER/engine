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
# Smoke test for `1bit serve`, runnable anywhere (CI included): the proxy in
# front of a child backend, with tests/fake_backend.py standing in for
# llama-server. Checks /health, /v1/models, a chat reply renamed to the served
# model, SSE relay, and that no backend outlives serve.
#
# usage: tests/smoke_serve.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: smoke_serve.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
touch "$scratch/tiny.gguf"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$bin" serve -m "$scratch/tiny.gguf" --device vulkan --port "$port" --alias smoke-model \
    --llama-server "$here/fake_backend.py" >"$scratch/serve.log" 2>&1 &
pid=$!
cleanup() { kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -rf "$scratch"; }
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
api="http://127.0.0.1:$port"

for _ in $(seq 1 100); do curl -s -o /dev/null "$api/health" && break; sleep 0.1; done
first=$(curl -s -o /dev/null -w '%{http_code}' "$api/health")
check "/health answers while loading ($first)" '[ "$first" = 503 ] || [ "$first" = 200 ]'
code=000
for _ in $(seq 1 100); do code=$(curl -s -o /dev/null -w '%{http_code}' "$api/health"); [ "$code" = 200 ] && break; sleep 0.1; done
check "/health 200 once the backend is up" '[ "$code" = 200 ]'
check "/v1/models names smoke-model" '[[ "$(curl -s "$api/v1/models")" == *smoke-model* ]]'
reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "smoke-model", "messages": [{"role": "user", "content": "hi"}]}')
check "chat reply comes through" '[[ "$reply" == *Paris* ]]'
check "reply renamed to the served model" '[[ "$reply" == *"\"model\":\"smoke-model\""* ]]'
chunks=$(curl -sN "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "smoke-model", "stream": true, "messages": [{"role": "user", "content": "hi"}]}' | grep -c '^data: {')
check "SSE relayed ($chunks chunks)" '[ "$chunks" = 7 ]'

child=$(pgrep -P "$pid" | head -1)
kill -9 "$pid"; wait "$pid" 2>/dev/null
sleep 1
if [ "$(uname)" = Linux ]; then
    check "no backend outlives serve (SIGKILL)" '! kill -0 "$child" 2>/dev/null'
fi

if [ $fail -ne 0 ]; then echo "--- serve log"; cat "$scratch/serve.log"; echo FAIL; exit 1; fi
echo PASS
