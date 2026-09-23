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
# End to end for `1bit serve` (docs/serve.md): serves one model on one device
# and checks the OpenAI-compatible API a host such as Lemonade relies on.
# /health goes 200, /v1/models names the model, a chat answers "Paris" under
# the served name, and a streamed chat arrives as SSE chunks.
#
# usage: tests/serve_e2e.sh path/to/1bit <model> <device> [extra serve args]
set -uo pipefail

bin=${1:?usage: serve_e2e.sh path/to/1bit <model> <device> [args]}
model=${2:?}
device=${3:?}
shift 3
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
log=$(mktemp)
"$bin" serve -m "$model" --device "$device" --port "$port" --alias e2e-model "$@" >"$log" 2>&1 &
pid=$!
cleanup() { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -f "$log"; }
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
api="http://127.0.0.1:$port"

code=000
for _ in $(seq 1 1200); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$api/health")
    [ "$code" = 200 ] && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
done
check "$device: /health 200" '[ "$code" = 200 ]'
models=$(curl -s "$api/v1/models")
check "$device: /v1/models names e2e-model" '[[ "$models" == *e2e-model* ]]'
body='{"model": "e2e-model", "messages": [{"role": "user", "content": "What is the capital of France? One word."}],
       "temperature": 0, "max_tokens": 48, "chat_template_kwargs": {"enable_thinking": false}}'
reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
content=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["choices"][0]["message"]["content"])' "$reply" 2>/dev/null)
name=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["model"])' "$reply" 2>/dev/null)
echo "     said: ${content:-<no content: ${reply:0:300}>}"
check "$device: answers Paris" '[[ "$content" == *Paris* ]]'
check "$device: reply names e2e-model" '[ "$name" = e2e-model ]'
chunks=$(curl -sN "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "e2e-model", "stream": true, "messages": [{"role": "user", "content": "Count to five."}], "max_tokens": 32}' \
    | grep -c '^data: {')
check "$device: streams ($chunks chunks)" '[ "$chunks" -gt 3 ]'

if [ $fail -ne 0 ]; then echo "--- serve log (tail)"; tail -30 "$log"; echo FAIL; exit 1; fi
echo PASS
