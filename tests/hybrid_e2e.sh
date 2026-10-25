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
# `1bit serve --device hybrid` (docs/hybrid.md) end to end on a real model, with no NPU: the decoding
# llama-server is the CPU build (or the HRX one, where the build has it), and the prefill provider is
# tests/fake_prefill_provider.py, which makes the state file by having that same server prefill the ids
# into slot 0 and save the slot. So the whole path runs: template + tokenize, the provider, the
# restore, the request pinned to the slot. Checked: the greedy reply equals the plain --device cpu
# reply for the same request, and the backend's timings show it prefilled one token (the last one,
# which llama-server always re-evaluates) rather than the prompt.
#
# usage: tests/hybrid_e2e.sh path/to/1bit <model.gguf> [llama-server]
# Skips (exit 77) without a model or a llama-server.
set -uo pipefail

bin=${1:?usage: hybrid_e2e.sh path/to/1bit <model.gguf> [llama-server]}
model=${2:-}
server=${3:-}
here=$(cd "$(dirname "$0")" && pwd)
[ -n "$model" ] && [ -f "$model" ] || { echo "skip: no model"; exit 77; }
if [ -z "$server" ]; then
    "$bin" serve -m "$model" --device cpu --print-backend --no-mem-check >/dev/null 2>&1 || { echo "skip: no llama-server in this build"; exit 77; }
    server_arg=()
else
    [ -x "$server" ] || { echo "skip: no llama-server at $server"; exit 77; }
    server_arg=(--llama-server "$server")
fi
scratch=$(mktemp -d)
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
pid=
cleanup() { [ -n "$pid" ] && { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }; rm -rf "$scratch"; }
trap cleanup EXIT

start() {   # start <device> <log> [env...]
    local device=$1 log=$2; shift 2
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    env "$@" "$bin" serve -m "$model" --device "$device" --port "$port" --alias e2e-model --ctx-size 2048 \
        --no-mem-check "${server_arg[@]}" >"$log" 2>&1 &
    pid=$!
    api="http://127.0.0.1:$port"
    local code=000
    for _ in $(seq 1 1200); do
        code=$(curl -s -o /dev/null -w '%{http_code}' "$api/health"); [ "$code" = 200 ] && break
        kill -0 "$pid" 2>/dev/null || break; sleep 0.5
    done
    [ "$code" = 200 ]
}
stop() { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; pid=; }
body='{"model": "e2e-model", "messages": [{"role": "user", "content": "Name three colours of the rainbow, then stop."}],
       "temperature": 0, "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": false}}'
content() { python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'; }
field() { python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("timings", {}).get(sys.argv[1], ""))' "$1"; }

check "cpu: serve starts" 'start cpu "$scratch/cpu.log"'
plain=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
plain_text=$(echo "$plain" | content)
plain_prompt_n=$(echo "$plain" | field prompt_n)
echo "     cpu said: $plain_text (prompt_n $plain_prompt_n)"
check "cpu: answered" '[ -n "$plain_text" ]'
stop

check "hybrid: serve starts with the fake provider" 'start hybrid "$scratch/hybrid.log" ONEBIT_HYBRID_PREFILL="$here/fake_prefill_provider.py"'
reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
text=$(echo "$reply" | content)
prompt_n=$(echo "$reply" | field prompt_n)
echo "     hybrid said: $text (prompt_n $prompt_n)"
check "hybrid: the greedy reply equals the plain one" '[ "$text" = "$plain_text" ]'
check "hybrid: the backend prefilled only the last token (prompt_n $prompt_n, plain $plain_prompt_n)" '[ "$prompt_n" = 1 ] && [ "$plain_prompt_n" -gt 1 ]'
sleep 0.3
check "hybrid: stderr has the per-request line" 'grep -q "hybrid: [0-9]* prompt tokens: prefill .* ms, restore .* ms, first byte .* ms" "$scratch/hybrid.log"'
check "hybrid: no fallback happened" '! grep -q "the backend prefills this request itself" "$scratch/hybrid.log"'
# a second turn: the handoff again, same answer
reply2=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
check "hybrid: the second request answers the same" '[ "$(echo "$reply2" | content)" = "$plain_text" ]'
chunks=$(curl -sN "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "${body%\}}, \"stream\": true}" | grep -c '^data: {')
check "hybrid: streams ($chunks chunks)" '[ "$chunks" -gt 3 ]'
stop

if [ $fail -ne 0 ]; then echo "--- hybrid log (tail)"; tail -40 "$scratch/hybrid.log"; echo FAIL; exit 1; fi
echo PASS
