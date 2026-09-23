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
# End to end: `1bit lemonade` built with ONEBIT_ZINC pulls Qwen3-0.6B-ZINC (a
# GGUF), loads it through the zinc recipe with this build's zinc, and answers
# over the Lemonade API, both non-streaming and streaming. Scratch cache/config
# dirs; the GGUF comes from the HF cache.
#
# usage: tests/zinc_lemonade_e2e.sh path/to/1bit path/to/zinc
set -uo pipefail

bin=${1:?usage: zinc_lemonade_e2e.sh path/to/1bit path/to/zinc}
zinc_bin=$(readlink -f "${2:?}")
scratch=$(mktemp -d)
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$bin" lemonade --port "$port" --no-broadcast --log-file disabled "$scratch/cache" "$scratch/config" \
    >"$scratch/server.log" 2>&1 &
pid=$!
cleanup() { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -rf "$scratch"; }
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
api="http://127.0.0.1:$port/api/v1"
model=Qwen3-0.6B-ZINC

for _ in $(seq 1 120); do curl -sf "$api/health" >/dev/null && break; sleep 0.5; done

pull=$(curl -s -X POST "$api/pull" -H 'Content-Type: application/json' -d "{\"model_name\": \"$model\"}")
check "$model: pulled" '[[ "$pull" == *success* ]]'
body="{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France? One word.\"}],
       \"temperature\": 0, \"max_tokens\": 24}"
reply=$(curl -s -X POST "$api/chat/completions" -H 'Content-Type: application/json' -d "$body")
content=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["choices"][0]["message"]["content"])' "$reply" 2>/dev/null)
name=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["model"])' "$reply" 2>/dev/null)
echo "     $model said: ${content:-<no content: ${reply:0:300}>}"
check "$model: answers Paris" '[[ "$content" == *Paris* ]]'
check "$model: reply carries the Lemonade name" '[[ "$name" == "$model" ]]'
launched=$(pgrep -a -f -- "-m .*-p " | grep -F "$zinc_bin" | head -1)
check "$model: served by this build's zinc" '[[ -n "$launched" ]]'
chunks=$(curl -sN -X POST "$api/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$model\", \"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"Count to five.\"}], \"max_tokens\": 32}" \
    | grep -c '^data: {')
check "$model: streams ($chunks chunks)" '[ "$chunks" -gt 3 ]'
tps=$(curl -s "$api/stats" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("%.1f tok/s, ttft %.3f s" % (d.get("tokens_per_second",0), d.get("time_to_first_token",0)))' 2>/dev/null)
echo "     telemetry: $tps"
curl -s -X POST "$api/unload" -H 'Content-Type: application/json' -d "{\"model_name\": \"$model\"}" >/dev/null

if [ $fail -ne 0 ]; then echo "--- server log (tail)"; tail -40 "$scratch/server.log"; echo FAIL; exit 1; fi
echo PASS
