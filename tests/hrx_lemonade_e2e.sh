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
# End to end on AMD hardware: `1bit lemonade` built with ONEBIT_HRX loads the
# same checkpoint through both recipes and generates. The llamacpp recipe must
# run it on Vulkan0 and the llamacpp-hrx recipe on HRX20, both through this
# build's llama-server. Scratch cache/config dirs; models come from the HF cache.
#
# usage: tests/hrx_lemonade_e2e.sh path/to/1bit path/to/llama-server
set -uo pipefail

bin=${1:?usage: hrx_lemonade_e2e.sh path/to/1bit path/to/llama-server}
server_bin=$(readlink -f "${2:?}")
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

for _ in $(seq 1 120); do curl -sf "$api/health" >/dev/null && break; sleep 0.5; done

# model recipe expected-device
for spec in "Qwen3-0.6B-GGUF llamacpp Vulkan0" "Qwen3-0.6B-HRX llamacpp-hrx HRX20"; do
    set -- $spec
    model=$1; recipe=$2; device=$3
    pull=$(curl -s -X POST "$api/pull" -H 'Content-Type: application/json' -d "{\"model_name\": \"$model\"}")
    check "$model: pulled" '[[ "$pull" == *success* ]]'
    body="{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France? One word.\"}],
           \"temperature\": 0, \"max_tokens\": 24, \"chat_template_kwargs\": {\"enable_thinking\": false}}"
    reply=$(curl -s -X POST "$api/chat/completions" -H 'Content-Type: application/json' -d "$body")
    content=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["choices"][0]["message"]["content"])' "$reply" 2>/dev/null)
    echo "     $model ($recipe) said: ${content:-<no content: ${reply:0:300}>}"
    check "$model: answers Paris" '[[ "$content" == *Paris* ]]'
    # Which binary and device did Lemonade launch for this recipe?
    launched=$(pgrep -a -f -- "--device $device" | grep -F "$server_bin" | head -1)
    check "$model: served by this build's llama-server on $device" '[[ -n "$launched" ]]'
    tps=$(curl -s "$api/stats" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("%.1f tok/s, ttft %.3f s" % (d.get("tokens_per_second",0), d.get("time_to_first_token",0)))' 2>/dev/null)
    echo "     telemetry: $tps"
    curl -s -X POST "$api/unload" -H 'Content-Type: application/json' -d "{\"model_name\": \"$model\"}" >/dev/null
done

if [ $fail -ne 0 ]; then echo "--- server log (tail)"; tail -40 "$scratch/server.log"; echo FAIL; exit 1; fi
echo PASS
