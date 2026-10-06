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
# `1bit serve --pm` (docs/pm.md) without a GPU: tests/fake_pm_backend.py plays the Project Manager's
# llama-server (a delegate(code) call on the first round, the final answer on the second) and
# tests/fake_lemonade.py plays Lemonade, recording the expert id it is asked for. Checks:
#   - a coding question delegates to the default code expert's catalog id, with its max_tokens, and
#     the expert's answer reaches the client under the served model's name (pm.delegations says so),
#   - the same when the PM writes the call as ZAYA's <zyphra_tool_call> text (context parsed too),
#   - a general question is answered by the PM alone (no request to Lemonade),
#   - stream=true gets the final answer as SSE chunks ending in [DONE],
#   - a Lemonade that is not running: the tool result says so and the PM still answers (HTTP 200),
#     with one delegation line in the log,
#   - an expert that errors: the reply says HTTP 500 and why,
#   - pm.max_rounds 1 offers no tools, so the PM answers at once,
#   - /v1/completions forwards unchanged,
#   - --pm with --laya, and a bad experts file, are refused.
#
# usage: tests/pm_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: pm_route.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
touch "$scratch/zaya.gguf"
pids=()
cleanup() { for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -rf "$scratch"; }
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
freeport() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
jq() { python3 -c 'import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))' "$1"; }  # jq EXPR over stdin JSON as d

lem_port=$(freeport)
python3 "$here/fake_lemonade.py" "$lem_port" "$scratch/lemonade.jsonl" &
pids+=($!)
lemonade="http://127.0.0.1:$lem_port"

# serve NAME [serve args...]: a `1bit serve --pm` on the fake PM backend, ready, api in $api
serve() {
    local name=$1; shift
    local port; port=$(freeport)
    PM_RECORD="$scratch/$name.backend.jsonl" "$bin" serve -m "$scratch/zaya.gguf" --device cpu --port "$port" --alias ZAYA1-8B-PM \
        --llama-server "$here/fake_pm_backend.py" --pm "$@" >"$scratch/$name.log" 2>&1 &
    pids+=($!)
    api="http://127.0.0.1:$port"
    for _ in $(seq 1 100); do [ "$(curl -s -o /dev/null -w '%{http_code}' "$api/health")" = 200 ] && return 0; sleep 0.1; done
    echo "serve $name did not become ready"; cat "$scratch/$name.log"; return 1
}
ask() {  # ask [extra json fields] -- a coding question, non-stream
    curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\": \"ZAYA1-8B-PM\", \"messages\": [{\"role\": \"user\", \"content\": \"write a C function that reverses a UTF-8 string in place, with tests\"}]${1:-}}"
}
lem_records() { [ -f "$scratch/lemonade.jsonl" ] && wc -l <"$scratch/lemonade.jsonl" || echo 0; }

# 1. the default experts file, llama-server's parsed tool_calls form
serve default --lemonade-url "$lemonade" || fail=1
check "ready line names the project manager and its experts" 'grep -q "pm: ZAYA1-8B-PM is the project manager; experts through $lemonade: code=Qwen2.5-Coder-7B-Instruct-1bit" "$scratch/default.log"'
reply=$(ask)
check "coding question: Lemonade was asked for the default code expert" '[ "$(lem_records)" = 1 ] && [ "$(tail -1 "$scratch/lemonade.jsonl" | jq "d[\"model\"]")" = Qwen2.5-Coder-7B-Instruct-1bit ]'
check "  with the expert's max_tokens from the experts file" '[ "$(tail -1 "$scratch/lemonade.jsonl" | jq "d[\"max_tokens\"]")" = 4096 ]'
check "  the task reached the expert" '[[ "$(tail -1 "$scratch/lemonade.jsonl" | jq "d[\"user\"]")" == *"reverses a UTF-8 string"* ]]'
check "  the expert's answer reaches the client" '[[ "$(echo "$reply" | jq "d[\"choices\"][0][\"message\"][\"content\"]")" == *EXPERT_ANSWER* ]]'
check "  under the served model's name" '[ "$(echo "$reply" | jq "d[\"model\"]")" = ZAYA1-8B-PM ]'
check "  pm.delegations records the expert and success" '[ "$(echo "$reply" | jq "d[\"pm\"][\"delegations\"][0][\"id\"], d[\"pm\"][\"delegations\"][0][\"ok\"], d[\"pm\"][\"rounds\"]")" = "('"'"'Qwen2.5-Coder-7B-Instruct-1bit'"'"', True, 2)" ]'
check "  the PM saw the tool result as a tool message" 'grep -q "\"role\": \"tool\"" "$scratch/default.backend.jsonl"'
check "  one delegation line in the log" '[ "$(grep -c "pm: delegate code -> Qwen2.5-Coder-7B-Instruct-1bit" "$scratch/default.log")" = 1 ] && grep -q "ms ok" "$scratch/default.log"'

# 2. a general question: no delegation
before=$(lem_records)
reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "ZAYA1-8B-PM", "messages": [{"role": "user", "content": "hello, how are you today?"}]}')
check "general question: answered by the PM, nothing asked of Lemonade" '[ "$(lem_records)" = "$before" ] && [[ "$(echo "$reply" | jq "d[\"choices\"][0][\"message\"][\"content\"]")" == "Hi there"* ]]'
check "  pm.rounds 1, no delegations" '[ "$(echo "$reply" | jq "d[\"pm\"][\"rounds\"], len(d[\"pm\"][\"delegations\"])")" = "(1, 0)" ]'

# 3. streaming: the final answer as SSE
sse=$(ask ', "stream": true')
check "stream=true: SSE chunks, role first, [DONE] last" '[[ "$(echo "$sse" | grep "^data: " | head -1)" == *"\"role\": \"assistant\""* || "$(echo "$sse" | grep "^data: " | head -1)" == *"\"role\":\"assistant\""* ]] && [ "$(echo "$sse" | grep "^data: " | tail -1)" = "data: [DONE]" ]'
assembled=$(echo "$sse" | grep '^data: {' | python3 -c 'import json,sys; print("".join(json.loads(l[6:])["choices"][0]["delta"].get("content","") for l in sys.stdin))')
check "  the chunks assemble to the expert's answer" '[[ "$assembled" == *EXPERT_ANSWER* ]]'
check "  the stream delegated once more" '[ "$(lem_records)" = $((before + 1)) ]'

# 4. /v1/completions forwards unchanged
plain=$(curl -s "$api/v1/completions" -H 'Content-Type: application/json' -d '{"model": "ZAYA1-8B-PM", "prompt": "write code"}')
check "/v1/completions forwards unchanged" '[[ "$plain" == *"plain completion"* ]]'

# 5. ZAYA's own tool-call text
PM_CALL_FORM=zyphra serve zyphra --lemonade-url "$lemonade" || fail=1
before=$(lem_records)
reply=$(ask)
check "ZAYA's <zyphra_tool_call> text is parsed into a delegation" '[ "$(lem_records)" = $((before + 1)) ] && [ "$(tail -1 "$scratch/lemonade.jsonl" | jq "d[\"model\"]")" = Qwen2.5-Coder-7B-Instruct-1bit ]'
check "  its <parameter=context> reached the expert" '[[ "$(tail -1 "$scratch/lemonade.jsonl" | jq "d[\"user\"]")" == *"C99, no libc beyond string.h"* ]]'
check "  and the answer came back" '[[ "$(echo "$reply" | jq "d[\"choices\"][0][\"message\"][\"content\"]")" == *EXPERT_ANSWER* ]]'

# 6. Lemonade not running
dead_port=$(freeport)
serve dead --lemonade-url "http://127.0.0.1:$dead_port" || fail=1
code=$(curl -s -o "$scratch/dead.json" -w '%{http_code}' "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "ZAYA1-8B-PM", "messages": [{"role": "user", "content": "write a C function that reverses a string"}]}')
check "Lemonade unreachable: HTTP 200, the PM answers" '[ "$code" = 200 ]'
check "  and the reply says the delegation failed" '[[ "$(jq "d[\"choices\"][0][\"message\"][\"content\"]" <"$scratch/dead.json")" == *"did not answer"* ]]'
check "  pm.delegations says ok false" '[ "$(jq "d[\"pm\"][\"delegations\"][0][\"ok\"]" <"$scratch/dead.json")" = False ]'
check "  logged as failed" 'grep -q "pm: delegate code -> Qwen2.5-Coder-7B-Instruct-1bit .* failed: Lemonade at" "$scratch/dead.log"'

# 7. an expert that errors, from a custom experts file with pm.max_rounds 1 checked after
cat >"$scratch/broken.json" <<'JSON'
{"experts": [{"expert": "code", "id": "broken-expert", "domain": "programming", "use_when": "code"}]}
JSON
serve broken --lemonade-url "$lemonade" --pm-experts "$scratch/broken.json" || fail=1
reply=$(ask)
check "an expert that errors: the reply says HTTP 500 and why" '[[ "$(echo "$reply" | jq "d[\"choices\"][0][\"message\"][\"content\"]")" == *"HTTP 500"*"fell over"* ]]'

cat >"$scratch/one.json" <<'JSON'
{"pm": {"max_rounds": 1}, "experts": [{"expert": "code", "id": "Qwen2.5-Coder-7B-Instruct-1bit", "domain": "programming", "use_when": "code"}]}
JSON
serve one --lemonade-url "$lemonade" --pm-experts "$scratch/one.json" || fail=1
before=$(lem_records)
reply=$(ask)
check "pm.max_rounds 1: no tools offered, the PM answers at once" '[ "$(lem_records)" = "$before" ] && [[ "$(echo "$reply" | jq "d[\"choices\"][0][\"message\"][\"content\"]")" == "Hi there"* ]]'
check "  the backend saw no tools" '! grep -q "\"tools\"" "$scratch/one.backend.jsonl"'

# 8. refusals
out=$("$bin" serve -m "$scratch/zaya.gguf" --device cpu --pm --laya --llama-server "$here/fake_pm_backend.py" 2>&1); rc=$?
check "--pm with --laya is refused" '[ "$rc" != 0 ] && [[ "$out" == *"--pm does not combine with --laya"* ]]'
echo '{"experts": [{"expert": "code"}]}' >"$scratch/bad.json"
out=$("$bin" serve -m "$scratch/zaya.gguf" --device cpu --pm --pm-experts "$scratch/bad.json" --llama-server "$here/fake_pm_backend.py" 2>&1); rc=$?
check "an experts file without ids is refused" '[ "$rc" != 0 ] && [[ "$out" == *"needs \"expert\" and \"id\""* ]]'

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
