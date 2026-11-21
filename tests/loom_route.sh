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
# The Loom GPU route of `1bit serve` (app/loom_route.h, docs/loom-route.md), end to end without a GPU:
# tests/fake_loom_serve.py stands in for loom-serve and tests/fake_backend.py for llama-server.
#   - route chosen: a model with an entry (by name, and by sha256 + size) is served by loom-serve with the
#     entry's card; the reply comes back streamed and not, under the served name, with timings;
#   - field normalisation: llama-server's hints never reach loom-serve, n_predict becomes max_tokens, a real
#     `stop` is refused by serve with a 400 that names it, /v1/completions answers 501;
#   - fallback: no entry (the built-in, empty table; an entry for another model), --no-loom, a missing card,
#     and a llama.cpp-route option (--mtp) all take the llama-server route, the first two without a word;
#   - readiness failure: a card loom-serve fails to prepare ends serve at once with loom-serve's reason;
#   - a foreign process on the port is refused; a port taken before loom-serve binds it is relaunched.
#
# usage: tests/loom_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: loom_route.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
. "$here/serve_port.sh"
scratch=$(mktemp -d)
pid=
cleanup() {
    [ -n "$pid" ] && { kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }
    [ -s "$scratch/impostor" ] && kill -9 "$(cat "$scratch/impostor")" 2>/dev/null
    rm -rf "$scratch"
}
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

python3 - "$scratch" <<'PY'
import struct, sys
def gguf(path, arch="llama"):
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv = [s("general.architecture") + struct.pack("<I", 8) + s(arch)]
    open(path, "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
d = sys.argv[1]
gguf(d + "/plain.gguf")
gguf(d + "/other.gguf", "qwen3")
PY
mkdir -p "$scratch/cards/q"
echo '{"schema_version": 1, "model": {"id": "Org/Plain-Loom"}}' >"$scratch/cards/q/model.json"
size=$(stat -c %s "$scratch/plain.gguf")
sha=$(sha256sum "$scratch/plain.gguf" | cut -d' ' -f1)
cat >"$scratch/by-name.json" <<EOF
{"routes": [{"id": "plain-by-name", "card": "q", "gate": "test", "preset": "chat", "match": {"model": ["plain"]},
             "args": ["--memory_bytes=1000"], "env": {"FAKE_LOOM_ROUTE_ENV": "1"}}]}
EOF
cat >"$scratch/by-sha.json" <<EOF
{"routes": [{"id": "plain-by-sha", "card": "$scratch/cards/q", "gate": "test", "preset": "chat",
             "match": {"gguf_sha256": "$sha", "gguf_size": $size}}]}
EOF
cat >"$scratch/no-card.json" <<EOF
{"routes": [{"id": "plain-gone", "card": "gone", "gate": "test", "preset": "chat", "match": {"model": ["plain"]}}]}
EOF

loom=(--loom-serve "$here/fake_loom_serve.py" --llama-server "$here/fake_backend.py" --loom-cards "$scratch/cards")
start() {  # <log> <args...>: serve in the background on port 0; sets pid and api
    local log=$1; shift
    "$bin" serve --port 0 "$@" >"$log" 2>&1 &
    pid=$!
    local port
    port=$(serve_port_wait "$log" "$pid" 300) || { echo "FAIL serve never became ready ($*)"; cat "$log"; fail=1; return 1; }
    api="http://127.0.0.1:$port"
}
stop() { [ -n "$pid" ] && { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }; pid=; }
chat() { curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"; }

# --- route chosen, by name ------------------------------------------------------------------------
export FAKE_LOOM_LAST="$scratch/last.json"
if start "$scratch/name.log" -m "$scratch/plain.gguf" --loom-routes "$scratch/by-name.json" "${loom[@]}"; then
    check "the entry is named in the log" 'grep -q "loom route plain-by-name: .*plain.gguf -> loom-serve card $scratch/cards/q/model.json (model Org/Plain-Loom; gate test)" "$scratch/name.log"'
    check "the backend is loom-serve" 'grep -q "plain on loom (.*fake_loom_serve.py)" "$scratch/name.log"'
    check "/v1/models names the device loom" '[[ "$(curl -s "$api/v1/models")" == *"\"device\":\"loom\""* ]]'
    check "/health is 200 (loom-serve /readyz)" '[ "$(curl -s -o /dev/null -w "%{http_code}" "$api/health")" = 200 ]'
    reply=$(chat '{"model": "plain", "messages": [{"role": "user", "content": "hi"}], "cache_prompt": true, "id_slot": 0, "n_predict": 16, "temperature": 0}')
    check "a chat reply comes through ($reply)" '[[ "$reply" == *Paris* ]]'
    check "under the served name" '[[ "$reply" == *"\"model\":\"plain\""* ]]'
    check "with timings from usage (prompt_n 10, cache_n 2, predicted_n 8)" \
        'python3 -c "import json,sys; t=json.loads(sys.argv[1])[\"timings\"]; sys.exit(not (t[\"prompt_n\"]==10 and t[\"cache_n\"]==2 and t[\"predicted_n\"]==8))" "$reply"'
    sent=$(cat "$scratch/last.json")
    check "loom-serve got the card's model name" '[[ "$sent" == *"\"model\": \"Org/Plain-Loom\""* ]]'
    check "and none of llama-server's hints" '[[ "$sent" != *cache_prompt* && "$sent" != *id_slot* && "$sent" != *n_predict* ]]'
    check "n_predict arrived as max_tokens" '[[ "$sent" == *"\"max_tokens\": 16"* ]]'
    check "and no top_k serve did not get from the client (no greedy shortcut on loom)" '[[ "$sent" != *top_k* ]]'
    stream=$(curl -sN "$api/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model": "plain", "stream": true, "messages": [{"role": "user", "content": "hi"}]}')
    check "a streamed reply comes through ($(grep -c '^data: {' <<<"$stream") chunks)" '[ "$(grep -c "^data: {" <<<"$stream")" = 9 ]'
    check "every chunk names the served model" '! grep -q "Org/Plain-Loom" <<<"$stream" && grep -q "\"model\":\"plain\"" <<<"$stream"'
    last=$(grep '^data: {' <<<"$stream" | tail -1 | cut -c7-)
    check "the final chunk carries finish_reason, usage and timings (llama-server's shape)" \
        'python3 -c "import json,sys; j=json.loads(sys.argv[1]); t=j[\"timings\"]; sys.exit(not (j[\"choices\"][0][\"finish_reason\"]==\"stop\" and j[\"usage\"][\"completion_tokens\"]==8 and t[\"prompt_ms\"]>0 and \"predicted_per_second\" in t))" "$last"'
    check "and the stream ends with [DONE]" '[ "$(grep -v "^$" <<<"$stream" | tail -n 1)" = "data: [DONE]" ]'
    sent=$(cat "$scratch/last.json")
    check "serve asked loom-serve for usage on the stream" '[[ "$sent" == *"\"include_usage\": true"* ]]'
    code=$(curl -s -o "$scratch/stop.json" -w '%{http_code}' "$api/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model": "plain", "messages": [{"role": "user", "content": "hi"}], "stop": ["\n"]}')
    check "a real stop list is refused by serve, by name ($code)" '[ "$code" = 400 ] && grep -q "\\\\\"stop\\\\\" is not supported on the Loom route" "$scratch/stop.json"'
    code=$(curl -s -o /dev/null -w '%{http_code}' "$api/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model": "plain", "messages": [{"role": "user", "content": "hi"}], "mirostat": 2}')
    check "a field loom-serve does not know gets loom-serve's own 400 ($code)" '[ "$code" = 400 ]'
    code=$(curl -s -o /dev/null -w '%{http_code}' "$api/v1/completions" -H 'Content-Type: application/json' -d '{"prompt": "hi"}')
    check "/v1/completions answers 501 on the Loom route ($code)" '[ "$code" = 501 ]'
    check "the entry's environment reached loom-serve" \
        'grep -qz "FAKE_LOOM_ROUTE_ENV=1" /proc/$(pgrep -P "$pid" -f fake_loom_serve.py | head -1)/environ'
    check "the entry's args reached loom-serve" 'pgrep -P "$pid" -af fake_loom_serve.py | grep -q -- "--memory_bytes=1000"'
    child=$(pgrep -P "$pid" -f fake_loom_serve.py | head -1)
    stop
    sleep 0.5
    check "loom-serve does not outlive serve" '! kill -0 "$child" 2>/dev/null'
fi

# --- route chosen, by sha256 + size (absolute card path) ----------------------------------------------
if start "$scratch/sha.log" -m "$scratch/plain.gguf" --loom-routes "$scratch/by-sha.json" "${loom[@]}"; then
    check "the checkpoint's sha256 selects its entry" 'grep -q "loom route plain-by-sha: .*sha256 $sha -> loom-serve" "$scratch/sha.log"'
    check "and serves" '[[ "$(chat "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}")" == *Paris* ]]'
    stop
fi

# --- fallback ---------------------------------------------------------------------------------------
if start "$scratch/builtin.log" -m "$scratch/plain.gguf" "${loom[@]}"; then
    check "the built-in (empty) table: llama-server, and not a word about loom" \
        'grep -q "plain on cpu\|plain on hrx" "$scratch/builtin.log" && ! grep -q "loom route\|on loom" "$scratch/builtin.log"'
    check "and the reply is llama-server's" '[[ "$(chat "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}")" == *fake_cache_prompt* ]]'
    stop
fi
if start "$scratch/other.log" -m "$scratch/other.gguf" --loom-routes "$scratch/by-name.json" "${loom[@]}"; then
    check "a model with no entry: llama-server, silently" \
        'grep -q "other on cpu\|other on hrx" "$scratch/other.log" && ! grep -q "loom route\|on loom" "$scratch/other.log"'
    stop
fi
if start "$scratch/noloom.log" -m "$scratch/plain.gguf" --loom-routes "$scratch/by-name.json" --no-loom "${loom[@]}"; then
    check "--no-loom keeps a matching model on llama-server" '! grep -q "on loom" "$scratch/noloom.log"'
    stop
fi
if start "$scratch/nocard.log" -m "$scratch/plain.gguf" --loom-routes "$scratch/no-card.json" "${loom[@]}"; then
    check "a missing card falls back, and says why" \
        'grep -q "loom route plain-gone matches .*, but its card is not usable (card path gone is not a model.json inside the cards directory): serving on the llama.cpp route" "$scratch/nocard.log" && ! grep -q "on loom" "$scratch/nocard.log"'
    stop
fi
touch "$scratch/head.gguf"
out=$("$bin" serve -m "$scratch/plain.gguf" --port 0 --device cpu --loom-routes "$scratch/by-name.json" "${loom[@]}" --mem-check 2>&1)
check "--device cpu never consults the table" '[[ "$out" != *"loom route"* ]]'
out=$("$bin" serve -m "$scratch/plain.gguf" --loom-routes "$scratch/by-name.json" "${loom[@]}" --mtp "$scratch/head.gguf" --print-backend 2>&1)
check "a llama.cpp-route option (--mtp) keeps the model there, and says so" \
    '[[ "$out" == *"but --mtp is a llama.cpp-route option: serving on the llama.cpp route"* ]]'
out=$("$bin" serve -m "$scratch/plain.gguf" --loom-routes "$scratch/by-name.json" "${loom[@]}" --print-backend 2>&1)
check "--print-backend on the Loom route says serve must stay in the path" '[[ "$out" == *"loom is not a llama-server"* ]]'
echo '{"routes": [{"id": "x", "card": "q", "match": {"model": ["plain"]}}]}' >"$scratch/bad.json"
out=$("$bin" serve -m "$scratch/plain.gguf" --port 0 --loom-routes "$scratch/bad.json" "${loom[@]}" 2>&1); rc=$?
check "a malformed table stops serve with the reason (rc $rc)" '[ "$rc" != 0 ] && [[ "$out" == *"\"gate\" is required"* ]]'

# --- readiness failure --------------------------------------------------------------------------------
t0=$(date +%s)
FAKE_LOOM_FAIL=1 timeout 60 "$bin" serve -m "$scratch/plain.gguf" --port 0 --loom-routes "$scratch/by-name.json" "${loom[@]}" \
    >"$scratch/fail.log" 2>&1; rc=$?
took=$(( $(date +%s) - t0 ))
check "a card loom-serve fails to prepare ends serve at once (rc $rc, ${took}s), with loom-serve's reason" \
    '[ "$rc" = 1 ] && [ "$took" -lt 20 ] && grep -q "loom-serve: Org/Plain-Loom failed: .*PARSE/003" "$scratch/fail.log" && grep -q "did not become ready" "$scratch/fail.log"'

# --- the port -------------------------------------------------------------------------------------------
if [ "$(uname)" = Linux ]; then
    ONEBIT_FOREIGN_PORT_GRACE_S=1 FAKE_LOOM_IMPOSTOR="$scratch/impostor" timeout 30 "$bin" serve -m "$scratch/plain.gguf" \
        --port 0 --loom-routes "$scratch/by-name.json" "${loom[@]}" >"$scratch/impostor.log" 2>&1; rc=$?
    check "a port held by another process is refused (rc $rc)" \
        '[ "$rc" = 1 ] && grep -q "not from the backend serve started" "$scratch/impostor.log"'
    if [ -s "$scratch/impostor" ]; then
        imp=$(cat "$scratch/impostor")
        for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$imp" 2>/dev/null || break; sleep 0.2; done
        check "the impostor does not outlive the case" '! kill -0 "$imp" 2>/dev/null'
    fi
    export FAKE_LOOM_PORT_TAKEN_ONCE="$scratch/taken"
    if start "$scratch/taken.log" -m "$scratch/plain.gguf" --loom-routes "$scratch/by-name.json" "${loom[@]}"; then
        check "a port taken before loom-serve bound it is relaunched on a fresh --listen" \
            'grep -q "could not bind port .*relaunching on port" "$scratch/taken.log" && [[ "$(chat "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}")" == *Paris* ]]'
        stop
    fi
    unset FAKE_LOOM_PORT_TAKEN_ONCE
fi

if [ $fail -ne 0 ]; then
    for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done
    echo FAIL
    exit 1
fi
echo PASS
