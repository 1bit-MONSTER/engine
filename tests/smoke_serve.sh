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
# model, SSE relay, that no backend outlives serve, and that a backend that exits on
# its own ends serve without a wait.
#
# usage: tests/smoke_serve.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: smoke_serve.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
touch "$scratch/tiny.gguf"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export FAKE_BACKEND_EVENTS="$scratch/events"
"$bin" serve -m "$scratch/tiny.gguf" --device cpu --port "$port" --alias smoke-model \
    --llama-server "$here/fake_backend.py" >"$scratch/serve.log" 2>&1 &
pid=$!
epid=
cleanup() {
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    [ -n "$epid" ] && kill -9 "$epid" 2>/dev/null
    [ -s "$scratch/impostor" ] && kill -9 "$(cat "$scratch/impostor")" 2>/dev/null
    rm -rf "$scratch"
}
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

# No authentication, so a web page must not drive it: a rebound hostname and a cross-site POST
# are refused; API clients (no Origin) and loopback pages are not.
code=$(curl -s -o /dev/null -w '%{http_code}' -H 'Host: rebind.example:80' "$api/v1/models")
check "a foreign Host is refused ($code)" '[ "$code" = 403 ]'
code=$(curl -s -o /dev/null -w '%{http_code}' "$api/v1/chat/completions" -H 'Content-Type: text/plain' \
    -H 'Origin: https://evil.example' -d '{"messages": [{"role": "user", "content": "hi"}]}')
check "a cross-site POST is refused ($code)" '[ "$code" = 403 ]'
code=$(curl -s -o /dev/null -w '%{http_code}' "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -H 'Origin: http://localhost:3000' -d '{"messages": [{"role": "user", "content": "hi"}]}')
check "a loopback page's POST is served ($code)" '[ "$code" = 200 ]'

# A client that hangs up on a non-streamed request: serve closes the backend connection too,
# so the backend can stop generating (the fake one notes which happened).
curl -s -m 1 -o /dev/null "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "smoke-model", "fake_delay": 5, "messages": [{"role": "user", "content": "hi"}]}'
for _ in $(seq 1 40); do [ -s "$scratch/events" ] && break; sleep 0.1; done
check "a client disconnect reaches the backend ($(cat "$scratch/events" 2>/dev/null))" \
    '[ "$(cat "$scratch/events" 2>/dev/null)" = cancelled ]'

child=$(pgrep -P "$pid" | head -1)
kill -9 "$pid"; wait "$pid" 2>/dev/null
sleep 1
if [ "$(uname)" = Linux ]; then
    check "no backend outlives serve (SIGKILL)" '! kill -0 "$child" 2>/dev/null'
fi

if [ "$(uname)" = Linux ]; then
    # A backend that exits on its own (a crash): serve reaps it where it noticed, says how it
    # ended, and its stop path neither waits the SIGTERM grace for the dead pid nor signals it
    # (the grace is set to 20 s here so a regression shows as a slow exit, not a hang).
    port3=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    start=$(date +%s)
    ONEBIT_BACKEND_STOP_GRACE_S=20 timeout 60 "$bin" serve -m "$scratch/tiny.gguf" --device cpu --port "$port3" \
        --alias smoke-model --llama-server "$here/fake_backend.py" >"$scratch/exit.log" 2>&1 &
    epid=$!
    for _ in $(seq 1 100); do [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port3/health")" = 200 ] && break; sleep 0.1; done
    curl -s -o /dev/null "http://127.0.0.1:$port3/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model": "smoke-model", "fake_exit": 3, "messages": [{"role": "user", "content": "hi"}]}'
    wait "$epid"; exit_rc=$?
    epid=
    took=$(( $(date +%s) - start ))
    check "a backend that exits on its own ends serve at once (rc $exit_rc, ${took}s)" \
        '[ "$exit_rc" = 1 ] && [ "$took" -lt 10 ] && grep -q "backend exited: exit 3" "$scratch/exit.log" && ! grep -q "ignored SIGTERM" "$scratch/exit.log"'
    [ $fail -ne 0 ] && { echo "--- exit log"; cat "$scratch/exit.log"; }
fi

both=$("$bin" serve -m "$scratch/tiny.gguf" --device cpu --mtp a.gguf --dflash b.gguf 2>&1); both_rc=$?
check "--mtp with --dflash is refused" '[ "$both_rc" != 0 ] && [[ "$both" == *"pick one"* ]]'

if [ "$(uname)" = Linux ]; then
    # Another process answers on the backend's port (the fake backend detaches one and does not
    # listen itself): serve must refuse it rather than route requests there.
    port2=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    FAKE_BACKEND_IMPOSTOR="$scratch/impostor" timeout 30 "$bin" serve -m "$scratch/tiny.gguf" --device cpu \
        --port "$port2" --llama-server "$here/fake_backend.py" >"$scratch/impostor.log" 2>&1
    imp_rc=$?
    check "a port held by another process is refused (rc $imp_rc)" \
        '[ "$imp_rc" = 1 ] && grep -q "not from the backend serve started" "$scratch/impostor.log"'
fi

if [ $fail -ne 0 ]; then echo "--- serve log"; cat "$scratch/serve.log"; echo FAIL; exit 1; fi
echo PASS
