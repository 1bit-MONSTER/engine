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
# `1bit serve` notices a backend that stops working (docs/serve.md, "When a backend hangs"),
# without a GPU: tests/fake_backend.py stands in for llama-server and, asked for "fake_hang",
# stops sending bytes and stops answering /slots while its /health still says 200.
#   - /health proxies the backend's: a stopped (SIGSTOP) backend makes it 503, and 200 again after,
#   - a slow answer from a backend whose main loop still answers /slots is not a hang,
#   - a request that gets no bytes for --watchdog seconds kills the backend, and serve exits
#     non-zero with the reason in its log (Lemonade's BackendWatchdog then reloads it),
#   - before its first byte, a request gets prompt tokens / --watchdog-prompt-tps more.
#
# usage: tests/watchdog_serve.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: watchdog_serve.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
touch "$scratch/tiny.gguf"
pids=()
cleanup() {
    for p in "${pids[@]}"; do pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done
    wait 2>/dev/null
    rm -rf "$scratch"
}
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
newport() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
export FAKE_BACKEND_EVENTS="$scratch/events"

start() {  # <log> [serve args]: starts serve on $port, waits for /health 200
    port=$(newport)
    "$bin" serve -m "$scratch/tiny.gguf" --device cpu --port "$port" --alias wd \
        --llama-server "$here/fake_backend.py" "${@:2}" >"$1" 2>&1 &
    pid=$!
    pids+=("$pid")
    api="http://127.0.0.1:$port"
    for _ in $(seq 1 100); do [ "$(curl -s -o /dev/null -w '%{http_code}' "$api/health")" = 200 ] && break; sleep 0.1; done
    child=$(pgrep -P "$pid" | head -1)
}
gone_within() {  # <pid> <seconds>
    for _ in $(seq 1 $(($2 * 10))); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.1; done
    return 1
}

# 1. /health follows the backend; a slow answer is not a hang; a hang is.
start "$scratch/a.log" --watchdog 2 --watchdog-prompt-tps 0
check "/health 200 with a live backend" '[ "$(curl -s -o /dev/null -w "%{http_code}" "$api/health")" = 200 ]'
kill -STOP "$child"
code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$api/health")
kill -CONT "$child"
check "/health 503 while the backend does not answer ($code)" '[ "$code" = 503 ]'
code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$api/health")
check "/health 200 again once it answers ($code)" '[ "$code" = 200 ]'

reply=$(curl -s -m 20 "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"fake_delay": 5, "messages": [{"role": "user", "content": "hi"}]}')
check "a 5 s answer with --watchdog 2 is not a hang (its backend answers /slots)" '[[ "$reply" == *Paris* ]] && kill -0 "$pid"'

t0=$(date +%s)
curl -sN -m 30 -o "$scratch/hang.out" "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"stream": true, "fake_hang": true, "messages": [{"role": "user", "content": "hi"}]}' &
pids+=($!)
gone_within "$pid" 20; wait "$pid"; rc=$?
t1=$(date +%s)
check "a hung stream takes serve down (rc $rc, $((t1 - t0)) s)" '[ "$rc" != 0 ] && [ $((t1 - t0)) -le 15 ]'
check "  saying why" 'grep -q "watchdog: POST /v1/chat/completions (streamed) on the cpu backend" "$scratch/a.log" && grep -q "made no progress" "$scratch/a.log"'
check "  and the backend is gone" '! kill -0 "$child" 2>/dev/null'

# 2. Before its first byte, a long prompt has more time: 2 s + 2000 tokens / 10 tok/s.
prompt=$(python3 -c 'print("word " * 1600)')
start "$scratch/b.log" --watchdog 2 --watchdog-prompt-tps 10
curl -s -m 30 -o /dev/null "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"fake_hang\": true, \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}]}" &
pids+=($!)
sleep 6
check "a long prompt is not judged at --watchdog alone" 'kill -0 "$pid" 2>/dev/null && ! grep -q "watchdog:" "$scratch/b.log"'

# 3. --watchdog 0 turns it off.
start "$scratch/c.log" --watchdog 0
curl -sN -m 30 -o /dev/null "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"stream": true, "fake_hang": true, "messages": [{"role": "user", "content": "hi"}]}' &
pids+=($!)
sleep 4
check "--watchdog 0: no watchdog" 'kill -0 "$pid" 2>/dev/null && ! grep -q "watchdog:" "$scratch/c.log"'

if [ $fail -ne 0 ]; then for l in a b c; do echo "--- serve log $l"; cat "$scratch/$l.log"; done; echo FAIL; exit 1; fi
echo PASS
