#!/bin/bash
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
# A client that hangs up mid-response must not take the server's accept loop with it (engine #1779, F10 fault 1).
#
# FastFlowLM 1.0.12's notes list "a server that stopped accepting connections after clients hung up" among their fixes.
# The fault is invisible to an ordinary test, because it needs a disconnect *during* a response: the listener is fine,
# the connection table is what leaks or the loop is what exits. So: a request is started against the streaming route,
# the client is killed after it has read a few bytes, and the loop repeats; then a fresh client must be served and the
# process must still be alive.
#
# No GPU, no model: tests/fake_backend.py stands in for the backend (app/serve.cpp only proxies to it), and the port is
# read back from serve's own line rather than picked and released here (tests/serve_port.sh, engine #1485).
#
# usage: tests/serve_hangup_test.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: serve_hangup_test.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=tests/serve_port.sh
. "$here/serve_port.sh"
scratch=$(mktemp -d)
touch "$scratch/tiny.gguf"
pids=()
cleanup() {
    for p in "${pids[@]}"; do pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done
    wait 2>/dev/null
    # serve keeps a state file naming the port it bound; drop it like tests/metrics_serve.sh does, but only if the
    # helper exists -- this test must not fail in cleanup for a reason unrelated to the fault it measures.
    if command -v statefile >/dev/null 2>&1; then statefile | xargs -r rm -f; fi
    rm -rf "$scratch"
}
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

"$bin" serve -m "$scratch/tiny.gguf" --device cpu --port 0 --alias hangup \
    --llama-server "$here/fake_backend.py" >"$scratch/serve.log" 2>&1 &
pid=$!
pids+=("$pid")
port=$(serve_port_wait "$scratch/serve.log" "$pid") || { echo "FAIL serve never reported a port"; cat "$scratch/serve.log"; exit 1; }
api="http://127.0.0.1:$port"

for _ in $(seq 1 100); do code=$(curl -s -o /dev/null -w '%{http_code}' "$api/health"); [ "$code" = 200 ] && break; sleep 0.1; done

body='{"model":"hangup","messages":[{"role":"user","content":"hello"}],"stream":true}'

# The hang-up: start a streaming request, let it read a little, kill the client. Five times, because a leak of one
# connection per hang-up survives a single iteration.
for i in $(seq 1 5); do
    timeout 0.4 curl -sN "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" >"$scratch/hang$i.txt" 2>/dev/null
done

# How many hang-ups actually reached the response stage: the client is killed by `timeout`, so a non-empty file means the
# server had begun writing when the socket went away -- which is the moment this fault lives in. Reported, not asserted:
# a run where none reached it proves less than it appears to, and the reader deserves to know which they got.
got=0; for i in $(seq 1 5); do [ -s "$scratch/hang$i.txt" ] && got=$((got + 1)); done
echo "note $got of 5 hang-ups had begun receiving a response before the client was killed"

check "serve is still running after 5 client hang-ups" "kill -0 $pid 2>/dev/null"
code=$(curl -s -o "$scratch/after.txt" -w '%{http_code}' --max-time 5 "$api/health")
check "health answers after the hang-ups (got $code)" "[ \"$code\" = 200 ]"
code=$(curl -s -o "$scratch/fresh.txt" -w '%{http_code}' --max-time 20 \
    "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"hangup","messages":[{"role":"user","content":"hello"}]}')
check "a fresh client is served after the hang-ups (got $code)" "[ \"$code\" = 200 ]"
check "and the fresh response carries a body" "[ -s \"$scratch/fresh.txt\" ]"

if [ $fail -ne 0 ]; then
    echo "--- serve log"; cat "$scratch/serve.log"
    echo "--- fresh response"; cat "$scratch/fresh.txt" 2>/dev/null
    echo FAIL; exit 1
fi
echo "all hang-up checks passed"
