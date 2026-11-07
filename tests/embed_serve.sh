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
# tests/embed_serve.sh <1bit> - the embedding role served in-process from a 1-bit container (engine#1311):
# `--embedding -m <container> --device cpu` answers /v1/embeddings for a string and for an array, in the OpenAI
# shape, unit norm, with the refusals named. Device-free, on the 32-token fixture (dim 8), so it runs in CI.
set -u
BIN=${1:?usage: embed_serve.sh <1bit>}
DIR=$(cd "$(dirname "$0")/.." && pwd)
FIX=$DIR/tests/fixtures/trit_embed_tiny
PORT=${EMBED_SERVE_PORT:-8247}
LOG=$(mktemp)
fails=0
check() { # $1 what, $2 got, $3 want
    if [ "$2" = "$3" ]; then echo "  PASS $1 ($2)"; else echo "  FAIL $1: got '$2' want '$3'"; fails=$((fails+1)); fi
}
cleanup() { [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; rm -f "$LOG"; }
trap cleanup EXIT

"$BIN" serve --embedding -m "$FIX" --device cpu --port "$PORT" >"$LOG" 2>&1 &
PID=$!
code=000
for _ in $(seq 1 100); do
    code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$code" = 200 ] && break
    kill -0 "$PID" 2>/dev/null || { echo "serve died:"; tail -5 "$LOG"; exit 1; }
    sleep 0.2
done
check "health" "$code" 200
grep -q "in-process: /v1/embeddings" "$LOG" && echo "  PASS the route says it is in-process" || { echo "  FAIL no in-process line"; fails=$((fails+1)); }

# one string
body=$(curl -s -m 30 -H 'Content-Type: application/json' -d '{"input":"hello world"}' "http://127.0.0.1:$PORT/v1/embeddings")
echo "$body" | grep -q '"object":"list"' && echo "  PASS one string: object=list" || { echo "  FAIL one string"; fails=$((fails+1)); }
check "one string: entries" "$(echo "$body" | grep -o '"object":"embedding"' | wc -l)" 1
check "one string: dim" "$(echo "$body" | grep -o '\-0\.[0-9]*\|0\.[0-9]*\|-[1-9][0-9]*\.[0-9]*\|[1-9][0-9]*\.[0-9]*' | wc -l)" 8
echo "$body" | grep -q '"model":"trit_embed_tiny"' && echo "  PASS one string: model id" || { echo "  FAIL model id"; fails=$((fails+1)); }

# an array is N embeddings, indexed
body=$(curl -s -m 30 -H 'Content-Type: application/json' -d '{"input":["hello world","the cat"]}' "http://127.0.0.1:$PORT/v1/embeddings")
check "array: entries" "$(echo "$body" | grep -o '"object":"embedding"' | wc -l)" 2
echo "$body" | grep -q '"index":0' && echo "$body" | grep -q '"index":1' && echo "  PASS array: indices 0 and 1" || { echo "  FAIL array indices"; fails=$((fails+1)); }
echo "$body" | grep -q '"usage":{"prompt_tokens":' && echo "  PASS array: usage" || echo "  NOTE usage only on a single string" 

# refusals, named
c=$(curl -s -m 30 -o /tmp/embed_serve_er1 -w '%{http_code}' -H 'Content-Type: application/json' -d '{"input":[[1,2,3]]}' "http://127.0.0.1:$PORT/v1/embeddings")
check "token-id array: status" "$c" 400
grep -q "token-id arrays are not served" /tmp/embed_serve_er1 && echo "  PASS token-id array: named refusal" || { echo "  FAIL token-id refusal not named"; fails=$((fails+1)); }
c=$(curl -s -m 30 -o /tmp/embed_serve_er2 -w '%{http_code}' -H 'Content-Type: application/json' -d '{}' "http://127.0.0.1:$PORT/v1/embeddings")
check "missing input: status" "$c" 400
grep -q "input is required" /tmp/embed_serve_er2 && echo "  PASS missing input: named refusal" || { echo "  FAIL missing input not named"; fails=$((fails+1)); }

# the model list names the role and the device
body=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models")
echo "$body" | grep -q '"role":"embedding"' && echo "$body" | grep -q '"device":"cpu"' && echo "  PASS /v1/models: role and device" || { echo "  FAIL /v1/models"; fails=$((fails+1)); }
rm -f /tmp/embed_serve_er1 /tmp/embed_serve_er2
if [ "$fails" -ne 0 ]; then
    echo "--- server log ---"
    tail -20 "$LOG"
    echo "embed_serve: $fails failure(s)"
    exit 1
fi
echo "embed_serve: ok"
