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
# mlx_lemonade_e2e.sh <1bit> <lemon-mlx-engine server>    (macOS, Apple Silicon)
#
# `1bit lemonade` serves Qwen3-0.6B-MLX through the mlx backend: the model is
# listed, a greedy chat answers "Paris" under its Lemonade name, and streaming
# returns content chunks. docs/apple.md.
set -euo pipefail
bin=$1 server=$2
port=${PORT:-18400}
dir=$(mktemp -d)
export LEMONADE_MLX_SERVER=$server
"$bin" lemonade --port "$port" --no-broadcast --log-file disabled "$dir/cache" "$dir/config" > "$dir/lemon.log" 2>&1 &
lemon=$!
cleanup() { kill "$lemon" 2>/dev/null || true; sleep 1; rm -rf "$dir"; }
trap cleanup EXIT

for _ in $(seq 1 60); do curl -sf "localhost:$port/api/v1/health" > /dev/null && break; sleep 1; done

curl -s "localhost:$port/api/v1/models" | grep -q '"Qwen3-0.6B-MLX"' || { echo "FAIL: Qwen3-0.6B-MLX not listed"; exit 1; }

answer=$(curl -s -m 600 "localhost:$port/api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3-0.6B-MLX","messages":[{"role":"user","content":"What is the capital of France? Answer in one sentence. /no_think"}],"max_tokens":64,"temperature":0}')
echo "$answer" | grep -q 'Paris' || { echo "FAIL: chat did not answer Paris: $answer"; tail -20 "$dir/lemon.log"; exit 1; }
echo "$answer" | grep -q '"model":"Qwen3-0.6B-MLX"' || { echo "FAIL: response does not carry the Lemonade model name"; exit 1; }

chunks=$(curl -sN -m 120 "localhost:$port/api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3-0.6B-MLX","messages":[{"role":"user","content":"Count from 1 to 5. /no_think"}],"max_tokens":40,"temperature":0,"stream":true}' \
    | grep -c '"content"' || true)
[ "$chunks" -gt 5 ] || { echo "FAIL: streaming returned $chunks content chunks"; exit 1; }

echo "PASS: Qwen3-0.6B-MLX listed, chat answers Paris, streaming returned $chunks chunks"
