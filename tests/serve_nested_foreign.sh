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
# serve_nested_foreign.sh: a nested `1bit serve` is not mistaken for a foreign backend (engine#3171).
#
# `--laya-models` is the only rail that launches its child with `--port 0` (app/serve.cpp's `b.port = 0`), so it is
# the only rail where serve's `port() <= 0` branch runs while a child is still loading -- and that branch is where
# the foreign-format refusal lives. Its child is always a nested `1bit serve`, whose own child is llama-server:
# llama-server prints "listening on http://host:port" at bind, the nested serve tees it on, and the OUTER serve reads
# its grandchild's line. A guard that decides by line content alone refuses a launch that was working.
#
# What makes this runnable without a GPU: the nested child is a fake llama-server (tests/fake_backend_route.py) that
# announces its port the way the real one does (tests/fake_llama_announce.sh). The front, the nested serve, the port
# read-back and the routing are the shipped code. Needs a Laya classifier checkpoint, like the three laya_* rows.
#
# usage: tests/serve_nested_foreign.sh path/to/1bit path/to/laya-checkpoint-dir
set -uo pipefail

bin=${1:?usage: serve_nested_foreign.sh path/to/1bit path/to/laya-checkpoint-dir}
laya_model=${2:-}
here=$(cd "$(dirname "$0")" && pwd)
if [ -z "$laya_model" ] || [ ! -f "$laya_model/model.safetensors" ]; then
    echo "skip: no Laya classifier checkpoint at '${laya_model}' (configure with -DONEBIT_LAYA_ROUTE_MODEL=<dir>)"
    exit 77
fi
# shellcheck source=tests/serve_port.sh
. "$here/serve_port.sh"     # serve_port_wait: read back the port serve bound instead of guessing one (engine#1485)
scratch=$(mktemp -d)
touch "$scratch/plan.gguf" "$scratch/reflex.gguf"
# Two classes (a table needs two) and min_confidence 1.01: every request takes the default row, so where a request
# lands is known and the only thing under test is whether the launch survived the grandchild's readiness line.
cat >"$scratch/table.json" <<EOF
{"question": {"instructions": "Where does this request go?",
              "classes": {"plan": "plan a build", "reflex": "one quick action"}},
 "default": "plan", "min_confidence": 1.01,
 "classes": {"plan": {"model": "$scratch/plan.gguf", "device": "cpu", "alias": "nested-row",
                      "args": ["--llama-server", "$here/fake_llama_announce.sh"]},
             "reflex": {"model": "$scratch/reflex.gguf", "device": "cpu", "alias": "other-row",
                        "args": ["--llama-server", "$here/fake_llama_announce.sh"]}}}
EOF
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

# 3 s between the fake llama-server's bind line and its first answer: the parent polls every 250 ms, so a child that
# answers instantly would never let the guard be seen at all (that is why this looked healthy on main).
export FAKE_LLAMA_LOAD_S=3
"$bin" serve --laya-models "$scratch/table.json" --laya-model "$laya_model" --port 0 >"$scratch/serve.log" 2>&1 &
pid=$!
# the trap goes in before the wait: a live serve left behind on this box holds ports and its backend
cleanup() { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -rf "$scratch"; }
trap cleanup EXIT
# Match the front's line: it is the only one carrying the laya-models suffix, so a nested child's own readiness line
# cannot be mistaken for it (the same anchor tests/laya_models_e2e.sh uses).
if ! port=$(serve_port_wait "$scratch/serve.log" "$pid" 1200 '(laya-models: model auto is routed per request)'); then
    echo "FAIL the launch was refused while its child was loading"
    sed -n '1,20p' "$scratch/serve.log"
    exit 1
fi
api="http://127.0.0.1:$port"
for _ in $(seq 1 1200); do [ "$(curl -s -o /dev/null -w '%{http_code}' "$api/health")" = 200 ] && break; sleep 0.1; done
check "ready with a nested child that announces its port" '[ "$(curl -s -o /dev/null -w "%{http_code}" "$api/health")" = 200 ]'

check "the grandchild's line is in this log (the case is live, not vacuous)" \
    'grep -q "listening on http://127.0.0.1" "$scratch/serve.log"'
check "no foreign-format refusal was made" '! grep -q "announced a port in a format" "$scratch/serve.log"'

code=$(curl -s -o "$scratch/reply.json" -w '%{http_code}' "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "auto", "messages": [{"role": "user", "content": "plan a redstone door"}]}')
check "a request reaches the nested row ($code)" '[ "$code" = 200 ] && grep -q "model:plan" "$scratch/reply.json"'

if [ $fail -ne 0 ]; then echo "--- serve log"; cat "$scratch/serve.log"; echo FAIL; exit 1; fi
echo PASS
