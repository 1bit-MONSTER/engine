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

# laya_models_e2e.sh: `1bit serve --laya-models` end to end without a GPU (docs/laya-models.md).
# Two rows, two fake llama-servers (tests/fake_backend_route.py, which names the model file that
# answered), each behind its own `1bit serve` child. The table's size gate sends long requests to
# `reflex` and min_confidence 1.01 sends every other one to the default `plan`, so where a request
# lands is known whatever Laya answers, while Laya is still asked (laya_class, router_ms > 0).
# Checks: the reply comes from the row the header names; X-1bit-Route / -Model / -Ms / -Id; the
# reply's `route` object; a request naming a row's alias skips Laya; a repeated state is cached;
# GET /v1/route/decisions, /v1/route/table, /v1/models and /metrics; refusals by name.
set -uo pipefail

bin=${1:?usage: laya_models_e2e.sh path/to/1bit path/to/laya-model-dir}
laya_model=${2:?usage: laya_models_e2e.sh path/to/1bit path/to/laya-model-dir}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
touch "$scratch/small.gguf" "$scratch/big.gguf"
fake="$here/fake_backend_route.py"
cat >"$scratch/table.json" <<EOF
{"question": {"instructions": "What does the game agent need to answer this step?",
              "classes": {"reflex": "one quick action from what is in view",
                          "plan": "planning: a crafting tree, a build plan or recovering after a death"}},
 "default": "plan", "min_confidence": 1.01,
 "size_gate": {"chars": 200, "class": "reflex"},
 "classes": {"reflex": {"model": "small.gguf", "device": "cpu", "alias": "small-npu", "args": ["--llama-server", "$fake"]},
             "plan": {"model": "$scratch/big.gguf", "device": "cpu", "alias": "big-hrx", "args": ["--llama-server", "$fake"]}}}
EOF
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
api="http://127.0.0.1:$port"
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

out=$("$bin" serve --laya-models "$scratch/table.json" --port "$port" 2>&1)
check "--laya-models without --laya says so" '[[ "$out" == *"needs --laya"* ]]'
out=$("$bin" serve --laya-models "$scratch/table.json" --laya-model "$laya_model" -m "$scratch/big.gguf" --port "$port" 2>&1)
check "--laya-models with -m is refused" '[[ "$out" == *"drop -m"* ]]'
echo '{"question":{"instructions":"x","classes":{"a":"d","b":"e"}},"classes":{"a":{"model":"m","device":"hrx"}}}' >"$scratch/bad.json"
out=$("$bin" serve --laya-models "$scratch/bad.json" --laya-model "$laya_model" --port "$port" 2>&1)
check "a table with a class and no row is refused by name" '[[ "$out" == *"\"b\" has no row"* ]]'

"$bin" serve --laya-models "$scratch/table.json" --laya-model "$laya_model" --port "$port" >"$scratch/serve.log" 2>&1 &
pid=$!
cleanup() { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -rf "$scratch"; }
trap cleanup EXIT
for _ in $(seq 1 1200); do [ "$(curl -s -o /dev/null -w '%{http_code}' "$api/health")" = 200 ] && break; sleep 0.1; done
check "ready" '[ "$(curl -s -o /dev/null -w "%{http_code}" "$api/health")" = 200 ]'

ask() {  # model, text -> headers in $scratch/h, body in $scratch/b
    python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"system","content":"You play a voxel game."},{"role":"user","content":sys.argv[2]}]}))' "$1" "$2" >"$scratch/req.json"
    curl -s -D "$scratch/h" -o "$scratch/b" "$api/v1/chat/completions" -H 'Content-Type: application/json' -d @"$scratch/req.json"
    tr -d '\r' <"$scratch/h" >"$scratch/h2"
}
hdr() { sed -n "s/^[Xx]-1bit-$1: //Ip" "$scratch/h2"; }
field() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$scratch/b" "$1"; }

ask auto "A tree is in front of you. What do you do now?"
check "short step -> plan row (min_confidence fallback): reply from big" 'grep -q "model:big" "$scratch/b"'
check "  X-1bit-Route ($(hdr Route))" '[[ "$(hdr Route)" == "plan "*" cpu" ]]'
check "  X-1bit-Route-Model ($(hdr Route-Model))" '[ "$(hdr Route-Model)" = big-hrx ]'
check "  X-1bit-Route-Ms > 0 ($(hdr Route-Ms))" 'python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "$(hdr Route-Ms)"'
check "  route object: Laya was asked ($(field "d[\"route\"][\"laya_class\"]"))" '[ -n "$(field "d[\"route\"][\"laya_class\"]")" ]'
check "  reply model is the row alias" '[ "$(field "d[\"model\"]")" = big-hrx ]'

long=$(python3 -c 'print("Inventory: " + ", ".join(["oak log"] * 40))')
ask auto "$long"
check "long step -> size gate reflex: reply from small" 'grep -q "model:small" "$scratch/b"'
check "  route class reflex, confidence 1" '[ "$(field "d[\"route\"][\"class\"]")" = reflex ] && [ "$(field "d[\"route\"][\"confidence\"]")" = 1.0 ]'

ask small-npu "A tree is in front of you. What do you do now?"
check "a row alias is pinned: reply from small, no Laya" 'grep -q "model:small" "$scratch/b" && [[ "$(hdr Route)" == "pinned "* ]]'

ask auto "A tree is in front of you. What do you do now?"
check "a repeated state is cached" '[ "$(field "d[\"route\"][\"cached\"]")" = True ] && [ "$(hdr Route-Ms)" = 0.0 ]'

dec=$(curl -s "$api/v1/route/decisions?since=0")
check "decisions: 4 recorded" '[ "$(echo "$dec" | python3 -c "import json,sys; print(len(json.load(sys.stdin)[\"data\"]))")" = 4 ]'
check "decisions: usage and timings relayed" 'echo "$dec" | python3 -c "import json,sys; d=json.load(sys.stdin)[\"data\"][0]; sys.exit(0 if d[\"completion_tokens\"]==3 and d[\"decode_tok_s\"]==50.0 and d[\"request_ms\"]>0 else 1)"'
check "decisions: since filters" '[ "$(curl -s "$api/v1/route/decisions?since=3" | python3 -c "import json,sys; print(len(json.load(sys.stdin)[\"data\"]))")" = 1 ]'
tab=$(curl -s "$api/v1/route/table")
check "table: two classes with their rows" 'echo "$tab" | python3 -c "import json,sys; t=json.load(sys.stdin); sys.exit(0 if [c[\"model\"] for c in t[\"classes\"]]==[\"small-npu\",\"big-hrx\"] else 1)"'
check "/v1/models lists auto and both aliases" '[ "$(curl -s "$api/v1/models" | python3 -c "import json,sys; print(\" \".join(m[\"id\"] for m in json.load(sys.stdin)[\"data\"]))")" = "auto small-npu big-hrx" ]'
met=$(curl -s "$api/metrics")
check "metrics: decisions by class/model/device" '[[ "$met" == *"onebit_route_decisions_total{class=\"plan\",model=\"big-hrx\",device=\"cpu\"} 2"* ]]'
check "metrics: router ms counted for Laya decisions only" '[[ "$met" == *"onebit_route_router_count{class=\"plan\",model=\"big-hrx\",device=\"cpu\"} 1"* ]]'
check "metrics: last decode tok/s" '[[ "$met" == *"onebit_route_last_decode_tok_s{model=\"small-npu\",device=\"cpu\"} 50"* ]]'
check "serve logged every decision" '[ "$(grep -c "laya-models: #[0-9]" "$scratch/serve.log")" = 4 ]'

if [ $fail -ne 0 ]; then echo "--- serve log"; cat "$scratch/serve.log"; echo FAIL; exit 1; fi
echo PASS
