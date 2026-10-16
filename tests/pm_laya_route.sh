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
# tests/pm_laya_route.sh: `1bit serve --pm --pm-router laya` end to end without a GPU
# (docs/pm.md, "Routing with Laya"). The real Laya classifier decides the class of two requests
# labelled in tests/laya_route_cases.json, and the experts file's `pm.routes` table decides
# whether the engine delegates before the PM's first round. The PM's own model and Lemonade are
# the fakes tests/pm_route.sh uses, so nothing is loaded but the Laya checkpoint. Checks:
#   - a labelled `code` request routes to the table's expert: the reply's pm.routing names the
#     class, the expert and a decision time, Lemonade is asked for the routed catalog id exactly
#     once, and the expert's answer reaches the client through the PM;
#   - a labelled `short` request routes nowhere: pm.routing has no expert, Lemonade is not asked,
#     and the PM answers itself;
#   - the startup log names the router and its table.
# usage: tests/pm_laya_route.sh path/to/1bit path/to/laya-checkpoint-dir
set -uo pipefail

bin=${1:?usage: pm_laya_route.sh path/to/1bit path/to/laya-checkpoint-dir}
laya=${2:?usage: pm_laya_route.sh path/to/1bit path/to/laya-checkpoint-dir}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
pids=()
cleanup() { for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -rf "$scratch"; }
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
freeport() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
val() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"; }

touch "$scratch/zaya.gguf"
# The router's table, and the one expert it may pick: an id of its own, so a delegation from the
# router is told apart from the fake PM's own delegate(code) call (which would name `code`).
cat >"$scratch/experts.json" <<'JSON'
{
  "pm": {"routes": {"code": "coder", "short": "", "prose": ""}},
  "experts": [{"expert": "coder", "id": "routed-coder-1bit", "domain": "programming",
               "use_when": "programming", "max_tokens": 64}]
}
JSON

lem_port=$(freeport)
python3 "$here/fake_lemonade.py" "$lem_port" "$scratch/lemonade.jsonl" &
pids+=($!)
port=$(freeport)
PM_RECORD="$scratch/backend.jsonl" "$bin" serve -m "$scratch/zaya.gguf" --device cpu --port "$port" \
    --alias ZAYA1-8B-PM --llama-server "$here/fake_pm_backend.py" --pm --pm-router laya \
    --laya-model "$laya" --pm-experts "$scratch/experts.json" \
    --lemonade-url "http://127.0.0.1:$lem_port" >"$scratch/pm.log" 2>&1 &
pids+=($!)
api="http://127.0.0.1:$port"
for _ in $(seq 1 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' "$api/health")" = 200 ] && break; sleep 0.2; done
check "serve --pm --pm-router laya comes up with the checkpoint" '[ "$(curl -s -o /dev/null -w "%{http_code}" "$api/health")" = 200 ]'
check "  the log names the router and its table" 'grep -q "pm: router laya: .*code=coder" "$scratch/pm.log"'

ask() { # ask OUT TEXT
    curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"model": "ZAYA1-8B-PM", "messages": [{"role": "user", "content": sys.argv[1]}]}))' "$2")" >"$scratch/$1.json"
}
lem_records() { if [ -f "$scratch/lemonade.jsonl" ]; then wc -l <"$scratch/lemonade.jsonl"; else echo 0; fi; }

# 1. a request labelled `code` in tests/laya_route_cases.json (no "code"/"function" in the text, so
#    a delegation can only come from the router, not from the fake PM's own heuristic).
before=$(lem_records)
ask code "Implement an LRU cache in C++ with O(1) get and put."
check "code request: the router's class is code" '[ "$(val "$scratch/code.json" "d[\"pm\"][\"routing\"][\"class\"]")" = code ]'
check "  it selected the table's expert" '[ "$(val "$scratch/code.json" "d[\"pm\"][\"routing\"][\"expert\"]")" = coder ]'
check "  the confidence is reported" '[ "$(val "$scratch/code.json" "0 < d[\"pm\"][\"routing\"][\"score\"] <= 1")" = True ]'
check "  the decision time is reported" '[ "$(val "$scratch/code.json" "int(d[\"pm\"][\"routing\"][\"ms\"])")" -ge 1 ]'
check "  one request reached Lemonade, for the routed id" '[ "$(lem_records)" = $((before + 1)) ] && [ "$(tail -1 "$scratch/lemonade.jsonl" | val /dev/stdin "d[\"model\"]")" = routed-coder-1bit ]'
check "  the PM relayed the expert's answer" '[[ "$(val "$scratch/code.json" "d[\"choices\"][0][\"message\"][\"content\"]")" == *EXPERT_ANSWER* ]]'
check "  pm.delegations says one ok delegation to coder" '[ "$(val "$scratch/code.json" "len(d[\"pm\"][\"delegations\"])")" = 1 ] && [ "$(val "$scratch/code.json" "d[\"pm\"][\"delegations\"][0][\"ok\"]")" = True ] && [ "$(val "$scratch/code.json" "d[\"pm\"][\"delegations\"][0][\"expert\"]")" = coder ]'

# 2. a request labelled `short`: the table leaves it to the PM.
before=$(lem_records)
ask short "What is 17*23?"
check "short request: the router's class is short" '[ "$(val "$scratch/short.json" "d[\"pm\"][\"routing\"][\"class\"]")" = short ]'
check "  no expert was selected" '[ "$(val "$scratch/short.json" "d[\"pm\"][\"routing\"].get(\"expert\")")" = None ]'
check "  Lemonade was not asked" '[ "$(lem_records)" = "$before" ]'
check "  the PM answered itself" '[[ "$(val "$scratch/short.json" "d[\"choices\"][0][\"message\"][\"content\"]")" == "Hi there"* ]]'

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
