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
# End-to-end test of Laya routing in `1bit serve` (RFC #186, docs/laya.md): Laya classifies each
# conversation and the route policy picks the device. tests/route-policy-e2e.json sends code,
# prose and short requests to three different devices (tests/fake_backend_route.py stands in for
# llama-server and zinc and answers with its device), so the test proves:
#   - each request reaches the device the policy names for the class `1bit route` reports,
#   - the reply carries X-1bit-Route "<class> <confidence> <device>",
#   - a later turn of the same conversation reuses the first decision (one decision logged),
#   - --laya finds the checkpoint through $ONEBIT_LAYA_MODEL, and fails clearly without one,
#   - a Q4NX-only candidate set (--devices npu) still routes to npu.
#
# usage: tests/laya_route_e2e.sh path/to/1bit path/to/laya-model-dir
set -uo pipefail

bin=${1:?usage: laya_route_e2e.sh path/to/1bit path/to/laya-model-dir}
laya_model=${2:?usage: laya_route_e2e.sh path/to/1bit path/to/laya-model-dir}
here=$(cd "$(dirname "$0")" && pwd)
policy="$here/route-policy-e2e.json"
# The devices serve offers Laya for a .gguf in this build (gguf_devices() in app/serve.cpp): HRX
# and zinc in a build with HRX; the CPU and zinc in one without (CI).
cache="$(dirname "$bin")/CMakeCache.txt"
devices=cpu,zinc
if grep -q "^ONEBIT_HRX:BOOL=ON" "$cache" 2>/dev/null; then devices=hrx,zinc; fi
echo "candidates: $devices"
scratch=$(mktemp -d)
touch "$scratch/tiny.gguf"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
api="http://127.0.0.1:$port"
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

npu_only=$("$bin" route --laya-model "$laya_model" --devices npu --state "anything" 2>/dev/null)
check "--devices npu routes to npu ($npu_only)" '[ "$npu_only" = npu ]'

missing=$(HOME="$scratch" XDG_DATA_HOME= ONEBIT_LAYA_MODEL= "$bin" serve -m "$scratch/tiny.gguf" --device auto --laya \
    --port "$port" --llama-server "$here/fake_backend_route.py" 2>&1)
check "--laya without a checkpoint says how to install one" '[[ "$missing" == *fetch-laya.sh* ]]'

ONEBIT_LAYA_MODEL="$laya_model" "$bin" serve -m "$scratch/tiny.gguf" --device auto --laya --route-policy "$policy" \
    --port "$port" --alias route-test \
    --llama-server "$here/fake_backend_route.py" --zinc "$here/fake_backend_route.py" \
    >"$scratch/serve.log" 2>&1 &
pid=$!
cleanup() { kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -rf "$scratch"; }
trap cleanup EXIT
for _ in $(seq 1 1200); do curl -s -o /dev/null "$api/health" && break; sleep 0.1; done

states=(
    "Write a Rust function that reverses a linked list."
    "Explain how vaccines train the immune system."
    "What is 17*23?"
)
seen=""
for state in "${states[@]}"; do
    read -r cls conf expected < <("$bin" route --laya-model "$laya_model" --route-policy "$policy" \
        --devices "$devices" --classify --state "$state" 2>/dev/null)
    seen="$seen $expected"
    body=$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}]}))' "$state")
    hdr=$(curl -s -D - -o "$scratch/reply.json" "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" \
        | tr -d '\r' | sed -n 's/^[Xx]-1bit-[Rr]oute: //p')
    check "[$state] class $cls -> $expected, reply from $expected" 'grep -q "device:$expected" "$scratch/reply.json"'
    check "  X-1bit-Route header ($hdr)" '[[ "$hdr" == "$cls "*" $expected" ]]'
done
distinct=$(for d in $seen; do echo "$d"; done | sort -u | wc -l)
check "the policy routed to $distinct distinct devices" '[ "$distinct" -ge 2 ]'

# a second turn of the first conversation: same class, no new decision
decisions_before=$(grep -c "1bit serve: laya: [a-z_]* [0-9.]* " "$scratch/serve.log")
body=$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]},{"role":"assistant","content":"fn reverse() {}"},{"role":"user","content":"thanks, now add a test"}]}))' "${states[0]}")
curl -s -o "$scratch/reply2.json" "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body"
decisions_after=$(grep -c "1bit serve: laya: [a-z_]* [0-9.]* " "$scratch/serve.log")
first=$(echo "$seen" | awk '{print $1}')
check "second turn reuses the decision (decisions $decisions_before -> $decisions_after)" '[ "$decisions_after" = "$decisions_before" ]'
check "  and goes to the same device ($first)" 'grep -q "device:$first" "$scratch/reply2.json"'

# a conversation seen first at a later turn is classified on its head (the messages through the
# first user message, which is all its cache key covers), not on the turns after it
read -r head_cls _ head_dev < <("$bin" route --laya-model "$laya_model" --route-policy "$policy" \
    --devices "$devices" --classify --state "${states[2]}" 2>/dev/null)
body=$(python3 -c 'import json,sys; print(json.dumps({"model":"other","messages":[{"role":"user","content":sys.argv[1]},{"role":"assistant","content":"391"},{"role":"user","content":sys.argv[2]}]}))' "${states[2]}" "${states[1]}")
hdr=$(curl -s -D - -o "$scratch/reply3.json" "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" \
    | tr -d '\r' | sed -n 's/^[Xx]-1bit-[Rr]oute: //p')
check "a new conversation's later turns do not decide its class ($hdr)" '[[ "$hdr" == "$head_cls "*" $head_dev" ]]'

if [ $fail -ne 0 ]; then echo "--- serve log"; cat "$scratch/serve.log"; echo FAIL; exit 1; fi
echo PASS
