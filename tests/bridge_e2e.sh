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
# The device bridge through `1bit serve` (docs/bridge.md), runnable anywhere (CI included):
# tests/fake_bridge_backend.py stands in for the child llama-server, and --bridge-script
# stands in for the NPU drafter, so the whole path is checked with no GPU and no NPU.
# It checks:
#   - serve starts the drafter and tells the child --spec-type draft-external,
#     --spec-external-addr and --spec-draft-n-max, and the child gets the drafts,
#   - the chat reply comes through and is renamed to the served model,
#   - the refusals: two drafters, --parallel > 1, --laya, a .gguf on another device,
#     an NPU directory in a build with no lane, and a child without draft-external.
#
# usage: tests/bridge_e2e.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: bridge_e2e.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
pid=
child=
plain=
cleanup() {
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
    [ -n "$pid" ] && wait "$pid" 2>/dev/null
    [ -n "$child" ] && kill -9 "$child" 2>/dev/null
    [ -n "$plain" ] && kill -9 "$plain" 2>/dev/null
    rm -rf "$scratch"
}
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export FAKE_BRIDGE_EVENTS="$scratch/events.json"
printf '11\n12\n13\n' > "$scratch/draft.txt"
touch "$scratch/tiny.gguf"
api="http://127.0.0.1:$port"
backend="$here/fake_bridge_backend.py"

"$bin" serve -m "$scratch/tiny.gguf" --device cpu --port "$port" --alias bridge-model \
    --llama-server "$backend" --bridge-script "$scratch/draft.txt" --bridge-k 3 \
    >"$scratch/serve.log" 2>&1 &
pid=$!

code=000
for _ in $(seq 1 100); do code=$(curl -s -o /dev/null -w '%{http_code}' "$api/health"); [ "$code" = 200 ] && break; sleep 0.1; done
check "/health 200 once the child is up ($code)" '[ "$code" = 200 ]'
check "serve says the bridge is on" 'grep -q "the bridge drafts on" "$scratch/serve.log"'
check "the bridge logs the round it served" 'grep -q "1bit bridge: draft round 1" "$scratch/serve.log"'

reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "bridge-model", "messages": [{"role": "user", "content": "hi"}]}')
check "the chat reply comes through" '[[ "$reply" == *Paris* ]]'
check "the reply names the served model" '[[ "$reply" == *"\"model\":\"bridge-model\""* ]]'

for _ in $(seq 1 50); do [ -s "$scratch/events.json" ] && break; sleep 0.1; done
check "the child was told the speculative type" 'python3 -c "
import json,sys; a=json.load(open(sys.argv[1]))[\"argv\"]
i=a.index(\"--spec-type\"); assert a[i+1]==\"draft-external\", a
i=a.index(\"--spec-draft-n-max\"); assert a[i+1]==\"3\", a
" "$scratch/events.json"'
check "the child was told the bridge address" 'python3 -c "
import json,sys; d=json.load(open(sys.argv[1])); assert d[\"addr\"].startswith(\"127.0.0.1:\"), d; assert not d[\"error\"], d
" "$scratch/events.json"'
check "the drafter answered with the script" 'python3 -c "
import json,sys; d=json.load(open(sys.argv[1])); assert d[\"draft\"]==[11,12,13], d
" "$scratch/events.json"'

# The child outlives the test if serve does not take it down.
child=$(pgrep -P "$pid" | head -1)
kill -9 "$pid"; wait "$pid" 2>/dev/null; pid=
sleep 1
check "no child outlives serve" '[ -z "$child" ] || ! kill -0 "$child" 2>/dev/null'

# The flag is the only thing that turns the bridge on: a serve that does not ask for it tells
# the child nothing about it, which is why the feature cannot change any other run.
rm -f "$scratch/events.json"
port2=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$bin" serve -m "$scratch/tiny.gguf" --device cpu --port "$port2" --llama-server "$backend" \
    >"$scratch/plain.log" 2>&1 &
plain=$!
code=000
for _ in $(seq 1 100); do code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port2/health"); [ "$code" = 200 ] && break; sleep 0.1; done
curl -s "http://127.0.0.1:$port2/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "plain", "messages": [{"role": "user", "content": "hi"}]}' >/dev/null
for _ in $(seq 1 50); do [ -s "$scratch/events.json" ] && break; sleep 0.1; done
check "a plain serve passes the child no speculative type" 'python3 -c "
import json,sys; a=json.load(open(sys.argv[1]))[\"argv\"]; assert \"--spec-type\" not in a, a
" "$scratch/events.json"'
check "a plain serve starts no drafter" '! grep -q "the bridge drafts on" "$scratch/plain.log"'
kill -9 "$plain" 2>/dev/null; wait "$plain" 2>/dev/null; plain=

# The refusals: each must fail fast with the reason, not launch something that ignores the flag.
refuse() {
    local name=$1 want=$2; shift 2
    out=$(timeout 25 "$bin" serve -m "$scratch/tiny.gguf" --device cpu --port 0 --llama-server "$backend" "$@" 2>&1)
    check "$name" '[[ "$out" == *"$want"* ]]'
}
: > "$scratch/head.gguf"
refuse "two drafters are refused" "two drafters" --bridge-script "$scratch/draft.txt" --mtp "$scratch/head.gguf"
refuse "--parallel is refused" "one NPU lane" --bridge-script "$scratch/draft.txt" --parallel 2
refuse "--laya is refused" "leave --laya off" --bridge-script "$scratch/draft.txt" --laya-model "$scratch/nope"
refuse "an NPU directory needs the lane" "NPU" --bridge-draft "$scratch/not-a-model-dir"

out=$(FAKE_BRIDGE_NO_EXTERNAL=1 timeout 25 "$bin" serve -m "$scratch/tiny.gguf" --device cpu --port 0 \
    --llama-server "$backend" --bridge-script "$scratch/draft.txt" 2>&1)
check "a child without draft-external is refused" '[[ "$out" == *"draft-external"* ]]'

if [ "$fail" = 0 ]; then echo "bridge e2e: all checks passed"; else echo "bridge e2e: $fail check(s) failed"; fi
exit "$fail"
