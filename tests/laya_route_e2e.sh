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

# Laya per-request routing end to end: `1bit serve --device auto --laya-model <dir>`
# must route each request to the device the Laya scorer picks among the GGUF
# devices (vulkan, hrx, zinc) -- not hardcode Vulkan, and never offer NPU for a
# .gguf (NPU runs Q4NX directories). tests/fake_backend_route.py stands in for
# llama-server and zinc, and names the device it stands in for in every reply.
#
# usage: tests/laya_route_e2e.sh path/to/1bit path/to/laya-model-dir
set -uo pipefail

bin=${1:?usage: laya_route_e2e.sh path/to/1bit path/to/laya-model-dir}
laya_model=${2:?usage: laya_route_e2e.sh path/to/1bit path/to/laya-model-dir}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
touch "$scratch/tiny.gguf"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
api="http://127.0.0.1:$port"

# A .gguf never routes to npu; NPU is the device of a Q4NX model directory.
npu_only=$("$bin" route --laya-model "$laya_model" --devices npu --state "anything")
echo "npu-only decision: $npu_only"
[ "$npu_only" = "npu" ] || { echo "FAIL: --devices npu did not route to npu"; exit 1; }

states=(
    "Summarize this support ticket: the customer was billed twice and wants a refund."
    "Write a haiku about the ocean."
    "What is the capital of France?"
)

"$bin" serve -m "$scratch/tiny.gguf" --device auto --laya-model "$laya_model" \
    --port "$port" --alias route-test \
    --llama-server "$here/fake_backend_route.py" --zinc "$here/fake_backend_route.py" \
    >"$scratch/serve.log" 2>&1 &
pid=$!
cleanup() { kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -rf "$scratch"; }
trap cleanup EXIT

for _ in $(seq 1 1200); do curl -s -o /dev/null "$api/health" && break; sleep 0.1; done

fail=0
seen=""
for state in "${states[@]}"; do
    expected=$("$bin" route --laya-model "$laya_model" --devices vulkan,hrx,zinc --state "$state")
    case "$expected" in vulkan|hrx|zinc) ;; *) echo "FAIL: unexpected device '$expected'"; exit 1 ;; esac
    seen="$seen $expected"
    reply=$(curl -s "$api/v1/completions" -H 'Content-Type: application/json' -d "{\"prompt\": \"$state\"}")
    if [[ "$reply" != *"device:$expected"* ]]; then
        echo "FAIL: for state [$state] laya picked $expected but reply was: $reply"
        fail=1
    else
        echo "ok   [$state] -> $expected"
    fi
    if ! grep -q "on $expected (" "$scratch/serve.log"; then
        echo "FAIL: serve did not spawn the $expected backend"
        fail=1
    fi
done

# The router must actually exercise more than one backend.
distinct=$(for d in $seen; do echo "$d"; done | sort -u | wc -l)
echo "distinct devices routed: $distinct"
if [ "$distinct" -lt 2 ]; then
    echo "FAIL: the test states only exercised one device; routing across devices is unproven"
    fail=1
fi

kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
if [ $fail -ne 0 ]; then
    echo "--- serve log"; cat "$scratch/serve.log"
    echo FAIL; exit 1
fi
echo PASS
