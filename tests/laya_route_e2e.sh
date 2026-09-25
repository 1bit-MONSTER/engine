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
# must route each request to the device the Laya scorer picks, not hardcode
# Vulkan. tests/fake_backend_route.py stands in for llama-server and zinc, and
# names the device it stands in for in every reply.
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

state="Please summarize the refund policy for this account."
# The scorer is deterministic: the device `1bit route` reports is the device the
# serve process must dispatch this request to.
expected=$("$bin" route --laya-model "$laya_model" --state "$state")
echo "laya picks: $expected"
case "$expected" in npu|hrx|vulkan|zinc) ;; *) echo "FAIL: unexpected device '$expected'"; exit 1 ;; esac

"$bin" serve -m "$scratch/tiny.gguf" --device auto --laya-model "$laya_model" \
    --port "$port" --alias route-test \
    --llama-server "$here/fake_backend_route.py" --zinc "$here/fake_backend_route.py" \
    >"$scratch/serve.log" 2>&1 &
pid=$!
cleanup() { kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -rf "$scratch"; }
trap cleanup EXIT

for _ in $(seq 1 1200); do curl -s -o /dev/null "$api/health" && break; sleep 0.1; done

reply=$(curl -s "$api/v1/completions" -H 'Content-Type: application/json' -d "{\"prompt\": \"$state\"}")
echo "reply: $reply"

fail=0
if ! grep -q "on $expected (" "$scratch/serve.log"; then
    echo "FAIL: serve did not spawn the backend for $expected"
    fail=1
fi
if [[ "$reply" != *"device:$expected"* ]]; then
    echo "FAIL: the reply did not come from the $expected backend"
    fail=1
fi

kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
if [ $fail -ne 0 ]; then
    echo "--- serve log"; cat "$scratch/serve.log"
    echo FAIL; exit 1
fi
echo PASS
