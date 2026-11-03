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
# The shared-KV route (ONEBIT_KV_SHARE=1, docs/hybrid.md) without a model, a GPU or a dma-buf: the fake
# backend speaks the two endpoints the real llama-server has (fork 1bit-MONSTER/llama.cpp#164: ?action=share
# sends [uint32 length][JSON] plus the region fds over a unix socket and returns a share_id; ?action=adopt
# takes the tokens back and refuses a wrong share_id), and fake_kvshare_provider.py plays the provider: it
# fails unless it was handed one readable fd per region, which is only true if the engine received the fds
# over the socket and passed them on through exec.
#
# What this pins down that the GPU gate cannot (that one needs an HRX box): the listen -> share -> fds ->
# provider -> adopt order, the inherited descriptors, that adopt carries the share_id, that the socket is
# cleaned up, and the fallback when a provider fails.
#
# usage: tests/hybrid_kvshare_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: hybrid_kvshare_route.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
touch "$scratch/tiny.gguf"
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
pid=
cleanup() { [ -n "$pid" ] && { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }; rm -rf "$scratch"; }
trap cleanup EXIT

start() {   # start <log> [env...]
    local log=$1; shift
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    env FAKE_KVSHARE_MARKER="$scratch/provider.json" "$@" "$bin" serve -m "$scratch/tiny.gguf" --device hybrid --port "$port" --alias hybrid-model \
        --llama-server "$here/fake_backend.py" --no-mem-check >"$log" 2>&1 &
    pid=$!
    api="http://127.0.0.1:$port"
    local code=000
    for _ in $(seq 1 100); do code=$(curl -s -o /dev/null -w '%{http_code}' "$api/health"); [ "$code" = 200 ] && break; sleep 0.1; done
    [ "$code" = 200 ]
}
stop() { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; pid=; }
ask() { curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "hybrid-model", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'; }

check "serve starts with the shared route on" \
    'start "$scratch/a.log" ONEBIT_KV_SHARE=1 ONEBIT_HYBRID_PREFILL="$here/fake_kvshare_provider.py"'
reply=$(ask)
check "the reply comes through" '[[ "$reply" == *Paris* ]]'
check "the request was pinned to slot 0" '[[ "$reply" == *"\"fake_id_slot\":0"* ]]'
sleep 0.3
# the share time is logged with %.3f ms, so a sub-millisecond match would assert the box's load, not the path
check "the handoff ran (share + adopt, not the file route)" 'grep -q "hybrid: .* share [0-9.]* ms, first byte" "$scratch/a.log"'
check "no fallback happened" '! grep -q "the backend prefills this request itself" "$scratch/a.log"'
check "the provider got one readable fd per region" \
    'python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d[\"fds\"]==1 and d[\"readable\"]==[16] and d[\"n_ctx\"]==4096 else 1)" "$scratch/provider.json" 2>/dev/null'
check "the socket is gone from the slot directory" '! ls /tmp/1bit-hybrid-*/kvshare-*.sock 2>/dev/null | grep -q .'
stop

# the adopt must carry the share_id: the fake backend refuses a wrong or missing one, so a provider that runs
# but whose id does not come back leaves the request to the backend -- that is the fallback this checks
check "serve starts with a provider that refuses the protocol" \
    'start "$scratch/b.log" ONEBIT_KV_SHARE=1 ONEBIT_HYBRID_PREFILL=/bin/false'
reply=$(ask)
check "the reply still comes through" '[[ "$reply" == *Paris* ]]'
check "the fallback is named once" 'grep -q "the backend prefills this request itself" "$scratch/b.log"'
check "no share/adopt line was reported" '! grep -q "hybrid: .* share [0-9.]* ms, first byte" "$scratch/b.log"'
stop

if [ $fail -ne 0 ]; then echo "--- log (tail)"; tail -30 "$scratch/a.log"; echo FAIL; exit 1; fi
echo PASS
