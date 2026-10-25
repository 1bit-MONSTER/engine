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
# `1bit serve`'s greedy short-circuit (issue #560, docs/serve.md "Greedy requests"), without a GPU:
# tests/fake_backend.py stands in for llama-server and echoes the sampling fields serve forwarded.
#   - temperature 0 gets top_k 1 on both OpenAI routes (argmax without the full-vocab chain),
#   - a request that names its own top_k keeps it,
#   - a request with a real temperature is forwarded untouched,
#   - a body with no temperature is not rewritten (llama-server's own default applies).
#
# usage: tests/greedy_shortcut.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: greedy_shortcut.sh path/to/1bit}
scratch=$(mktemp -d)
trap 'kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; rm -rf "$scratch"' EXIT
pid=
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fi; [ "$(eval "$2" >/dev/null 2>&1; echo $?)" = 0 ] || fail=1; }

python3 - "$scratch" <<'PY'
import struct, sys
def gguf(path, arch="llama"):
    kv = []
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv.append(s("general.architecture") + struct.pack("<I", 8) + s(arch))
    open(path, "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
gguf(sys.argv[1] + "/m.gguf")
PY

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$bin" serve -m "$scratch/m.gguf" --device auto --port "$port" \
    --llama-server "$(dirname "$0")/fake_backend.py" >"$scratch/serve.log" 2>&1 &
pid=$!
for _ in $(seq 1 100); do curl -sf "localhost:$port/health" >/dev/null 2>&1 && break; sleep 0.2; done

ask() {  # ask <path> <json> -> reply body
    curl -s -m 20 -X POST "localhost:$port$1" -H 'Content-Type: application/json' -d "$2"
}

r=$(ask /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}')
check "temperature 0 on /v1/chat/completions gets top_k 1" '[[ "$r" == *"\"fake_top_k\": 1"* || "$r" == *"\"fake_top_k\":1"* ]]'

r=$(ask /v1/completions '{"model":"m","prompt":"hi","max_tokens":4,"temperature":0}')
check "temperature 0 on /v1/completions gets top_k 1" '[[ "$r" == *"\"fake_top_k\": 1"* || "$r" == *"\"fake_top_k\":1"* ]]'

r=$(ask /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"top_k":5}')
check "an explicit top_k is kept" '[[ "$r" == *"\"fake_top_k\": 5"* || "$r" == *"\"fake_top_k\":5"* ]]'

r=$(ask /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0.7}')
check "a real temperature is forwarded untouched (no top_k added)" '[[ "$r" == *"\"fake_top_k\": null"* || "$r" == *"\"fake_top_k\":null"* ]] && [[ "$r" == *"0.7"* ]]'

r=$(ask /v1/completions '{"model":"m","prompt":"hi","max_tokens":4}')
check "a body without temperature is not rewritten" '[[ "$r" == *"\"fake_top_k\": null"* || "$r" == *"\"fake_top_k\":null"* ]]'

if [ $fail -ne 0 ]; then echo "--- $scratch/serve.log"; tail -20 "$scratch/serve.log"; echo FAIL; exit 1; fi
echo PASS
