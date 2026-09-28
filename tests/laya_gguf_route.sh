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
# `1bit serve --laya` with the GGUF scorer (ggmlc's `laya daemon`, docs/laya.md), without a GPU
# or a model: a fake daemon (ONEBIT_LAYA_GGML) answers by keyword and records what it was asked,
# the pinned GGUF name under $ONEBIT_LAYA_MODEL/gguf/ is an empty file, and fake llama-servers
# and zinc answer with their device (tests/route-policy-e2e.json: code -> hrx, prose -> zinc,
# short -> vulkan). It checks that
#   - serve starts the daemon (daemon <gguf> --family typed-decisions --device vulkan) and needs
#     no safetensors checkpoint,
#   - each class reaches its device, with X-1bit-Route "<class> <confidence> <device>",
#   - the daemon gets serve's question: the three non-long_doc classes, the request as state,
#   - a request of 1024 characters or more is long_doc without asking the daemon,
#   - --laya-model FILE.gguf without ggmlc's laya fails and says how to build it.
#
# usage: tests/laya_gguf_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: laya_gguf_route.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
pid=
trap 'kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

touch "$scratch/tiny.gguf"
mkdir -p "$scratch/laya/gguf"
touch "$scratch/laya/gguf/laya_typed_decisions_q8_0.gguf"
cat > "$scratch/laya-daemon.py" <<'PY'
#!/usr/bin/env python3
import json, os, sys
log = open(os.environ["LAYA_LOG"], "a")
log.write(json.dumps({"argv": sys.argv[1:]}) + "\n"); log.flush()
print(json.dumps({"status": "ready", "model": "laya"}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    log.write(json.dumps(req) + "\n"); log.flush()
    s = req["state"].lower()
    choice = "code" if "function" in s else "prose" if "explain" in s else "short"
    q = req["questions"]["cls"]
    probs = {k: (0.9 if k == choice else 0.05) for k in q["criteria"]}
    print(json.dumps({"answers": {"cls": {"type": "choice", "choice": choice, "confidence": 0.8,
                                          "probabilities": probs}}}), flush=True)
PY
chmod +x "$scratch/laya-daemon.py"

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
api="http://127.0.0.1:$port"

nobin=$(env -u ONEBIT_LAYA_GGML "$bin" serve -m "$scratch/tiny.gguf" --device auto \
    --laya-model "$scratch/laya/gguf/laya_typed_decisions_q8_0.gguf" --port "$port" \
    --llama-server "$here/fake_backend_route.py" 2>&1)
check "--laya-model FILE.gguf without ggmlc's laya says how to build it" '[[ "$nobin" == *ONEBIT_LAYA_GGML* ]]'

LAYA_LOG="$scratch/daemon.log" ONEBIT_LAYA_GGML="$scratch/laya-daemon.py" ONEBIT_LAYA_MODEL="$scratch/laya" \
    "$bin" serve -m "$scratch/tiny.gguf" --device auto --laya --route-policy "$here/route-policy-e2e.json" \
    --port "$port" --llama-server "$here/fake_backend_route.py" --zinc "$here/fake_backend_route.py" \
    >"$scratch/serve.log" 2>&1 &
pid=$!
for _ in $(seq 1 300); do curl -sf -o /dev/null "$api/health" && break; sleep 0.1; done

argv=$(head -1 "$scratch/daemon.log" 2>/dev/null)
check "serve starts the daemon on the pinned GGUF, typed-decisions, Vulkan" \
    '[[ "$argv" == *"\"daemon\", \"$scratch/laya/gguf/laya_typed_decisions_q8_0.gguf\", \"--family\", \"typed-decisions\", \"--device\", \"vulkan\""* ]]'

ask() {  # <text>: the replying device, then the X-1bit-Route header
    local body
    body=$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}]}))' "$1")
    curl -s -D "$scratch/h" -o "$scratch/r.json" "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body"
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"])' "$scratch/r.json"
    tr -d '\r' < "$scratch/h" | sed -n 's/^[Xx]-1bit-[Rr]oute: //p'
}
out=$(ask "Write a Rust function that reverses a linked list.")
check "code -> hrx ($(echo $out))" '[[ "$out" == "device:hrx"*"code 0.80 hrx"* ]]'
out=$(ask "Explain how vaccines train the immune system.")
check "prose -> zinc ($(echo $out))" '[[ "$out" == "device:zinc"*"prose 0.80 zinc"* ]]'
out=$(ask "What is 17*23?")
check "short -> vulkan ($(echo $out))" '[[ "$out" == "device:vulkan"*"short 0.80 vulkan"* ]]'

q=$(sed -n 2p "$scratch/daemon.log")
check "the daemon gets the request as state and the three classes" \
    'python3 -c "import json,sys; r=json.loads(sys.argv[1]); c=r[\"questions\"][\"cls\"]; sys.exit(not (r[\"state\"].startswith(\"Write a Rust\") and c[\"type\"]==\"choice\" and list(c[\"criteria\"])==[\"code\",\"prose\",\"short\"]))" "$q"'

asked=$(grep -c '"state"' "$scratch/daemon.log")
out=$(ask "$(python3 -c 'print("log line 42: ok\n" * 80)')")
check "a 1024+ character request is long_doc ($(echo $out))" '[[ "$out" == "device:vulkan"*"long_doc 1.00 vulkan"* ]]'
check "  without asking the daemon" '[ "$(grep -c "\"state\"" "$scratch/daemon.log")" = "$asked" ]'

if [ $fail -ne 0 ]; then echo "--- serve log"; cat "$scratch/serve.log"; echo "--- daemon log"; cat "$scratch/daemon.log"; echo FAIL; exit 1; fi
echo PASS
