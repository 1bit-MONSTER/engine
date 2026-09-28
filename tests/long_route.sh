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
# How `1bit serve --long-model` routes by prompt length (docs/lean.md, "End to end"), without a
# GPU. Fake llama-servers stand in for Vulkan0 (-m) and ROCm0 (--long-model); each answers
# /apply-template and /tokenize (one token per word) and names its device in every reply:
#   - a short first prompt goes to Vulkan0, a long one (>= --long-from tokens) to ROCm0,
#   - chats are counted through /apply-template, completions from their prompt,
#   - a conversation stays where its first request went, even once it is long,
#   - the ROCm backend runs with the Hadamard W4A4 environment,
#   - an unrotated --long-model and a rotated -m are refused.
#
# usage: tests/long_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: long_route.sh path/to/1bit}
scratch=$(mktemp -d)
trap 'kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; rm -rf "$scratch"' EXIT
pid=
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

python3 - "$scratch" <<'PY'
import struct, sys
def gguf(path, stamp):
    kv = []
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv.append(s("general.architecture") + struct.pack("<I", 8) + s("llama"))
    if stamp:
        kv.append(s("onebit.hadamard_q4_0") + struct.pack("<I", 5) + struct.pack("<i", 32))
    open(path, "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
gguf(sys.argv[1] + "/h32.gguf", True)
gguf(sys.argv[1] + "/plain.gguf", False)
PY

cat > "$scratch/backend.py" <<'PY'
#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
dev = argv[argv.index("--device") + 1]
env = {k: v for k, v in os.environ.items() if k.startswith("GGML_")}
open(os.path.join(os.environ["RECORDS"], dev + ".json"), "w").write(json.dumps({"argv": argv, "env": env}))
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._json({"status": "ok"})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/apply-template":
            return self._json({"prompt": " ".join("<%s> %s" % (m["role"], m["content"]) for m in req["messages"])})
        if self.path == "/tokenize":
            return self._json({"tokens": list(range(len(req["content"].split())))})
        text = "device:" + dev
        msg = {"message": {"role": "assistant", "content": text}} if "chat" in self.path else {"text": text}
        self._json({"id": "x", "model": "fake", "choices": [dict(index=0, finish_reason="stop", **msg)]})
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
PY
chmod +x "$scratch/backend.py"

bad=$("$bin" serve -m "$scratch/plain.gguf" --long-model "$scratch/plain.gguf" --port 1 --llama-server "$scratch/backend.py" 2>&1)
check "an unrotated --long-model is refused" '[[ "$bad" == *"Hadamard-rotated"* ]]'
bad=$("$bin" serve -m "$scratch/h32.gguf" --long-model "$scratch/h32.gguf" --port 1 --llama-server "$scratch/backend.py" 2>&1)
check "a rotated -m is refused" '[[ "$bad" == *"already on the W4A4 route"* ]]'

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
mkdir -p "$scratch/rec"
env -u GGML_Q4_0_HADAMARD -u GGML_W4A4_TENSORS RECORDS="$scratch/rec" ONEBIT_LEAN_SERVER="$scratch/backend.py" \
    "$bin" serve -m "$scratch/plain.gguf" --long-model "$scratch/h32.gguf" --long-from 100 --port "$port" \
    --llama-server "$scratch/backend.py" >"$scratch/serve.log" 2>&1 &
pid=$!
for _ in $(seq 1 100); do curl -sf "127.0.0.1:$port/health" >/dev/null && break; sleep 0.1; done

words() { python3 -c 'import sys; print(" ".join(["w"] * int(sys.argv[1])))' "$1"; }
ask() {  # <path> <json body>: the device that answered, then the X-1bit-Route header
    curl -s -D "$scratch/h" "127.0.0.1:$port$1" -H 'Content-Type: application/json' -d "$2" |
        python3 -c 'import json,sys; c=json.load(sys.stdin)["choices"][0]; print((c.get("message") or {}).get("content") or c.get("text"))'
    tr -d '\r' < "$scratch/h" | sed -n 's/^X-1bit-Route: //Ip'
}
chat() { python3 -c 'import json,sys; print(json.dumps({"messages": [{"role": r, "content": c} for r, c in zip(sys.argv[1::2], sys.argv[2::2])]}))' "$@"; }

out=$(ask /v1/chat/completions "$(chat user "hello there")")
check "a short chat goes to Vulkan0" '[[ "$out" == "device:Vulkan0"*"short"*"vulkan"* ]]'
out=$(ask /v1/chat/completions "$(chat user "$(words 150)")")
check "a long chat goes to ROCm0" '[[ "$out" == "device:ROCm0"*"long"*"rocm"* ]]'
out=$(ask /v1/completions "{\"prompt\": \"$(words 40)\"}")
check "a short completion goes to Vulkan0" '[[ "$out" == "device:Vulkan0"* ]]'
out=$(ask /v1/completions "{\"prompt\": \"$(words 120)\"}")
check "a long completion goes to ROCm0" '[[ "$out" == "device:ROCm0"* ]]'
out=$(ask /v1/chat/completions "$(chat user "hello there" assistant "hi" user "$(words 300)")")
check "a conversation that grows long stays on Vulkan0" '[[ "$out" == "device:Vulkan0"*"short"* ]]'
check "the ROCm backend has the Hadamard W4A4 environment" \
    'python3 -c "import json,sys; e=json.load(open(sys.argv[1]))[\"env\"]; sys.exit(not (e.get(\"GGML_Q4_0_HADAMARD\")==\"1\" and e.get(\"GGML_W4A4_TENSORS\")==\"all\"))" "$scratch/rec/ROCm0.json"'
check "the Vulkan backend does not" \
    'python3 -c "import json,sys; sys.exit(bool(json.load(open(sys.argv[1]))[\"env\"]))" "$scratch/rec/Vulkan0.json"'

if [ $fail -ne 0 ]; then cat "$scratch/serve.log"; echo FAIL; exit 1; fi
echo PASS
