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
# How `1bit serve` routes a Hadamard-rotated Q4_0 file (tools/hadamard_q4_0.py stamps
# onebit.hadamard_q4_0 = 32; docs/lean.md), without a GPU:
#   - --device vulkan refuses it,
#   - --device auto sends it to the ROCm route (--device ROCm0) with GGML_Q4_0_HADAMARD=1 and
#     GGML_W4A4_TENSORS=all in the backend's environment,
#   - an unstamped file with --device auto still goes to Vulkan with neither variable.
#
# usage: tests/hadamard_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: hadamard_route.sh path/to/1bit}
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

# a backend that records how it was started, then answers /health
cat > "$scratch/backend.py" <<'PY'
#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
rec = {"argv": argv, "env": {k: v for k, v in os.environ.items() if k.startswith("GGML_")}}
open(os.environ["RECORD"], "w").write(json.dumps(rec))
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = b'{"status":"ok"}'
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
chmod +x "$scratch/backend.py"

refused=$(env -u GGML_Q4_0_HADAMARD -u GGML_W4A4_TENSORS "$bin" serve -m "$scratch/h32.gguf" --device vulkan \
    --port 1 --llama-server "$scratch/backend.py" 2>&1)
check "--device vulkan refuses a rotated file" '[[ "$refused" == *"Hadamard-rotated"* ]]'

run() {  # <model> <record>: serve with --device auto until the backend has recorded its start
    local port
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    env -u GGML_Q4_0_HADAMARD -u GGML_W4A4_TENSORS RECORD="$2" "$bin" serve -m "$1" --device auto --port "$port" \
        --llama-server "$scratch/backend.py" >"$2.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 100); do [ -s "$2" ] && break; sleep 0.1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}
field() { python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"; }

run "$scratch/h32.gguf" "$scratch/h32.json"
check "--device auto sends a rotated file to ROCm0" '[ "$(field "$scratch/h32.json" "r[\"argv\"][r[\"argv\"].index(\"--device\")+1]")" = ROCm0 ]'
check "  with GGML_Q4_0_HADAMARD=1" '[ "$(field "$scratch/h32.json" "r[\"env\"].get(\"GGML_Q4_0_HADAMARD\")")" = 1 ]'
check "  and GGML_W4A4_TENSORS=all" '[ "$(field "$scratch/h32.json" "r[\"env\"].get(\"GGML_W4A4_TENSORS\")")" = all ]'

run "$scratch/plain.gguf" "$scratch/plain.json"
check "an unstamped file still goes to Vulkan0" '[ "$(field "$scratch/plain.json" "r[\"argv\"][r[\"argv\"].index(\"--device\")+1]")" = Vulkan0 ]'
check "  with neither variable" '[ "$(field "$scratch/plain.json" "sorted(r[\"env\"])")" = "[]" ]'

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
