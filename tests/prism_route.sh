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
# How `1bit serve` routes a PrismML Hadamard-folded file (Ternary Bonsai; prism.hadamard.* keys,
# docs/hrx.md), without a GPU:
#   - --device auto sends a converted file (tools/ternary_to_q4_0.py) to HRX0,
#   - --device vulkan refuses it (upstream llama.cpp would ignore the rotation),
#   - a file still in PrismML's ternary types is refused with the converter's name,
#   - a plain file with --device auto still goes to Vulkan0.
#
# usage: tests/prism_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: prism_route.sh path/to/1bit}
scratch=$(mktemp -d)
trap 'kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; rm -rf "$scratch"' EXIT
pid=
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

python3 - "$scratch" <<'PY'
import struct, sys
def gguf(path, prism, file_type):
    kv = []
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv.append(s("general.architecture") + struct.pack("<I", 8) + s("qwen35"))
    kv.append(s("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", file_type))
    if prism:
        names = [b"output.weight", b"blk.0.ffn_up.weight"]
        kv.append(s("prism.hadamard.weight_names") + struct.pack("<I", 9) + struct.pack("<IQ", 8, len(names))
                  + b"".join(struct.pack("<Q", len(n)) + n for n in names))
        kv.append(s("prism.hadamard.version") + struct.pack("<I", 4) + struct.pack("<I", 1))
    open(path, "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
gguf(sys.argv[1] + "/bonsai-q4_0.gguf", True, 2)
gguf(sys.argv[1] + "/bonsai-ptq1_0.gguf", True, 143)
gguf(sys.argv[1] + "/plain.gguf", False, 2)
PY

# a backend that records how it was started, then answers /health
cat > "$scratch/backend.py" <<'PY'
#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
open(os.environ["RECORD"], "w").write(json.dumps({"argv": argv}))
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = b'{"status":"ok"}'
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
chmod +x "$scratch/backend.py"

refused=$("$bin" serve -m "$scratch/bonsai-q4_0.gguf" --device vulkan --port 1 --llama-server "$scratch/backend.py" 2>&1)
check "--device vulkan refuses a Hadamard-folded file" '[[ "$refused" == *"Hadamard-folded"* ]]'
refused=$("$bin" serve -m "$scratch/bonsai-ptq1_0.gguf" --device auto --port 1 --llama-server "$scratch/backend.py" 2>&1)
check "PrismML ternary types are refused with the converter's name" '[[ "$refused" == *"ternary_to_q4_0.py"* ]]'

run() {  # <model> <record>: serve with --device auto until the backend has recorded its start
    local port
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    RECORD="$2" "$bin" serve -m "$1" --device auto --port "$port" --llama-server "$scratch/backend.py" >"$2.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 100); do [ -s "$2" ] && break; sleep 0.1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}
field() { python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"; }

run "$scratch/bonsai-q4_0.gguf" "$scratch/b.json"
check "--device auto sends a Hadamard-folded file to HRX0" '[ "$(field "$scratch/b.json" "r[\"argv\"][r[\"argv\"].index(\"--device\")+1]")" = HRX0 ]'
run "$scratch/plain.gguf" "$scratch/plain.json"
check "a plain file still goes to Vulkan0" '[ "$(field "$scratch/plain.json" "r[\"argv\"][r[\"argv\"].index(\"--device\")+1]")" = Vulkan0 ]'

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
