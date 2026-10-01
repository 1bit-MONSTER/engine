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
# Which device `1bit serve` starts a plain .gguf on, without a GPU:
#   - --device cpu: llama-server with no GPU layers and no --device,
#   - --device auto: HRX0 in a build with HRX, Vulkan0 in one without (Vulkan is leaving the engine,
#     RFC #213; a build without HRX, like CI's, keeps it until the Vulkan build is removed),
#   - --device auto keeps Vulkan0 for an architecture HRX does not map (Qwen3.8-Flash-Next's
#     qwen4exp), for --mtp and for --mmproj,
#   - serve.cpp's hrx_missing_arch list matches registry/architectures.json.
#
# usage: tests/device_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: device_route.sh path/to/1bit}
scratch=$(mktemp -d)
trap 'kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; rm -rf "$scratch"' EXIT
pid=
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

python3 - "$scratch" <<'PY'
import struct, sys
def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
for name, arch in (("plain", "llama"), ("flashnext", "qwen4exp")):
    kv = s("general.architecture") + struct.pack("<I", 8) + s(arch)
    open(f"{sys.argv[1]}/{name}.gguf", "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + kv)
PY
: > "$scratch/head.gguf"

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

run() {  # <record> <model> [serve args]
    local port
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    RECORD="$1" "$bin" serve -m "$scratch/$2.gguf" --port "$port" --llama-server "$scratch/backend.py" "${@:3}" >"$1.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 100); do [ -s "$1" ] && break; sleep 0.1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}
field() { python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"; }
after() { field "$1" "r[\"argv\"][r[\"argv\"].index(\"$2\")+1] if \"$2\" in r[\"argv\"] else None"; }

run "$scratch/cpu.json" plain --device cpu
check "--device cpu runs no GPU layers" '[ "$(after "$scratch/cpu.json" -ngl)" = 0 ]'
check "  and names no device" '[ "$(after "$scratch/cpu.json" --device)" = None ]'

run "$scratch/auto.json" plain
expected=Vulkan0   # the build's CMake cache says whether it has HRX
if grep -q "^ONEBIT_HRX:BOOL=ON" "$(dirname "$bin")/CMakeCache.txt" 2>/dev/null; then expected=HRX0; fi
check "--device auto runs a plain .gguf on $expected (this build)" '[ "$(after "$scratch/auto.json" --device)" = "$expected" ]'

run "$scratch/flashnext.json" flashnext
check "--device auto keeps an architecture HRX does not map on Vulkan0" '[ "$(after "$scratch/flashnext.json" --device)" = Vulkan0 ]'
run "$scratch/mtp.json" plain --mtp "$scratch/head.gguf"
check "--device auto keeps --mtp on Vulkan0" '[ "$(after "$scratch/mtp.json" --device)" = Vulkan0 ]'
run "$scratch/mmproj.json" plain --mmproj "$scratch/head.gguf"
check "--device auto keeps --mmproj on Vulkan0" '[ "$(after "$scratch/mmproj.json" --device)" = Vulkan0 ]'

root=$(cd "$(dirname "$0")/.." && pwd)
check "hrx_missing_arch matches registry/architectures.json" 'python3 - "$root" <<'"'"'PY'"'"'
import json, re, sys
root = sys.argv[1]
a = json.load(open(root + "/registry/architectures.json"))["architectures"].values()
want = {v["gguf"] for v in a if "vulkan" in v["backends"] and "hrx" not in v["backends"]}
src = open(root + "/app/serve.cpp").read()
body = src[src.index("bool hrx_missing_arch"):]
body = body[body.index("{"):body.index("};")]
have = set(re.findall(r"\"([^\"]+)\"", body))
if want != have:
    print("registry only:", sorted(want - have), "serve.cpp only:", sorted(have - want))
    sys.exit(1)
PY'

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
