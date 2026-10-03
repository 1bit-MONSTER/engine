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
# How `1bit serve` hands --parallel to llama-server, without a GPU:
#   - --parallel 1 starts it with -np 1 (llama-server's own default is several slots),
#   - no --parallel leaves -np out (llama-server's default),
#   - --parallel 4 starts it with -np 4.
#
# usage: tests/parallel_args.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: parallel_args.sh path/to/1bit}
scratch=$(mktemp -d)
trap 'kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; rm -rf "$scratch"' EXIT
pid=
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

python3 - "$scratch" <<'PY'
import struct, sys
def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
kv = [s("general.architecture") + struct.pack("<I", 8) + s("llama")]
open(sys.argv[1] + "/m.gguf", "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
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

run() {  # <record> [serve args]: serve until the backend has recorded its start
    local port
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    RECORD="$1" "$bin" serve -m "$scratch/m.gguf" --device cpu --port "$port" --llama-server "$scratch/backend.py" "${@:2}" >"$1.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 100); do [ -s "$1" ] && break; sleep 0.1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}
np() { python3 -c 'import json,sys; a=json.load(open(sys.argv[1]))["argv"]; print(a[a.index("-np")+1] if "-np" in a else "none")' "$1"; }

run "$scratch/one.json" --parallel 1
check "--parallel 1 starts llama-server with -np 1" '[ "$(np "$scratch/one.json")" = 1 ]'
run "$scratch/unset.json"
check "no --parallel leaves -np out" '[ "$(np "$scratch/unset.json")" = none ]'
run "$scratch/four.json" --parallel 4
check "--parallel 4 starts llama-server with -np 4" '[ "$(np "$scratch/four.json")" = 4 ]'

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
