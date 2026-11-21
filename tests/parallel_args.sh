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
def write(path, arch):
    kv = [s("general.architecture") + struct.pack("<I", 8) + s(arch)]
    open(path, "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
write(sys.argv[1] + "/m.gguf", "llama")
write(sys.argv[1] + "/gdn.gguf", "qwen35")   # a gated delta-net arch, for the engine#3125 arms
PY

# a backend that records how it was started, then answers /health
cat > "$scratch/backend.py" <<'PY'
#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
record = {"argv": argv, "hrx_disable": os.environ.get("GGML_HRX_DISABLE_DISPATCH")}
open(os.environ["RECORD"], "w").write(json.dumps(record))
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = b'{"status":"ok"}'
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
chmod +x "$scratch/backend.py"

run() {  # [--hrx] [--gdn] <record> [serve args]: serve until the backend has recorded its start
    local port model="$scratch/m.gguf" device=cpu
    if [ "${1:-}" = "--hrx" ]; then device=hrx; shift; fi
    if [ "${1:-}" = "--gdn" ]; then model="$scratch/gdn.gguf"; shift; fi
    # --port 0: the kernel picks and serve reports the port it bound, so nothing here picks a port and releases
    # it before serve binds (engine #1444/#1485). A bind failure can no longer be our own port collision.
    RECORD="$1" "$bin" serve -m "$model" --device "$device" --port 0 --llama-server "$scratch/backend.py" "${@:2}" >"$1.log" 2>&1 &
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

# engine#3125: the unasked gated delta-net case is no longer clamped to one slot, so it takes llama-server's own slot
# count and therefore needs the shared-KV treatment the explicit --parallel path gets. `serve` does not open a device on
# this path -- it names the HSA runtime it would dlopen and reads the GTT counters from sysfs, then forks the child -- so
# these arms stay device-free and run in the same row as the three above.
kv() { python3 -c 'import json,sys; a=json.load(open(sys.argv[1]))["argv"]; print("yes" if "-kvu" in a else "no")' "$1"; }
disable() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hrx_disable"] or "")' "$1"; }
run --hrx --gdn "$scratch/gdn_unset.json"
check "engine#3125: an unasked qwen35 server asks for 4 slots (-np 4), it is not left at one" '[ "$(np "$scratch/gdn_unset.json")" = 4 ]'
check "engine#3125: ...which is the measured configuration, so it gets -kvu" '[ "$(kv "$scratch/gdn_unset.json")" = yes ]'
check "engine#3125: ...and qwen.attention off under that unified cache" 'case "$(disable "$scratch/gdn_unset.json")" in *qwen.attention*) true;; *) false;; esac'
run --hrx --gdn "$scratch/gdn_four.json" --parallel 4
check "engine#3125: --parallel 4 on qwen35 still passes -np 4 and -kvu" '[ "$(np "$scratch/gdn_four.json")" = 4 ] && [ "$(kv "$scratch/gdn_four.json")" = yes ]'
run --hrx "$scratch/llama_unset.json"
check "engine#3125: a non-delta-net arch is untouched (no -kvu, no -np)" '[ "$(kv "$scratch/llama_unset.json")" = no ] && [ "$(np "$scratch/llama_unset.json")" = none ]'

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
