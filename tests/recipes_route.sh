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
# How `1bit serve` applies recipes (config/recipes.json, docs/recipes.md), without a GPU:
#   - the built-in recipes: a rotated MoE file gets -ub 1024 and says so on stderr, a rotated
#     dense file does not, --dflash gets --spec-draft-p-min 0.4 on rocm (the rocm recipe is listed
#     first, so it wins over dflash-p-min-0) and 0 on vulkan,
#   - a flag serve already set wins (--mtp-p-min 0.5 stays the only p-min),
#   - --no-recipes adds nothing, --recipes FILE replaces the built-in set (flags and environment),
#   - a malformed recipe file stops serve with the reason.
#
# usage: tests/recipes_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: recipes_route.sh path/to/1bit}
scratch=$(mktemp -d)
trap 'kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; rm -rf "$scratch"' EXIT
pid=
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

python3 - "$scratch" <<'PY'
import struct, sys
def gguf(path, stamp, arch="llama", ints={}):
    kv = []
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv.append(s("general.architecture") + struct.pack("<I", 8) + s(arch))
    if stamp:
        kv.append(s("onebit.hadamard_q4_0") + struct.pack("<I", 5) + struct.pack("<i", 32))
    for k, v in ints.items():
        kv.append(s(k) + struct.pack("<I", 4) + struct.pack("<I", v))
    open(path, "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
d = sys.argv[1]
gguf(d + "/dense.gguf", True)
gguf(d + "/moe.gguf", True, "qwen3moe", {"qwen3moe.expert_count": 128})
gguf(d + "/plain.gguf", False)
gguf(d + "/draft.gguf", False, "dflash", {"dflash.block_size": 8})
open(d + "/mine.json", "w").write("""{"recipes": [{"id": "mine", "match": {"architecture": ["llama"]},
  "args": ["--threads", "7"], "env": {"ONEBIT_RECIPE_TEST": "1"}, "measured": "test"}]}""")
open(d + "/bad.json", "w").write("""{"recipes": [{"id": "bad", "match": {"archtecture": ["llama"]},
  "args": ["--threads", "7"], "measured": "test"}]}""")
PY

# a backend that records how it was started, then answers /health
cat > "$scratch/backend.py" <<'PY'
#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
env = {k: v for k, v in os.environ.items() if k.startswith("GGML_") or k.startswith("ONEBIT_RECIPE")}
open(os.environ["RECORD"], "w").write(json.dumps({"argv": argv, "env": env}))
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = b'{"status":"ok"}'
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
chmod +x "$scratch/backend.py"

run() {  # <model> <record> [serve args]: serve with --device auto until the backend has recorded its start
    local port
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    env -u GGML_Q4_0_HADAMARD -u GGML_W4A4_TENSORS -u ONEBIT_RECIPE_TEST RECORD="$2" "$bin" serve -m "$1" --device auto \
        --port "$port" --llama-server "$scratch/backend.py" "${@:3}" >"$2.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 100); do [ -s "$2" ] && break; sleep 0.1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}
field() { python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"; }
after() { field "$1" "r[\"argv\"][r[\"argv\"].index(\"$2\")+1] if \"$2\" in r[\"argv\"] else None"; }
count() { field "$1" "r[\"argv\"].count(\"$2\")"; }

run "$scratch/moe.gguf" "$scratch/moe.json"
check "a rotated MoE file gets -ub 1024 from rotated-moe-ub1024" '[ "$(after "$scratch/moe.json" -ub)" = 1024 ]'
check "  and serve names the recipe on stderr" 'grep -q "recipe rotated-moe-ub1024: -ub 1024" "$scratch/moe.json.log"'

run "$scratch/dense.gguf" "$scratch/dense.json"
check "a rotated dense file keeps llama-server's micro-batch" '[ "$(after "$scratch/dense.json" -ub)" = None ]'

run "$scratch/moe.gguf" "$scratch/none.json" --no-recipes
check "--no-recipes adds nothing" '[ "$(after "$scratch/none.json" -ub)" = None ] && ! grep -q recipe "$scratch/none.json.log"'

run "$scratch/dense.gguf" "$scratch/df.json" --dflash "$scratch/draft.gguf"
check "--dflash on rocm gets --spec-draft-p-min 0.4 from rocm-dflash-p-min-0.4" '[ "$(after "$scratch/df.json" --spec-draft-p-min)" = 0.4 ] && [ "$(count "$scratch/df.json" --spec-draft-p-min)" = 1 ]'

run "$scratch/plain.gguf" "$scratch/dfv.json" --dflash "$scratch/draft.gguf"
check "--dflash on vulkan gets --spec-draft-p-min 0 from dflash-p-min-0" '[ "$(after "$scratch/dfv.json" --spec-draft-p-min)" = 0 ]'

run "$scratch/dense.gguf" "$scratch/dfp.json" --dflash "$scratch/draft.gguf" --mtp-p-min 0.5
check "a flag serve set wins: --mtp-p-min 0.5 is the only p-min" '[ "$(after "$scratch/dfp.json" --spec-draft-p-min)" = 0.5 ] && [ "$(count "$scratch/dfp.json" --spec-draft-p-min)" = 1 ]'

run "$scratch/plain.gguf" "$scratch/mine-run.json" --recipes "$scratch/mine.json"
check "--recipes FILE adds its flags" '[ "$(after "$scratch/mine-run.json" --threads)" = 7 ]'
check "  and its environment" '[ "$(field "$scratch/mine-run.json" "r[\"env\"].get(\"ONEBIT_RECIPE_TEST\")")" = 1 ]'

run "$scratch/moe.gguf" "$scratch/repl.json" --recipes "$scratch/mine.json"
check "  and replaces the built-in set" '[ "$(after "$scratch/repl.json" -ub)" = None ]'

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
bad=$(timeout 20 "$bin" serve -m "$scratch/plain.gguf" --device auto --port "$port" --llama-server "$scratch/backend.py" \
      --recipes "$scratch/bad.json" 2>&1)
check "a malformed recipe file stops serve with the reason" '[[ "$bad" == *"unknown match key \"archtecture\""* ]]'

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
