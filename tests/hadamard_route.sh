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
#     GGML_W4A4_TENSORS=all in the backend's environment; a MoE file (an expert count) also gets
#     1024-token micro-batches (-ub 1024), a dense one keeps llama-server's 512,
#   - an unstamped file with --device auto still goes to Vulkan with neither variable,
#   - --dflash on a rotated file adds the DFlash drafter (p-min 0.4 from its rocm recipe, n-max = the
#     drafter's dflash.block_size - 1, or 16 when the file does not say): one ROCm server, W4A4
#     prompt processing and DFlash2 decode,
#   - a drafter with unrotated Q4_0 tensors is refused next to a rotated file (the rotation is
#     process-wide), one with Q8_0 tensors is not.
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
def gguf(path, stamp, arch="llama", ints={}, tensors=()):
    kv = []
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv.append(s("general.architecture") + struct.pack("<I", 8) + s(arch))
    if stamp:
        kv.append(s("onebit.hadamard_q4_0") + struct.pack("<I", 5) + struct.pack("<i", 32))
    for k, v in ints.items():
        kv.append(s(k) + struct.pack("<I", 4) + struct.pack("<I", v))
    ti = b"".join(s(name) + struct.pack("<I", 1) + struct.pack("<Q", 32) + struct.pack("<I", t) + struct.pack("<Q", 0)
                  for name, t in tensors)
    open(path, "wb").write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv) + ti)
gguf(sys.argv[1] + "/h32.gguf", True)
gguf(sys.argv[1] + "/h32moe.gguf", True, "qwen3moe", {"qwen3moe.expert_count": 128})
gguf(sys.argv[1] + "/plain.gguf", False)
gguf(sys.argv[1] + "/draft.gguf", False)
gguf(sys.argv[1] + "/draft8.gguf", False, "dflash", {"dflash.block_size": 8})
gguf(sys.argv[1] + "/draftq4.gguf", False, "dflash", {"dflash.block_size": 8}, [("blk.0.ffn_up.weight", 2)])
gguf(sys.argv[1] + "/draftq8.gguf", False, "dflash", {"dflash.block_size": 8}, [("blk.0.ffn_up.weight", 8)])
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

run() {  # <model> <record> [serve args]: serve with --device auto until the backend has recorded its start
    local port
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    env -u GGML_Q4_0_HADAMARD -u GGML_W4A4_TENSORS RECORD="$2" "$bin" serve -m "$1" --device auto --port "$port" \
        --llama-server "$scratch/backend.py" "${@:3}" >"$2.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 100); do [ -s "$2" ] && break; sleep 0.1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}
field() { python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"; }

run "$scratch/h32.gguf" "$scratch/h32.json"
check "--device auto sends a rotated file to ROCm0" '[ "$(field "$scratch/h32.json" "r[\"argv\"][r[\"argv\"].index(\"--device\")+1]")" = ROCm0 ]'
check "  with GGML_Q4_0_HADAMARD=1" '[ "$(field "$scratch/h32.json" "r[\"env\"].get(\"GGML_Q4_0_HADAMARD\")")" = 1 ]'
check "  and GGML_W4A4_TENSORS=all" '[ "$(field "$scratch/h32.json" "r[\"env\"].get(\"GGML_W4A4_TENSORS\")")" = all ]'
check "  dense: llama-server's own micro-batch" '[ "$(field "$scratch/h32.json" "\"-ub\" in r[\"argv\"]")" = False ]'

run "$scratch/h32moe.gguf" "$scratch/h32moe.json"
check "a rotated MoE file gets 1024-token micro-batches" '[ "$(field "$scratch/h32moe.json" "r[\"argv\"][r[\"argv\"].index(\"-ub\")+1]")" = 1024 ]'

run "$scratch/plain.gguf" "$scratch/plain.json"
check "an unstamped file still goes to Vulkan0" '[ "$(field "$scratch/plain.json" "r[\"argv\"][r[\"argv\"].index(\"--device\")+1]")" = Vulkan0 ]'
check "  with neither variable" '[ "$(field "$scratch/plain.json" "sorted(r[\"env\"])")" = "[]" ]'

run "$scratch/h32.gguf" "$scratch/df.json" --dflash "$scratch/draft.gguf"
after() { field "$scratch/df.json" "r[\"argv\"][r[\"argv\"].index(\"$1\")+1]"; }
check "--dflash on a rotated file stays on ROCm0" '[ "$(after --device)" = ROCm0 ]'
check "  with the DFlash drafter" '[ "$(after --spec-type)" = draft-dflash ] && [ "$(after -md)" = "$scratch/draft.gguf" ]'
check "  n-max 16 without a block size, p-min 0.4" '[ "$(after --spec-draft-n-max)" = 16 ] && [ "$(after --spec-draft-p-min)" = 0.4 ]'
check "  and the Hadamard W4A4 environment" '[ "$(field "$scratch/df.json" "r[\"env\"].get(\"GGML_W4A4_TENSORS\")")" = all ]'

run "$scratch/h32.gguf" "$scratch/df8.json" --dflash "$scratch/draft8.gguf"
check "--dflash with a block-8 drafter drafts 7" '[ "$(field "$scratch/df8.json" "r[\"argv\"][r[\"argv\"].index(\"--spec-draft-n-max\")+1]")" = 7 ]'

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
q4=$(timeout 20 "$bin" serve -m "$scratch/h32.gguf" --device auto --port "$port" --llama-server "$scratch/backend.py" \
     --dflash "$scratch/draftq4.gguf" 2>&1)
check "a Q4_0 drafter next to a rotated file is refused" '[[ "$q4" == *"not Hadamard-rotated"* ]]'
run "$scratch/h32.gguf" "$scratch/dfq8.json" --dflash "$scratch/draftq8.gguf"
check "  a Q8_0 drafter is not" '[ "$(field "$scratch/dfq8.json" "r[\"argv\"][r[\"argv\"].index(\"-md\")+1]")" = "$scratch/draftq8.gguf" ]'

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
