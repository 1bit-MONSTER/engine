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
# Which device `1bit serve` starts a plain .gguf on, without a GPU (RFC #213 stage 3: HRX only,
# no Vulkan, no ROCm):
#   - --device cpu: llama-server with no GPU layers and no --device,
#   - --device auto: HRX0 in a build with HRX, the CPU in one without (CI, Windows),
#   - --device auto with --mtp: the same route (HRX drafts on HRX0 now; it used to keep Vulkan),
#   - --device auto with --mmproj: the CPU, with a note (HRX vision is an open RFC #213 gate),
#   - an architecture our llama.cpp does not map (Qwen3.8-Flash-Next's qwen4exp, Zyphra Zamba2) is
#     refused with a pointer to Lemonade's llamacpp backend, under auto and when asked for by name,
#   - --moe-slots is refused as not in this build (it moves to HRX with Flash-Next),
#   - in a build with HRX, --parallel 4 on a gated delta-net model under auto is refused,
#   - --device vulkan / rocm and the removed flags (--lean, --adaptive, --long-model,
#     --prefill-device) are refused with the reason,
#   - serve.cpp's hrx_missing_arch list: no architecture in it is mapped to hrx by
#     registry/architectures.json (a pin bump that brings one in must drop it from the list).
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
for name, arch in (("plain", "llama"), ("flashnext", "qwen4exp"), ("zamba", "zamba2"), ("delta", "qwen35")):
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
# refused <expected text> <model> [serve args]: serve exits at once, non-zero, saying so
refused() {
    local out rc
    out=$(RECORD="$scratch/refused.json" timeout 20 "$bin" serve -m "$scratch/$2.gguf" --port 1 \
          --llama-server "$scratch/backend.py" "${@:3}" 2>&1)
    rc=$?
    [ "$rc" != 0 ] && [ "$rc" != 124 ] && [[ "$out" == *"$1"* ]] || { echo "     got (exit $rc): $out"; return 1; }
}

run "$scratch/cpu.json" plain --device cpu
check "--device cpu runs no GPU layers" '[ "$(after "$scratch/cpu.json" -ngl)" = 0 ]'
check "  and names no device" '[ "$(after "$scratch/cpu.json" --device)" = None ]'

hrx=0
if grep -q "^ONEBIT_HRX:BOOL=ON" "$(dirname "$bin")/CMakeCache.txt" 2>/dev/null; then hrx=1; fi
expected=None where="the CPU"   # the build's CMake cache says whether it has HRX
if [ $hrx = 1 ]; then expected=HRX0 where=HRX0; fi

run "$scratch/auto.json" plain
check "--device auto runs a plain .gguf on $where (this build)" '[ "$(after "$scratch/auto.json" --device)" = "$expected" ]'
run "$scratch/mtp.json" plain --mtp "$scratch/head.gguf"
check "--device auto runs --mtp on $where" '[ "$(after "$scratch/mtp.json" --device)" = "$expected" ] && [ "$(after "$scratch/mtp.json" -md)" = "$scratch/head.gguf" ]'
run "$scratch/mmproj.json" plain --mmproj "$scratch/head.gguf"
check "--device auto runs --mmproj on the CPU" '[ "$(after "$scratch/mmproj.json" -ngl)" = 0 ] && [ "$(after "$scratch/mmproj.json" --device)" = None ]'
if [ $hrx = 1 ]; then
    check "  and says why" 'grep -q "mmproj has no checked HRX route" "$scratch/mmproj.json.log"'
fi

check "an architecture our llama.cpp does not map (qwen4exp) is refused under auto" 'refused "Lemonade" flashnext'
check "  and on --device hrx" 'refused "architecture qwen4exp is not in" flashnext --device hrx'
check "  and on --device cpu (zamba2)" 'refused "architecture zamba2 is not in" zamba --device cpu'
check "--moe-slots is refused as not in this build" 'refused "--moe-slots is not available in this build" plain --moe-slots 64'
check "  under auto too, saying where it goes" 'refused "moves to HRX" plain --moe-slots auto --device auto'
if [ $hrx = 1 ]; then
    check "--parallel 4 on a gated delta-net model under auto is refused" 'refused "has no GPU route" delta --parallel 4'
fi
for dev in vulkan rocm; do
    check "--device $dev is refused as removed" 'refused "--device $dev was removed" plain --device $dev'
done
for flag in --lean --adaptive "--long-model x.gguf" "--prefill-device hrx"; do
    # shellcheck disable=SC2086
    check "$flag is refused as removed" 'refused "${flag%% *} was removed" plain $flag'
done

root=$(cd "$(dirname "$0")/.." && pwd)
check "no hrx_missing_arch architecture is mapped to hrx in registry/architectures.json" 'python3 - "$root" <<'"'"'PY'"'"'
import json, re, sys
root = sys.argv[1]
a = json.load(open(root + "/registry/architectures.json"))["architectures"].values()
on_hrx = {v["gguf"] for v in a if "hrx" in v["backends"]}
src = open(root + "/app/serve.cpp").read()
body = src[src.index("bool hrx_missing_arch"):]
body = body[body.index("{"):body.index("};")]
have = set(re.findall(r"\"([^\"]+)\"", body))
if not have or have & on_hrx:
    print("mapped to hrx now, drop from hrx_missing_arch:", sorted(have & on_hrx))
    sys.exit(1)
PY'

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
