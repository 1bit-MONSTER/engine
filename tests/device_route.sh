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
#   - --device auto with --mmproj: the same route as a plain file (HRX0 is unverified for vision,
#     an open RFC #213 gate to check on ZAYA1-VL),
#   - an architecture our llama.cpp does not build (not in registry/architectures.json's
#     gguf_architectures.hrx: Qwen3.8-Flash-Next's qwen4exp, Zyphra Zamba2, a made-up one) is
#     refused with a pointer to Lemonade's llamacpp backend, under auto and when asked for by name,
#   - --moe-slots is refused as not in this build (it moves to HRX with Flash-Next),
#   - --parallel 4 on a gated delta-net model: in a build with HRX, auto serves it on HRX0 with one
#     slot and says why; --device cpu keeps -np 4,
#   - an H32 file (onebit.hadamard_q4_0, a dropped format) is refused by name,
#   - --device vulkan / rocm / zinc and the removed flags (--lean, --adaptive, --long-model,
#     --prefill-device, --zinc) are refused with the reason.
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
for name, arch, extra in (("plain", "llama", b""), ("flashnext", "qwen4exp", b""), ("zamba", "zamba2", b""),
                          ("madeup", "not-an-arch", b""), ("delta", "qwen35", b""),
                          ("h32", "qwen35", s("onebit.hadamard_q4_0") + struct.pack("<I", 5) + struct.pack("<i", 32))):
    kv = s("general.architecture") + struct.pack("<I", 8) + s(arch) + extra
    open(f"{sys.argv[1]}/{name}.gguf", "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, 2 if extra else 1) + kv)
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
check "--device auto runs --mmproj on $where" '[ "$(after "$scratch/mmproj.json" --device)" = "$expected" ] && [ "$(after "$scratch/mmproj.json" --mmproj)" = "$scratch/head.gguf" ]'

check "an architecture our llama.cpp does not map (qwen4exp) is refused under auto" 'refused "Lemonade" flashnext'
check "  and on --device hrx" 'refused "architecture qwen4exp is not in" flashnext --device hrx'
check "  and on --device cpu (zamba2)" 'refused "architecture zamba2 is not in" zamba --device cpu'
check "  and any architecture the registry does not list" 'refused "architecture not-an-arch is not in" madeup'
check "an H32 file is refused by name" 'refused "H32 file" h32'
check "  on --device hrx too" 'refused "dropped that format" h32 --device hrx'
check "--moe-slots is refused as not in this build" 'refused "--moe-slots is not available in this build" plain --moe-slots 64'
check "  under auto too, saying where it goes" 'refused "moves to HRX" plain --moe-slots auto --device auto'
if [ $hrx = 1 ]; then
    run "$scratch/delta.json" delta --parallel 4
    check "--parallel 4 on a gated delta-net model under auto: HRX0 with one slot" '[ "$(after "$scratch/delta.json" --device)" = HRX0 ] && [ "$(after "$scratch/delta.json" -np)" = 1 ]'
    check "  and a warning that says why" 'grep -q "one gated delta-net sequence at a time" "$scratch/delta.json.log"'
fi
run "$scratch/deltacpu.json" delta --parallel 4 --device cpu
check "--device cpu keeps --parallel 4 on a gated delta-net model" '[ "$(after "$scratch/deltacpu.json" -np)" = 4 ]'
for dev in vulkan rocm zinc; do
    check "--device $dev is refused as removed" 'refused "--device $dev was removed" plain --device $dev'
done
for flag in --lean --adaptive "--long-model x.gguf" "--prefill-device hrx" "--zinc x"; do
    # shellcheck disable=SC2086
    check "$flag is refused as removed" 'refused "${flag%% *} was removed" plain $flag'
done

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
