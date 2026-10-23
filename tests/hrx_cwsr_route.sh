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
# engine#315 candidate 1 end to end, without a GPU: what `1bit serve --hrx-cwsr-placement` puts in an
# HRX backend's environment (app/hrx_cwsr.h, docs/hrx.md "CWSR save-area placement"). A fake
# llama-server records its environment.
#   - off: none of HSA_USE_SVM, HSA_USERPTR_FOR_PAGED_MEM, HRX_ALLOW_NO_SVM, and serve says so;
#   - on, in a build whose hrx-system has the gate: all three; a value the user set wins;
#   - on, in a build without the gate: refused at start, with the reason;
#   - auto on a CWSR-on box: all three with the gate, none without it; with one of the three set
#     by the user, none, and serve says so;
#   - auto elsewhere (CWSR off, or no amdgpu, as in CI): none;
#   - --device cpu: never any of them.
#
# usage: tests/hrx_cwsr_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: hrx_cwsr_route.sh path/to/1bit}
scratch=$(mktemp -d)
trap 'kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; rm -rf "$scratch"' EXIT
pid=
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
unset HSA_USE_SVM HSA_USERPTR_FOR_PAGED_MEM HRX_ALLOW_NO_SVM

python3 - "$scratch" <<'PY'
import struct, sys
def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
kv = s("general.architecture") + struct.pack("<I", 8) + s("llama")
open(f"{sys.argv[1]}/plain.gguf", "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + kv)
PY

cat > "$scratch/backend.py" <<'PY'
#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
keys = ("HSA_USE_SVM", "HSA_USERPTR_FOR_PAGED_MEM", "HRX_ALLOW_NO_SVM")
open(os.environ["RECORD"], "w").write(json.dumps({k: os.environ[k] for k in keys if k in os.environ}))
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = b'{"status":"ok"}'
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
chmod +x "$scratch/backend.py"

run() {  # <record> [serve args]: the backend's three variables, as JSON, in <record>; serve's log in <record>.log
    local port
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    RECORD="$1" "$bin" serve -m "$scratch/plain.gguf" --port "$port" --llama-server "$scratch/backend.py" \
        --no-mem-check "${@:2}" >"$1.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 100); do [ -s "$1" ] && break; kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}
env_is() { [ "$(cat "$1" 2>/dev/null)" = "$2" ]; }
all='{"HSA_USE_SVM": "0", "HSA_USERPTR_FOR_PAGED_MEM": "0", "HRX_ALLOW_NO_SVM": "1"}'

run "$scratch/off.json" --device hrx --hrx-cwsr-placement off
check "off: the HRX backend gets none of the three" 'env_is "$scratch/off.json" "{}"'
check "  and serve says so" 'grep -q "hrx-cwsr-placement=off; CWSR save-area placement left to ROCr (engine#315)" "$scratch/off.json.log"'

run "$scratch/on.json" --device hrx --hrx-cwsr-placement=on
gate=1
grep -q "no HRX_ALLOW_NO_SVM gate" "$scratch/on.json.log" && gate=0
if [ $gate = 0 ]; then
    echo "     (this build's hrx-system has no HRX_ALLOW_NO_SVM gate)"
    check "on without the gate: refused, no backend started" '[ ! -s "$scratch/on.json" ]'
else
    check "on: the HRX backend gets all three" 'env_is "$scratch/on.json" "$all"'
    check "  and serve logs the GTT line" 'grep -q "hrx-cwsr-placement=on; CWSR save area placed in GTT (engine#315)" "$scratch/on.json.log"'
    HSA_USERPTR_FOR_PAGED_MEM=5 run "$scratch/on-user.json" --device hrx --hrx-cwsr-placement on
    check "on: a value the user set wins" \
        'env_is "$scratch/on-user.json" "{\"HSA_USE_SVM\": \"0\", \"HSA_USERPTR_FOR_PAGED_MEM\": \"5\", \"HRX_ALLOW_NO_SVM\": \"1\"}"'
    run "$scratch/cpu.json" --device cpu --hrx-cwsr-placement on
    check "--device cpu: none of the three, even with on" 'env_is "$scratch/cpu.json" "{}"'
fi

HSA_USE_SVM=1 run "$scratch/auto-user.json" --device hrx
check "auto with HSA_USE_SVM set by the user: serve adds nothing" 'env_is "$scratch/auto-user.json" "{\"HSA_USE_SVM\": \"1\"}"'
if [ "$(cat /sys/module/amdgpu/parameters/cwsr_enable 2>/dev/null)" = 1 ]; then
    check "  and says the environment decides (CWSR on here)" 'grep -q "HSA_USE_SVM set in the environment" "$scratch/auto-user.json.log"'
    run "$scratch/auto-on.json" --device hrx
    if [ $gate = 1 ]; then
        check "auto, CWSR on here: the HRX backend gets all three" 'env_is "$scratch/auto-on.json" "$all"'
        check "  and serve logs the GTT line" 'grep -q "HRX: CWSR is on; CWSR save area placed in GTT (engine#315)" "$scratch/auto-on.json.log"'
    else
        check "auto, CWSR on here, no gate: none of the three" 'env_is "$scratch/auto-on.json" "{}"'
        check "  and serve says the gate is missing" 'grep -q "no HRX_ALLOW_NO_SVM gate; placement unchanged" "$scratch/auto-on.json.log"'
    fi
else
    run "$scratch/auto.json" --device hrx
    check "auto, CWSR not on here: the HRX backend gets none of the three" 'env_is "$scratch/auto.json" "{}"'
    check "  and serve says placement is unchanged" 'grep -q "placement unchanged (engine#315)" "$scratch/auto.json.log"'
fi

exit $fail
