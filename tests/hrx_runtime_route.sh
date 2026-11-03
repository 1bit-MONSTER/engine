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
# engine#713 criterion 4 end to end, without a GPU: serve names the HSA runtime HRX will dlopen - path
# and sha256 (app/hrx_runtime.h) - once, at the first HRX backend, and that is the same path it hands
# the backend as IREE_HAL_AMDGPU_LIBHSA_PATH. A fake runtime file and a fake llama-server stand in for
# the real pair, so the test needs neither ROCm nor a device:
#   - --hrx-libhsa FILE: the line carries FILE and sha256sum(FILE), the backend's env carries FILE;
#   - announced exactly once, whatever else the route prints;
#   - a FILE that cannot be read: still named, with no digest, instead of a silent success;
#   - --device cpu: no runtime line at all (HRX is never initialised).
#
# usage: tests/hrx_runtime_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: hrx_runtime_route.sh path/to/1bit}
scratch=$(mktemp -d)
trap 'kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; rm -rf "$scratch"' EXIT
pid=
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
unset HSA_USE_SVM HSA_USERPTR_FOR_PAGED_MEM HRX_ALLOW_NO_SVM IREE_HAL_AMDGPU_LIBHSA_PATH

# A GGUF serve accepts without a tokenizer or weights (the cwsr route test's fixture).
python3 - "$scratch" <<'PY'
import struct, sys
def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
kv = s("general.architecture") + struct.pack("<I", 8) + s("llama")
open(f"{sys.argv[1]}/plain.gguf", "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + kv)
PY

# The runtime under test: content nobody else has, so the digest can only come from this file.
printf 'not really an HSA runtime\n' > "$scratch/fake-libhsa-runtime64.so.1"
want=$(sha256sum "$scratch/fake-libhsa-runtime64.so.1" | cut -d' ' -f1)

cat > "$scratch/backend.py" <<'PY'
#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
open(os.environ["RECORD"], "w").write(json.dumps({"libhsa": os.environ.get("IREE_HAL_AMDGPU_LIBHSA_PATH", "")}))
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = b'{"status":"ok"}'
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
chmod +x "$scratch/backend.py"

run() {  # <record> [serve args]: the backend's env, as JSON, in <record>; serve's log in <record>.log
    local port
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    RECORD="$1" "$bin" serve -m "$scratch/plain.gguf" --port "$port" --llama-server "$scratch/backend.py" \
        --no-mem-check "${@:2}" >"$1.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 100); do [ -s "$1" ] && break; kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}
env_is() { [ "$(cat "$1" 2>/dev/null)" = "$2" ]; }

# --hrx-cwsr-placement off with the acknowledgement keeps this test out of engine#715's refusal, in
# builds with and without the HRX_ALLOW_NO_SVM gate; it is the CWSR variable test, not this one.
ONEBIT_HRX_NO_TRIO_OK=1 run "$scratch/hrx.json" --device hrx --hrx-cwsr-placement off \
    --hrx-libhsa "$scratch/fake-libhsa-runtime64.so.1"
check "hrx: the backend gets the runtime serve named" \
    'env_is "$scratch/hrx.json" "{\"libhsa\": \"$scratch/fake-libhsa-runtime64.so.1\"}"'
check "hrx: serve names the runtime and its sha256" \
    'grep -qx "1bit serve: HRX: HSA runtime $scratch/fake-libhsa-runtime64.so.1 sha256 $want" "$scratch/hrx.json.log"'
check "hrx: the digest is the file's, checked with sha256sum" \
    'grep -q "sha256 $want\$" "$scratch/hrx.json.log"'
check "hrx: announced exactly once" \
    '[ "$(grep -c "HRX: HSA runtime " "$scratch/hrx.json.log")" = 1 ]'

# A runtime path that cannot be read is still named, without a digest: HRX will try to dlopen it and
# the log has to say which file that was.
ONEBIT_HRX_NO_TRIO_OK=1 run "$scratch/missing.json" --device hrx --hrx-cwsr-placement off \
    --hrx-libhsa "$scratch/not-there/libhsa-runtime64.so.1"
check "missing: the line names the path and says it is unreadable" \
    'grep -qx "1bit serve: HRX: HSA runtime $scratch/not-there/libhsa-runtime64.so.1 (unreadable: no sha256, HRX may fail to open it)" "$scratch/missing.json.log"'
check "missing: no fabricated digest" '! grep -Eq "sha256 [0-9a-f]{64}" "$scratch/missing.json.log"'

# The CPU route never initialises HRX, so it must not claim a runtime.
run "$scratch/cpu.json" --device cpu
check "--device cpu: no runtime line" '! grep -q "HRX: HSA runtime" "$scratch/cpu.json.log"'
check "--device cpu: no IREE_HAL_AMDGPU_LIBHSA_PATH in the backend's env" \
    'env_is "$scratch/cpu.json" "{\"libhsa\": \"\"}"'

exit $fail
