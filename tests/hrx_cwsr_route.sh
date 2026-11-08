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
#   - off: refused at start with the recipe (engine#715) when the trio would be absent, unless
#     ONEBIT_HRX_NO_TRIO_OK=1 acknowledges it, in which case none of the three and serve says so; the refusal keys on
#     the opt-out, not on an empty trio, so a build without the gate is never refused;
#   - on, in a build whose hrx-system has the gate: all three; a value the user set wins;
#   - on, in a build without the gate: refused at start, with the reason;
#   - auto (the default) in a build with the gate: all three, whatever cwsr_enable says, and
#     HRX_SVM=1 turns that off again; a value the user set still wins;
#   - auto in a build without the gate: refused the same way (no gate means no trio), acknowledged -> none;
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
    local attempt port
    for attempt in 1 2 3; do
        port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
        RECORD="$1" "$bin" serve -m "$scratch/plain.gguf" --port "$port" --llama-server "$scratch/backend.py" \
            --no-mem-check "${@:2}" >"$1.log" 2>&1 &
        pid=$!
        for _ in $(seq 1 100); do [ -s "$1" ] && break; kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
        kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        if ! grep -qiE 'EADDRINUSE|address already in use|cannot listen on' "$1.log"; then break; fi
        if [ "$attempt" -ge 3 ]; then break; fi
            # A server that never came up because it could not bind is a port collision, not a defect in what
            # this test checks: serve picks the port, releases it, and launches its backends before it binds, so on
            # a -j runner another serve can take it inside that window. Re-pick a bounded number of times; the FAIL
            # message below stays for a collision that will not go away (engine #1444).
        echo "     note: port $port was taken before serve bound it; retrying with a fresh port (attempt $((attempt + 1))/3)"
    done
    if grep -qiE 'EADDRINUSE|address already in use|cannot listen on' "$1.log"; then
        echo "FAIL: hrx_cwsr_route: could not bind a port $port ($(grep -m1 -oiE 'EADDRINUSE|address already in use|cannot listen on [^ ]*' "$1.log" | head -1))"
        exit 1
    fi
}
env_is() { [ "$(cat "$1" 2>/dev/null)" = "$2" ]; }
all='{"HSA_USE_SVM": "0", "HSA_USERPTR_FOR_PAGED_MEM": "0", "HRX_ALLOW_NO_SVM": "1"}'

# engine#715: an HRX route without the trio is refused, with the recipe a human can paste
run "$scratch/off.json" --device hrx --hrx-cwsr-placement off
check "off: an HRX route is refused, no backend started" '[ ! -s "$scratch/off.json" ]'
check "  and the refusal carries the recipe" \
    'grep -q "HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1" "$scratch/off.json.log"'
check "  and names the acknowledgement" 'grep -q "ONEBIT_HRX_NO_TRIO_OK=1" "$scratch/off.json.log"'
ONEBIT_HRX_NO_TRIO_OK=1 run "$scratch/off-ack.json" --device hrx --hrx-cwsr-placement off
check "off + ONEBIT_HRX_NO_TRIO_OK=1: the HRX backend gets none of the three" 'env_is "$scratch/off-ack.json" "{}"'
check "  and serve says so" \
    'grep -q "hrx-cwsr-placement=off; CWSR save-area placement left to ROCr (engine#315)" "$scratch/off-ack.json.log"'
# --device cpu never initialises HRX, so it is never refused
run "$scratch/cpu-off.json" --device cpu --hrx-cwsr-placement off
check "--device cpu with off: not refused, no variables" 'env_is "$scratch/cpu-off.json" "{}"'

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

HSA_USERPTR_FOR_PAGED_MEM=5 run "$scratch/auto-user.json" --device hrx
if [ $gate = 1 ]; then
    check "auto: a value the user set wins, the other two are still added" \
        'env_is "$scratch/auto-user.json" "{\"HSA_USE_SVM\": \"0\", \"HSA_USERPTR_FOR_PAGED_MEM\": \"5\", \"HRX_ALLOW_NO_SVM\": \"1\"}"'
    check "  and serve names it in the log" 'grep -q "kept from the environment: HSA_USERPTR_FOR_PAGED_MEM" "$scratch/auto-user.json.log"'
    run "$scratch/auto.json" --device hrx
    check "auto (the default): the HRX backend gets all three" 'env_is "$scratch/auto.json" "$all"'
    check "  and serve logs the GTT default line" 'grep -q "CWSR save area placed in GTT by default" "$scratch/auto.json.log"'
    HRX_SVM=1 run "$scratch/auto-off.json" --device hrx
    check "auto with HRX_SVM=1: refused (the opt-out has to be acknowledged)" '[ ! -s "$scratch/auto-off.json" ]'
    check "  and the refusal names HRX_SVM" 'grep -q "HRX_SVM=1" "$scratch/auto-off.json.log"'
    HRX_SVM=1 ONEBIT_HRX_NO_TRIO_OK=1 run "$scratch/auto-off-ack.json" --device hrx
    check "auto with HRX_SVM=1 + acknowledgement: none of the three (the opt-out)" 'env_is "$scratch/auto-off-ack.json" "{}"'
    check "  and serve says HRX_SVM=1 decided it" 'grep -q "HRX_SVM=1; CWSR save-area placement left to ROCr" "$scratch/auto-off-ack.json.log"'
    # engine#715: serve also puts the trio in its *own* environment so a child that inherits it gets it. That is not
    # observable through this fake backend (serve passes an explicit envp), and it must not add a line here: the
    # memory refusal counts lines (tests/mem_admit_route.sh). It is covered by tests/hrx_cwsr_test.cpp instead.
    check "the placement decision is announced exactly once" \
        '[ "$(grep -c "CWSR save area placed in GTT" "$scratch/auto.json.log")" = 1 ]'
else
    run "$scratch/auto.json" --device hrx
    check "auto, no gate: none of the three" 'env_is "$scratch/auto.json" "{}"'
    check "  and serve says the gate is missing" 'grep -q "no HRX_ALLOW_NO_SVM gate; CWSR save-area placement unchanged" "$scratch/auto.json.log"'
fi

exit $fail
