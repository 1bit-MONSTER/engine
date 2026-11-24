#!/bin/sh
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
# engine#3478: a held turn waits inside Lemonade's load path with NO bytes on the wire until the hold ends, and the
# appliance page's fetch has no timeout of its own (embed/opt/1bit/webapp/index.html:232) - so 1bit-webapp-proxy's
# upstream socket timeout is the only deadline a held turn has. It was a hardcoded 900 s while 1bit-heavy's default
# hold window is 3600 s, so a hold between 15 and 60 minutes cut the page's turn with a 502 that reads as "Lemonade is
# not answering" while the hold was healthy.
#
# Two arms, both of which FAIL on the pre-fix tree (no derived bound; the 900 s socket swallows a stalled upstream):
#   1. the shipped values: the proxy's hold window is 1bit-heavy's --max-seconds default, and the upstream timeout is
#      that window plus the old 900 s resume - so the two files carry ONE number and CI fails if they drift;
#   2. a fake Lemonade that sends nothing for a while, the way a hold does. A silent upstream inside the bound is
#      forwarded and answered; one past the bound answers 502 AT the bound, not at the stall. The shipped 4500 s bound
#      cannot be waited, so the arms scale it through the proxy's own ONEBIT_WEBAPP_UPSTREAM_TIMEOUT and stay a few
#      seconds long.
# No root, no device, no real Lemonade.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
proxy=$here/os/v3/puppy/embed/usr/local/bin/1bit-webapp-proxy
heavy=$here/os/v3/puppy/embed/usr/local/bin/1bit-heavy
t=$(mktemp -d) || exit 1
bad=0
srv=
proxy_pid=
cleanup() { [ -n "$proxy_pid" ] && kill "$proxy_pid" 2> /dev/null; [ -n "$srv" ] && kill "$srv" 2> /dev/null; rm -rf "${t:?}"; }
trap cleanup EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
command -v python3 > /dev/null && command -v curl > /dev/null || {
    echo "webapp_proxy_hold_timeout_test: needs python3 and curl"; exit 1; }

# 1. the shipped values: one hold window, and the proxy's bound derived from it (plus the 900 s resume it always allowed
#    for a load and a first token). On the pre-fix tree this import finds no HOLD_WINDOW and the failure names the
#    hardcoded 900 s that was shorter than the 3600 s window.
window=$(sed -n 's/.*maxs=\([0-9][0-9]*\).*/\1/p' "$heavy" | head -n 1)
[ -n "$window" ]; check $? "1bit-heavy carries the hold window (maxs=${window:-?})"
vals=$(PYTHONDONTWRITEBYTECODE=1 ONEBIT_WEBAPP_UPSTREAM_TIMEOUT= python3 - "$proxy" <<'PY' 2> /dev/null
import importlib.util, sys
from importlib.machinery import SourceFileLoader
loader = SourceFileLoader("webapp_proxy", sys.argv[1])   # the proxy has no .py suffix
spec = importlib.util.spec_from_loader("webapp_proxy", loader)
m = importlib.util.module_from_spec(spec)
loader.exec_module(m)
print(m.HOLD_WINDOW, m.RESUME_GRACE, m.UPSTREAM_TIMEOUT)
PY
)
set -- $vals
if [ $# -eq 3 ]; then
    [ "$1" = "$window" ]; check $? "the proxy's hold window ($1 s) is 1bit-heavy's --max-seconds default ($window s)"
    [ "$3" -eq $(( $1 + $2 )) ]; check $? "the upstream timeout ($3 s) is the window plus the resume ($1 + $2)"
    [ "$2" -ge 900 ]; check $? "the resume does not go below the 900 s the socket allowed before ($2 s)"
    [ "$3" -gt "$window" ]; check $? "the bound outlasts a whole default hold ($3 s > $window s)"
else
    check 1 "the proxy derives its bound from the hold window (HOLD_WINDOW/RESUME_GRACE/UPSTREAM_TIMEOUT: none) - it still hardcodes $(sed -n 's/.*HTTPConnection(\*UPSTREAM, timeout=\([0-9]*\)).*/\1/p' "$proxy") s, shorter than the $window s hold window"
fi

# 2. a fake Lemonade that sends nothing for a while, the way a hold does, with the proxy in front of it.
cat > "$t/fake.py" << 'PY'
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
stall_file, port_file = sys.argv[1], sys.argv[2]
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        time.sleep(float(open(stall_file).read()))   # a held turn: Lemonade sends nothing while the hold lasts
        data = json.dumps({"answered": True}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
s = ThreadingHTTPServer(("127.0.0.1", 0), H)
s.daemon_threads = True
open(port_file, "w").write(str(s.server_address[1]))
s.serve_forever()
PY
mkdir -p "$t/app"; echo '<html>chat</html>' > "$t/app/index.html"
echo 1 > "$t/stall"
python3 "$t/fake.py" "$t/stall" "$t/up.port" & srv=$!
for _ in $(seq 1 50); do [ -s "$t/up.port" ] && break; sleep 0.1; done
up=$(cat "$t/up.port")
free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'; }
start_proxy() {  # <upstream timeout s>
    pport=$(free_port)
    ONEBIT_WEBAPP_APP=$t/app ONEBIT_WEBAPP_PORT=$pport ONEBIT_WEBAPP_CONF=$t/none \
        ONEBIT_WEBAPP_UPSTREAM_PORT=$up ONEBIT_WEBAPP_UPSTREAM_TIMEOUT=$1 \
        python3 "$proxy" > "$t/proxy.log" 2>&1 &
    proxy_pid=$!
    for _ in $(seq 1 50); do curl -s -o /dev/null -m 1 "http://127.0.0.1:$pport/" && break; sleep 0.1; done
}
stop_proxy() { kill "$proxy_pid" 2> /dev/null; wait "$proxy_pid" 2> /dev/null; proxy_pid=; }
ask() {  # -> sets code, elapsed s; a POST the page would send, through the proxy
    t0=$(date +%s)
    code=$(curl -s -m 30 -o "$t/body" -w '%{http_code}' -X POST -H 'Content-Type: application/json' \
        -d '{"stream":true}' "http://127.0.0.1:$pport/api/v1/chat/completions")
    elapsed=$(( $(date +%s) - t0 ))
}

# 2a. a silent upstream inside the bound is a held turn still waiting, not an error: forwarded and answered
start_proxy 4
ask
[ "$code" = 200 ] && grep -q '"answered": true' "$t/body"
check $? "a silent (held) upstream inside the bound is forwarded: HTTP $code after ${elapsed} s, $(head -c 60 "$t/body")"
stop_proxy

# 2b. a silence PAST the bound answers 502 at the bound, not at the stall. The pre-fix proxy has no such knob, so its
#     hardcoded 900 s socket waits out the stall and answers 200 where this arm wants 502.
echo 3 > "$t/stall"
start_proxy 1
ask
[ "$code" = 502 ] && [ "$elapsed" -le 2 ]
check $? "a stall past the bound answers 502 at the bound (HTTP $code after ${elapsed} s for a 3 s stall, bound 1 s)"
stop_proxy

echo "webapp_proxy_hold_timeout_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
