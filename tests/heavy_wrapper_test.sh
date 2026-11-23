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
# engine#3350: 1bit-heavy holds Lemonade's models around a command and ALWAYS releases them. Runs the wrapper against
# a fake Lemonade (python http.server on a free loopback port) that records every call and can play an old Lemonade
# (404) or a busy one (409): the hold comes before the command, the release after it on success, failure and Ctrl-C,
# the command's exit status is the wrapper's, --require refuses without the endpoint, and the key is sent only when set.
# No root, no device, no real Lemonade.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
heavy=$here/os/v3/puppy/embed/usr/local/bin/1bit-heavy
t=$(mktemp -d) || exit 1
bad=0
srv=
cleanup() { [ -n "$srv" ] && kill "$srv" 2> /dev/null; rm -rf "${t:?}"; }
trap cleanup EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
command -v python3 > /dev/null && command -v curl > /dev/null || { echo "heavy_wrapper_test: needs python3 and curl"; exit 1; }

cat > "$t/fake.py" << 'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
log, mode_file = sys.argv[1], sys.argv[2]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        mode = open(mode_file).read().strip()
        with open(log, "a") as f:
            f.write(json.dumps({"path": self.path, "body": body, "auth": self.headers.get("Authorization")}) + "\n")
        if mode == "old":
            code, out = 404, {"error": "not found"}
        elif mode == "busy" and self.path.endswith("/hold"):
            code, out = 409, {"error": "a hold is already active (hold-7: quantize)"}
        elif self.path.endswith("/hold"):
            code, out = 200, {"hold_id": "hold-1", "reason": body.get("reason"), "max_seconds": body.get("max_seconds"),
                              "unloaded": ["Qwen3-0.6B-1bit"]}
        else:
            code, out = 200, {"hold_id": body.get("hold_id"), "reloaded": ["Qwen3-0.6B-1bit"], "failed": [], "expired": False}
        data = json.dumps(out).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
s = ThreadingHTTPServer(("127.0.0.1", 0), H)
open(log + ".port", "w").write(str(s.server_address[1]))
s.serve_forever()
PY
echo ok > "$t/mode"
python3 "$t/fake.py" "$t/calls" "$t/mode" & srv=$!
for _ in $(seq 1 50); do [ -s "$t/calls.port" ] && break; sleep 0.1; done
export ONEBIT_LEMONADE_URL="http://127.0.0.1:$(cat "$t/calls.port")" ONEBIT_LEMOND_CONF=$t/none
unset LEMONADE_API_KEY
calls() { sed -n 's/.*"path": "\([^"]*\)".*/\1/p' "$t/calls" | tr '\n' ' '; }

# 1. success: hold, then the command (which sees the hold taken), then release; the command's status passes through
: > "$t/calls"
sh "$heavy" --reason "quantize a model" --max-seconds 120 -- sh -c "grep -c /api/v1/hold '$t/calls' > '$t/seen'; exit 0" 2> "$t/err"
check $? "exit 0 when the command succeeds"
[ "$(calls)" = "/api/v1/hold /api/v1/hold/release " ]; check $? "hold before the command, release after it ($(calls))"
[ "$(cat "$t/seen")" = 1 ]; check $? "the hold is already taken when the command starts"
grep -q '"reason": "quantize a model", "max_seconds": 120' "$t/calls"; check $? "--reason and --max-seconds are sent"
grep -q '"hold_id": "hold-1"' "$t/calls"; check $? "the release names the hold that was taken"
! grep -q '"auth": "Bearer' "$t/calls"; check $? "no Authorization header on a blank-key appliance"
grep -q "released: Qwen3-0.6B-1bit" "$t/err"; check $? "it says which models came back"

# 2. failure: the command's own status, and the release still happens
: > "$t/calls"
sh "$heavy" -- sh -c 'exit 7' 2> /dev/null; rc=$?
[ $rc = 7 ]; check $? "the command's exit status is the wrapper's (7, got $rc)"
[ "$(calls)" = "/api/v1/hold /api/v1/hold/release " ]; check $? "a failing command is still followed by the release"

# 3. Ctrl-C: the release still happens
: > "$t/calls"
sh "$heavy" -- sh -c 'kill -INT $PPID; sleep 1' 2> /dev/null
[ "$(calls)" = "/api/v1/hold /api/v1/hold/release " ]; check $? "an interrupted wrapper still releases ($(calls))"

# 4. the key is sent when set
: > "$t/calls"
LEMONADE_API_KEY=k123 sh "$heavy" -- true 2> /dev/null
[ "$(grep -c '"auth": "Bearer k123"' "$t/calls")" = 2 ]; check $? "LEMONADE_API_KEY is sent on the hold and the release"
# ... but never on curl's command line (argv is world-readable in ps), and no file holding it is left behind
real_curl=$(command -v curl); mkdir -p "$t/shim" "$t/tmp"
printf '#!/bin/sh\nprintf "%%s\\n" "$*" >> "%s"\nexec "%s" "$@"\n' "$t/argv" "$real_curl" > "$t/shim/curl"; chmod +x "$t/shim/curl"
: > "$t/calls"; : > "$t/argv"
PATH=$t/shim:$PATH TMPDIR=$t/tmp LEMONADE_API_KEY=k123 sh "$heavy" -- true 2> /dev/null
[ -s "$t/argv" ] && ! grep -q k123 "$t/argv" && [ "$(grep -c '"auth": "Bearer k123"' "$t/calls")" = 2 ]
check $? "the key reaches Lemonade but is in no curl argv"
[ -z "$(ls -A "$t/tmp")" ]; check $? "the wrapper leaves no temp file (the header file holds the key)"

# 5. an old Lemonade without the endpoint: the command runs with a warning; --require refuses and does not run it
echo old > "$t/mode"; : > "$t/calls"
sh "$heavy" -- sh -c "echo ran > '$t/ran'" 2> "$t/err"; rc=$?
[ $rc = 0 ] && [ -f "$t/ran" ] && grep -q "WARNING: running without a hold - this Lemonade has no hold endpoint" "$t/err"
check $? "without the endpoint the command still runs, with a warning that names why"
[ "$(calls)" = "/api/v1/hold " ]; check $? "and nothing is released that was never held"
rm -f "$t/ran"
sh "$heavy" --require -- sh -c "echo ran > '$t/ran'" 2> "$t/err"; rc=$?
[ $rc = 2 ] && [ ! -f "$t/ran" ] && grep -q REFUSED "$t/err"; check $? "--require refuses (exit 2) and does not run the command"

# 6. another hold active: named in the warning
echo busy > "$t/mode"
sh "$heavy" -- true 2> "$t/err"
grep -q "another hold is active: a hold is already active (hold-7: quantize)" "$t/err"; check $? "a busy Lemonade's 409 is explained"

# 7. usage
sh "$heavy" 2> /dev/null; [ $? = 64 ]; check $? "no command is a usage error (64)"
sh "$heavy" --max-seconds soon -- true 2> /dev/null; [ $? = 64 ]; check $? "a non-numeric --max-seconds is a usage error"

echo "heavy_wrapper_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
