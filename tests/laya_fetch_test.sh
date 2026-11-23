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
# engine#3350 part 1: `1bit-laya fetch` on a FAKE checkpoint served from file:// (no network, no 808 MB): the files
# land checked by sha256, a second fetch pulls nothing, a tampered source is refused with nothing put in place, an
# interrupted download resumes, and `status` tells complete from incomplete. Also: the shipped list pins exactly
# config/laya.json's repo and revision, and the three files laya::Scorer loads.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
laya=$here/os/v3/puppy/embed/usr/local/bin/1bit-laya
shipped=$here/os/v3/puppy/embed/etc/1bit/laya-files
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
command -v curl > /dev/null && command -v flock > /dev/null || { echo "laya_fetch_test: needs curl and flock"; exit 1; }

src=$t/src
mkdir -p "$src/typed-decisions/tokenizer"
head -c 300000 /dev/urandom > "$src/typed-decisions/model.safetensors"
printf '{"a":1}\n' > "$src/typed-decisions/rl_agent_config.json"
printf '{"tok":true}\n' > "$src/typed-decisions/tokenizer/tokenizer.json"
{
    echo "repo fake/laya"; echo "revision 0000000000000000000000000000000000000000"
    for f in typed-decisions/model.safetensors typed-decisions/rl_agent_config.json typed-decisions/tokenizer/tokenizer.json; do
        echo "$(sha256sum "$src/$f" | cut -c1-64) $(stat -c %s "$src/$f") $f"
    done
} > "$t/list"
export ONEBIT_LAYA_FILES=$t/list ONEBIT_LAYA_URL="file://$src"
d=$t/laya

sh "$laya" fetch "$d" 2> "$t/err"; check $? "fetch exits 0"
cmp -s "$src/typed-decisions/model.safetensors" "$d/typed-decisions/model.safetensors" &&
    cmp -s "$src/typed-decisions/tokenizer/tokenizer.json" "$d/typed-decisions/tokenizer/tokenizer.json"
check $? "the files land byte for byte at the paths laya::Scorer reads"
[ -z "$(find "$d" -name '*.partial')" ]; check $? "no .partial is left behind"
grep -q "3 files, 3 pulled now" "$t/err"; check $? "it says what it pulled"
sh "$laya" fetch "$d" 2> "$t/err"; grep -q "3 files, 0 pulled now" "$t/err"; check $? "a second fetch pulls nothing"
[ "$(sh "$laya" status "$d")" = "checkpoint: complete in $d" ]; check $? "status: complete"

# tampered source: refused, and the good copy already in place is not touched; a fresh dir gets nothing
cp "$src/typed-decisions/rl_agent_config.json" "$t/good"
printf '{"a":2}\n' > "$src/typed-decisions/rl_agent_config.json"
sh "$laya" fetch "$t/fresh" 2> "$t/err"; rc=$?
[ $rc = 1 ] && grep -q "does not match its pinned sha256" "$t/err" && [ ! -e "$t/fresh/typed-decisions/rl_agent_config.json" ] &&
    [ ! -e "$t/fresh/typed-decisions/rl_agent_config.json.partial" ]
check $? "a file that does not match its sha256 is refused (exit 1) and nothing is put in place"
[ "$(sh "$laya" status "$t/fresh")" != "checkpoint: complete in $t/fresh" ]; check $? "status: not complete after a refusal"
cp "$t/good" "$src/typed-decisions/rl_agent_config.json"

# resume: half a file already in .partial
d2=$t/resume; mkdir -p "$d2/typed-decisions"
head -c 100000 "$src/typed-decisions/model.safetensors" > "$d2/typed-decisions/model.safetensors.partial"
sh "$laya" fetch "$d2" 2> "$t/err"; rc=$?
[ $rc = 0 ] && cmp -s "$src/typed-decisions/model.safetensors" "$d2/typed-decisions/model.safetensors"
check $? "an interrupted download resumes from its .partial and still verifies"

# the shipped list
unset ONEBIT_LAYA_FILES ONEBIT_LAYA_URL
cr=$(sed -n 's/.*"repo": *"\([^"]*\)".*/\1/p' "$here/config/laya.json" | head -1)
cv=$(sed -n 's/.*"revision": *"\([^"]*\)".*/\1/p' "$here/config/laya.json" | head -1)
[ "$(sed -n 's/^repo //p' "$shipped")" = "$cr" ] && [ "$(sed -n 's/^revision //p' "$shipped")" = "$cv" ]
check $? "the shipped list pins config/laya.json's repo and revision ($cr@$(printf '%s' "$cv" | cut -c1-12))"
ok=0
for f in typed-decisions/model.safetensors typed-decisions/rl_agent_config.json typed-decisions/tokenizer/tokenizer.json; do
    grep -qE "^[0-9a-f]{64} [0-9]+ $f\$" "$shipped" || ok=1
done
check $ok "it lists the three files laya::Scorer loads, each with a sha256 and a size"
# a stalled connection must end the transfer, so --retry and the resume take over (`1bit route --serve` waits on fetch)
grep -q -- '--speed-limit [0-9][0-9]* --speed-time [0-9][0-9]*' "$laya" && grep -q -- '--connect-timeout [0-9]' "$laya"
check $? "the download has a connect timeout and a speed floor (a stall becomes an error, not a hang)"

# register: the key reaches Lemonade but is on no curl command line (argv is world-readable), and its header file goes
if command -v python3 > /dev/null; then
    cat > "$t/lemond.py" << 'PY'
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
log = sys.argv[1]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, code, body):
        b = json.dumps(body).encode(); self.send_response(code); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.reply(200, {"status": "ok"} if self.path.startswith("/api/v1/health") else {"data": []})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); self.rfile.read(n)
        open(log, "a").write(json.dumps({"path": self.path, "auth": self.headers.get("Authorization")}) + "\n")
        self.reply(200, {"ok": True})
s = HTTPServer(("127.0.0.1", 0), H)
open(sys.argv[2], "w").write(str(s.server_address[1]))
s.serve_forever()
PY
    python3 "$t/lemond.py" "$t/reg.log" "$t/port" & srv=$!
    for _ in $(seq 1 50); do [ -s "$t/port" ] && break; sleep 0.1; done
    printf '#!/bin/sh\necho %s\n' "'{\"model_name\": \"user.Laya-First\", \"recipe\": \"router\"}'" > "$t/1bit"; chmod +x "$t/1bit"
    real_curl=$(command -v curl); mkdir -p "$t/shim" "$t/tmp"
    printf '#!/bin/sh\nprintf "%%s\\n" "$*" >> "%s"\nexec "%s" "$@"\n' "$t/argv" "$real_curl" > "$t/shim/curl"
    chmod +x "$t/shim/curl"; : > "$t/argv"
    PATH=$t/shim:$PATH TMPDIR=$t/tmp ONEBIT_BIN=$t/1bit ONEBIT_LEMONADE_URL="http://127.0.0.1:$(cat "$t/port")" \
        LEMONADE_API_KEY=k456 ONEBIT_LAYA_WAIT=10 sh "$laya" register 2> "$t/err"; rc=$?
    kill "$srv" 2> /dev/null; wait "$srv" 2> /dev/null
    [ $rc = 0 ] && grep -q '"auth": "Bearer k456"' "$t/reg.log"; check $? "register sends the key to Lemonade ($(tail -1 "$t/err"))"
    [ -s "$t/argv" ] && ! grep -q k456 "$t/argv"; check $? "and the key is in no curl argv"
    [ -z "$(ls -A "$t/tmp")" ]; check $? "and no temp file holding it is left"
fi

sh "$laya" bogus 2> /dev/null; [ $? = 64 ]; check $? "an unknown command is a usage error"

echo "laya_fetch_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
