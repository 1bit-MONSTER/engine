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
# 1bit-loom-companions: a Loom card's companion weights (a second Hugging Face repository its lock pins - the native
# Inkling card's MXFP8 experts) fetched into the cache where 1bit-loom-cards links them: only once the card's model is
# pulled, pinned, resumable, and only with the lock's size + sha256. A fake card, a fake Hugging Face and a temp cache.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
tool=$here/os/v3/puppy/embed/usr/local/bin/1bit-loom-companions
cards=$here/os/v3/puppy/embed/usr/local/bin/1bit-loom-cards
t=$(mktemp -d) || exit 1
bad=0
srv=
cleanup() { [ -n "$srv" ] && kill "$srv" 2> /dev/null; rm -rf "${t:?}"; }
trap cleanup EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
command -v python3 > /dev/null || { echo "loom_companions_test: needs python3"; exit 1; }

REV=a9955dddd10125865cbd76a720d9530c6be89ab4 CREV=8078d149286eec7193795413b9e0643da6d11fa1
mkdir -p "$t/www/org/experts/resolve/$REV" "$t/payload/Card-loom" "$t/hf"
head -c 3000000 /dev/urandom > "$t/www/org/experts/resolve/$REV/E.safetensors"
SHA=$(sha256sum "$t/www/org/experts/resolve/$REV/E.safetensors" | cut -c1-64); SIZE=$(stat -c %s "$t/www/org/experts/resolve/$REV/E.safetensors")
cat > "$t/payload/Card-loom/model.json" <<JSON
{"variants": {"gguf": {"services": {"chat": {"parameters": {"model": {"files": [
  {"path": "parameter/M.gguf", "origin": "@checkpoint/M.gguf"},
  {"path": "parameter/E.safetensors", "format": "safetensors", "origin": "@mxfp8/E.safetensors"}]}}}}}}}
JSON
cat > "$t/payload/Card-loom/model.json.lock" <<JSON
{"origins": {"checkpoint": {"requested": {"kind": "hf", "repository": "unsloth/M-GGUF", "revision": "$CREV"}, "resolved": {"revision": "$CREV"}},
             "mxfp8": {"requested": {"kind": "hf", "repository": "org/experts", "revision": "$REV"}, "resolved": {"revision": "$REV"}}},
 "files": [{"origin": "@checkpoint/M.gguf", "size": 10, "sha256": "$(printf %064d 0)"},
           {"origin": "@mxfp8/E.safetensors", "size": $SIZE, "sha256": "$SHA"}]}
JSON
# a fake Hugging Face that serves /<org>/<repo>/resolve/<rev>/<file> with Range, and logs each request
cat > "$t/hf.py" <<'PY'
import os, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
root, portfile, log = sys.argv[1], sys.argv[2], sys.argv[3]
class H(SimpleHTTPRequestHandler):
    def __init__(s, *a, **k): super().__init__(*a, directory=root, **k)
    def log_message(s, *a): pass
    def do_GET(s):
        p = s.translate_path(s.path)
        open(log, "a").write("%s %s\n" % (s.path, s.headers.get("Range", "-")))
        if not os.path.isfile(p): return s.send_error(404)
        data = open(p, "rb").read(); start = 0
        r = s.headers.get("Range")
        if r and r.startswith("bytes="): start = int(r[6:].split("-")[0])
        s.send_response(206 if start else 200)
        s.send_header("Content-Length", str(len(data) - start)); s.end_headers(); s.wfile.write(data[start:])
srv = ThreadingHTTPServer(("127.0.0.1", 0), H); open(portfile, "w").write(str(srv.server_address[1])); srv.serve_forever()
PY
python3 "$t/hf.py" "$t/www" "$t/port" "$t/req.log" & srv=$!
for _ in $(seq 50); do [ -s "$t/port" ] && break; sleep 0.1; done
export ONEBIT_LOOM_HF_BASE="http://127.0.0.1:$(cat "$t/port")"
C() { python3 "$tool" "$@" "$t/payload" "$t/hf"; }
dest=$t/hf/hub/models--org--experts/snapshots/$REV/E.safetensors

out=$(C --status); rc=$?
[ $rc = 0 ] && printf '%s' "$out" | grep -q "not needed yet"; check $? "status: the model is not pulled -> the companion is not needed yet (exit 0)"
C --once > /dev/null; [ ! -e "$dest" ] && [ ! -s "$t/req.log" ]; check $? "--once with the model not pulled downloads nothing"

mkdir -p "$t/hf/hub/models--unsloth--M-GGUF/snapshots/$CREV"          # the person pulled the model
out=$(C --status); [ $? = 1 ] && printf '%s' "$out" | grep -q "MISSING"; check $? "status: the model is pulled, the companion MISSING (exit 1)"
out=$(C --once); rc=$?
[ $rc = 0 ] && [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -c1-64)" = "$SHA" ] && grep -q "in place" <<EOF2
$out
EOF2
check $? "--once fetches it at the PINNED revision into the snapshot, the lock's sha256"
grep -q "^/org/experts/resolve/$REV/E.safetensors " "$t/req.log"; check $? "  ... from <repo>/resolve/<lock revision>/<file>"
[ "$(stat -c %a "$dest")" = 640 ] && [ "$(stat -c %a "$(dirname "$dest")")" = 750 ] && [ "$(stat -c %a "$t/hf/.1bit-loom-companions.lock")" = 600 ] && [ ! -e "$dest.part" ]
check $? "  ... private modes (file 0640, its new directory 0750, the lock 0600; nothing for others), no .part left"
C --status > /dev/null; check $? "status: present (exit 0)"
: > "$t/req.log"; C --once > /dev/null; [ ! -s "$t/req.log" ]; check $? "a second run downloads nothing"

# 1bit-loom-cards links the card's parameter exactly where the fetcher put it
sh "$cards" "$t/payload" "$t/cards" "$t/hf" 2> /dev/null
[ "$(readlink "$t/cards/Card-loom/parameter/E.safetensors")" = "$dest" ] && [ -f "$t/cards/Card-loom/parameter/E.safetensors" ]
check $? "1bit-loom-cards' parameter/E.safetensors link resolves to the fetched file"

# resume: half a .part is completed with a Range request
rm -f "$dest"; head -c 1000000 "$t/www/org/experts/resolve/$REV/E.safetensors" > "$dest.part"; : > "$t/req.log"
C --once > /dev/null; [ "$(sha256sum "$dest" | cut -c1-64)" = "$SHA" ] && grep -q "bytes=1000000-" "$t/req.log"
check $? "a .part is resumed with Range: bytes=<have>- and completed"

# a server that serves other bytes cannot substitute the companion
rm -f "$dest"; cp "$t/www/org/experts/resolve/$REV/E.safetensors" "$t/good"; head -c "$SIZE" /dev/urandom > "$t/www/org/experts/resolve/$REV/E.safetensors"
out=$(C --once); rc=$?
[ $rc = 1 ] && [ ! -e "$dest" ] && [ ! -e "$dest.part" ] && printf '%s' "$out" | grep -q "REFUSED"
check $? "other bytes of the right size are REFUSED (sha256), nothing is left in the cache"
C --status > /dev/null; [ $? = 1 ]; check $? "  ... and status says MISSING again"

# a lock that names a path-like repository or revision is skipped, never followed
sed -i "s|\"org/experts\"|\"org/../../etc\"|" "$t/payload/Card-loom/model.json.lock"
out=$(C --status); [ -z "$out" ]; check $? "a repository with .. in the lock is not a companion (skipped)"

echo "loom_companions_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
