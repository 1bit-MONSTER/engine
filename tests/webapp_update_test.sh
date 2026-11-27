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
# OS updater U5 (local): the update page's API on 1bit-webapp-proxy. status/progress are open on loopback; install,
# rollback and reboot are served only to the administrator's socket (uid 0 on the box; here the TEST uid, which the
# proxy honours only because it does not run as root), with X-1bit-Update: 1, the Origin rule and the typed
# confirmation, one at a time. A fake 1bit-update and a fake reboot: no root, no network, no real update.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
proxy=$here/os/v3/puppy/embed/usr/local/bin/1bit-webapp-proxy
t=$(mktemp -d) || exit 1
bad=0
proxy_pid=
cleanup() { [ -n "$proxy_pid" ] && kill "$proxy_pid" 2> /dev/null; rm -rf "${t:?}"; }
trap cleanup EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
command -v python3 > /dev/null && command -v curl > /dev/null || { echo "webapp_update_test: needs python3 and curl"; exit 1; }
# As root (CI's container) the proxy's test hooks are inert by design, so the test re-runs itself as nobody - the
# proxy AND the clients must be non-root for "a client that is not root is refused" to mean anything
if [ "$(id -u)" = 0 ]; then
    command -v setpriv > /dev/null || { echo "SKIP webapp_update_test: root, and no setpriv to drop to nobody"; exit 77; }
    exec setpriv --reuid=65534 --regid=65534 --clear-groups env HOME=/tmp PATH="$PATH" sh "$here/tests/webapp_update_test.sh"
fi

cat > "$t/1bit-update" <<'SH'
#!/bin/sh
case "$1" in
    --status) echo "1bit-update: active slot = 1bit (data partition on nvme0n1, ESP /dev/nvme0n1p1)" ;;
    --check)  echo x >> "$(dirname "$0")/check-calls"; printf 'running:  v2026.49-rc1\ntest: v2026.49-rc9 (2026-11-27, pre-release)\nan update is available: 1bit-update --latest installs it into the other slot\n' ;;
    --latest) echo "downloading v2026.49-rc9"; echo "env-pub=[${ONEBIT_UPDATE_PUB:-}] env-any=[${ONEBIT_UPDATE_ANY:-}]"; sleep 2; echo "UPDATED: /1bit-alt"; exit 0 ;;
    --rollback) echo "rolled back: next boot is /1bit-alt" ;;
esac
SH
chmod 755 "$t/1bit-update"
free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'; }
start() {   # [admin uid]
    port=$(free_port); uport=$(free_port)
    ONEBIT_WEBAPP_PORT=$port ONEBIT_WEBAPP_UPDATE_PORT=$uport ONEBIT_WEBAPP_CONF=$t/none ONEBIT_WEBAPP_APP=$here/os/v3/puppy/embed/opt/1bit/webapp \
    ONEBIT_UPDATE_BIN=$t/1bit-update ONEBIT_WEBAPP_UPDATE_LOG=$t/update.log ONEBIT_WEBAPP_TEST_ADMIN_UID=${1:-} \
    ONEBIT_WEBAPP_TEST_REBOOT="touch $t/rebooted" ONEBIT_UPDATE_PUB=$t/attacker.pub ONEBIT_UPDATE_ANY=1 \
    PYTHONDONTWRITEBYTECODE=1 python3 "$proxy" > "$t/proxy.log" 2>&1 &
    proxy_pid=$!
    for _ in $(seq 50); do curl -s -o /dev/null "http://127.0.0.1:$uport/" && return 0; sleep 0.1; done
    return 1
}
stop() { kill "$proxy_pid" 2> /dev/null; wait "$proxy_pid" 2> /dev/null; proxy_pid=; }
U() { curl -s -o "$t/body" -w '%{http_code}' "$@"; }
P() { U -X POST -H 'Content-Type: application/json' "$@"; }
j() { python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); [v := v[k] for k in sys.argv[2:]]; print(v)' "$t/body" "$@"; }

# 1. not the administrator (the default: uid 0, and this test is not root)
start || { echo "FAIL the proxy did not start: $(tail -3 "$t/proxy.log")"; exit 1; }
[ "$(U "http://127.0.0.1:$uport/1bit/update/status")" = 200 ] && [ "$(j offered)" = v2026.49-rc9 ] && [ "$(j rollback_to)" = 1bit-alt ] && [ "$(j admin)" = False ]
check $? "status: open on loopback, names the offered release and the rollback slot, says this client is not admin"
[ "$(U "http://127.0.0.1:$uport/")" = 200 ] && grep -q 'src="/update.js"' "$t/body" && [ "$(U "http://127.0.0.1:$uport/update.js")" = 200 ] && grep -q "X-1bit-Update" "$t/body"
check $? "the update page (and its script, which sends X-1bit-Update) is served on its own origin"
curl -s -D "$t/hdr" -o /dev/null "http://127.0.0.1:$uport/"
grep -qi "^Content-Security-Policy: default-src 'self'; frame-ancestors 'none'" "$t/hdr" && grep -qi '^X-Frame-Options: DENY' "$t/hdr" && grep -qi '^X-Content-Type-Options: nosniff' "$t/hdr"
check $? "the update origin sends CSP default-src 'self', no framing, nosniff"
[ "$(U "http://127.0.0.1:$port/1bit/update/status")" = 404 ] && [ "$(U "http://127.0.0.1:$port/update.html")" = 404 ] && [ "$(U "http://127.0.0.1:$port/update.js")" = 404 ]
check $? "the chat origin (:8080, which also serves Lemonade's /api/) serves neither the update page nor its API"
[ "$(U "http://127.0.0.1:$uport/index.html")" = 404 ] && [ "$(U "http://127.0.0.1:$uport/api/v1/models")" = 404 ]; check $? "the update origin serves nothing else"
: > "$t/check-calls"; U "http://127.0.0.1:$uport/1bit/update/status" > /dev/null; U "http://127.0.0.1:$uport/1bit/update/status" > /dev/null
[ "$(wc -l < "$t/check-calls")" = 0 ]; check $? "status reuses a cached --check (0 new calls for 2 requests) (GitHub's 60 calls an hour are not a local loop's to spend)"
[ "$(P "http://127.0.0.1:$uport/1bit/update/latest" -d '{"confirm":"v2026.49-rc9"}')" = 403 ] && grep -q "X-1bit-Update" "$t/body"
check $? "a POST without X-1bit-Update is refused (403)"
[ "$(P -H 'X-1bit-Update: 1' "http://127.0.0.1:$uport/1bit/update/latest" -d '{"confirm":"v2026.49-rc9"}')" = 403 ] && grep -q "not root" "$t/body"
check $? "a POST from a client that is not root is refused (403), even with the header and the right confirmation"
[ ! -s "$t/update.log" ]; check $? "  ... and nothing ran"
[ "$(U -H 'Origin: http://evil.example' "http://127.0.0.1:$uport/1bit/update/status")" = 403 ] && [ "$(U -H "Origin: http://127.0.0.1:$port" "http://127.0.0.1:$uport/1bit/update/status")" = 403 ]
check $? "a cross-origin request is refused (403) - the chat origin's included"
stop

# 2. the administrator (the TEST uid = this user)
start "$(id -u)" || { echo "FAIL the proxy did not start"; exit 1; }
[ "$(U "http://127.0.0.1:$uport/1bit/update/status")" = 200 ] && [ "$(j admin)" = True ]; check $? "status says the admin client is admin"
H='X-1bit-Update: 1'
[ "$(P "http://127.0.0.1:$uport/1bit/update/latest" -d '{"confirm":"v2026.49-rc9"}')" = 403 ] && grep -q "X-1bit-Update" "$t/body" && [ ! -s "$t/update.log" ]
check $? "even the ADMIN client is refused without X-1bit-Update (the boundary against a root daemon relaying a request)"
[ "$(P -H "$H" "http://127.0.0.1:$uport/1bit/update/latest" -d '{"confirm":"v2026.50"}')" = 400 ] && grep -q "v2026.49-rc9" "$t/body"
check $? "install with the wrong confirmation is refused (400) and names the offered release"
[ "$(P -H "$H" "http://127.0.0.1:$uport/1bit/update/latest" -d 'not json')" = 400 ]; check $? "a body that is not JSON is refused (400)"
c0=$(wc -l < "$t/check-calls" 2> /dev/null || echo 0)
[ "$(P -H "$H" "http://127.0.0.1:$uport/1bit/update/latest" -d '{"confirm":"v2026.49-rc9"}')" = 202 ]; check $? "install with the offered release starts (202)"
[ "$(wc -l < "$t/check-calls")" -gt "$c0" ]; check $? "  ... after a FRESH --check (never the cached one)"
[ "$(P -H "$H" "http://127.0.0.1:$uport/1bit/update/rollback" -d '{"confirm":"1bit-alt"}')" = 409 ]; check $? "a second action while one runs is refused (409)"
for _ in $(seq 60); do U "http://127.0.0.1:$uport/1bit/update/progress" > /dev/null; [ "$(j job running)" = False ] && break; sleep 0.1; done
[ "$(j job rc)" = 0 ] && [ "$(j job action)" = "install v2026.49-rc9" ] && j log | grep -q "UPDATED: /1bit-alt"
check $? "progress shows the finished install, its exit 0 and its output"
j log | grep -qF 'env-pub=[] env-any=[]'; check $? "the proxy's ONEBIT_UPDATE_PUB / ONEBIT_UPDATE_ANY never reach 1bit-update (fixed child environment)"
[ "$(stat -c %a "$t/update.log")" = 600 ]; check $? "the action log is root-only (0600)"
[ "$(P -H "$H" "http://127.0.0.1:$uport/1bit/update/rollback" -d '{"confirm":"1bit"}')" = 400 ]; check $? "rollback to the wrong slot is refused (400)"
[ "$(P -H "$H" "http://127.0.0.1:$uport/1bit/update/rollback" -d '{"confirm":"1bit-alt"}')" = 202 ]; check $? "rollback to the other slot starts (202)"
for _ in $(seq 60); do U "http://127.0.0.1:$uport/1bit/update/progress" > /dev/null; [ "$(j job running)" = False ] && break; sleep 0.1; done
[ "$(P -H "$H" "http://127.0.0.1:$uport/1bit/update/reboot" -d '{"confirm":"yes"}')" = 400 ] && [ ! -e "$t/rebooted" ]; check $? "reboot without the word is refused"
[ "$(P -H "$H" "http://127.0.0.1:$uport/1bit/update/reboot" -d '{"confirm":"reboot"}')" = 202 ]
for _ in $(seq 50); do [ -e "$t/rebooted" ] && break; sleep 0.1; done
[ -e "$t/rebooted" ]; check $? "reboot with the word runs the reboot command (the test's stand-in)"
[ "$(P -H "$H" "http://127.0.0.1:$uport/1bit/update/nothing" -d '{}')" = 404 ]; check $? "an unknown action is 404"
stop

# 3. peer_uid reads the CLIENT end of the connection from the kernel's table (a fake table here)
out=$(PYTHONDONTWRITEBYTECODE=1 python3 - "$proxy" "$t" <<'PY'
import importlib.util, sys
from importlib.machinery import SourceFileLoader
loader = SourceFileLoader("webapp_proxy", sys.argv[1]); spec = importlib.util.spec_from_loader("webapp_proxy", loader)
m = importlib.util.module_from_spec(spec); loader.exec_module(m)
tbl = sys.argv[2] + "/tcp"
open(tbl, "w").write(
 "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
 "   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 1\n"     # the listener
 "   1: 0100007F:1F90 0100007F:D431 01 00000000:00000000 00:00000000 00000000     0        0 2\n"     # server end, root
 "   2: 0100007F:D431 0100007F:1F90 01 00000000:00000000 00:00000000 00000000   502        0 3\n")   # client end, spot
print(m.peer_uid(("127.0.0.1", 0xD431), 8080, tbl), m.peer_uid(("127.0.0.1", 1234), 8080, tbl), m.ADMIN_UID)
PY
)
[ "$out" = "502 None 0" ]; check $? "peer_uid takes the client end's uid (502, not the server end's 0); unknown -> None; admin uid 0 by default (got: $out)"

echo "webapp_update_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
