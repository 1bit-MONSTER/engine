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
# OS updater U5b: the LAN update page on 1bit-webapp-proxy. It exists only while an update credential is set; it is
# HTTPS only, serves update.html and /1bit/update/* and nothing else, needs Basic auth (root + the update password)
# on every request, locks an address after 5 failed sign-ins, and keeps the header / origin / typed-confirmation
# rules. A throwaway credential, certificate, fake 1bit-update and reboot stand-in in a temp dir; no root.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
proxy=$here/os/v3/puppy/embed/usr/local/bin/1bit-webapp-proxy
t=$(mktemp -d) || exit 1
bad=0
proxy_pid=
cleanup() { [ -n "$proxy_pid" ] && kill "$proxy_pid" 2> /dev/null; rm -rf "${t:?}"; }
trap cleanup EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
for c in python3 curl openssl; do command -v $c > /dev/null || { echo "webapp_update_lan_test: needs $c"; exit 1; }; done
# As root (CI's container) the proxy's test hooks are inert by design: re-run the whole test as nobody
if [ "$(id -u)" = 0 ]; then
    command -v setpriv > /dev/null || { echo "SKIP webapp_update_lan_test: root, and no setpriv to drop to nobody"; exit 77; }
    exec setpriv --reuid=65534 --regid=65534 --clear-groups env HOME=/tmp PATH="$PATH" sh "$here/tests/webapp_update_lan_test.sh"
fi
cat > "$t/1bit-update" <<'SH'
#!/bin/sh
case "$1" in
    --status) echo "1bit-update: active slot = 1bit (data partition on nvme0n1, ESP /dev/nvme0n1p1)" ;;
    --check)  printf 'running:  v2026.49-rc1\ntest: v2026.49-rc9 (2026-11-27, pre-release)\nan update is available: x\n' ;;
    --latest) echo "UPDATED: /1bit-alt" ;;
esac
SH
chmod 755 "$t/1bit-update"
PW='correct horse battery staple'
export ONEBIT_WEBAPP_CRED=$t/cred ONEBIT_WEBAPP_CERT=$t/crt ONEBIT_WEBAPP_KEY=$t/key PYTHONDONTWRITEBYTECODE=1
free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'; }

# 1. setting the password
printf 'short\nshort\n' | python3 "$proxy" --set-update-password > /dev/null 2>&1; [ $? != 0 ] && [ ! -e "$t/cred" ]; check $? "a password under 12 characters is refused"
printf '%s\n%s-x\n' "$PW" "$PW" | python3 "$proxy" --set-update-password > /dev/null 2>&1; [ $? != 0 ] && [ ! -e "$t/cred" ]; check $? "two different passwords are refused"
printf '%s\n%s\n' "$PW" "$PW" | python3 "$proxy" --set-update-password > /dev/null 2>&1; check $? "a good password is set"
[ "$(stat -c %a "$t/cred")" = 600 ] && grep -q '^pbkdf2_sha256\$600000\$' "$t/cred" && ! grep -qF "$PW" "$t/cred"
check $? "the credential is a 0600 PBKDF2-SHA256 hash (600k rounds), not the password"

start() {
    port=$(free_port); lport=$(free_port)
    ONEBIT_WEBAPP_PORT=$port ONEBIT_WEBAPP_LAN_PORT=$lport ONEBIT_WEBAPP_CONF=$t/none \
    ONEBIT_WEBAPP_APP=$here/os/v3/puppy/embed/opt/1bit/webapp ONEBIT_UPDATE_BIN=$t/1bit-update \
    ONEBIT_WEBAPP_UPDATE_LOG=$t/update.log ONEBIT_WEBAPP_TEST_REBOOT="touch $t/rebooted" ONEBIT_WEBAPP_TEST_CHECK_LOG=$t/checks \
    python3 "$proxy" > "$t/proxy.log" 2>&1 &
    proxy_pid=$!
    for _ in $(seq 50); do curl -s -o /dev/null "http://127.0.0.1:$port/" && return 0; sleep 0.1; done
    return 1
}
stop() { kill "$proxy_pid" 2> /dev/null; wait "$proxy_pid" 2> /dev/null; proxy_pid=; }
L() { curl -sk -o "$t/body" -w '%{http_code}' "$@"; }
j() { python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); [v := v[k] for k in sys.argv[2:]]; print(v)' "$t/body" "$@"; }

# 2. a credential but no certificate: no LAN page
start || { echo "FAIL the proxy did not start: $(tail -3 "$t/proxy.log")"; exit 1; }
[ "$(L "https://127.0.0.1:$lport/")" = 000 ] && grep -q "missing: no LAN update page" "$t/proxy.log"; check $? "no certificate: no LAN listener, and the log says why"
stop
( umask 077; openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 1 -subj /CN=test -keyout "$t/key" -out "$t/crt" > /dev/null 2>&1 )

# 3. the LAN listener
start || { echo "FAIL the proxy did not start"; exit 1; }
for _ in $(seq 30); do [ "$(L "https://127.0.0.1:$lport/")" != 000 ] && break; sleep 0.1; done
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$lport/")" != 200 ]; check $? "plain HTTP is not served on the LAN port"
[ "$(L "https://127.0.0.1:$lport/1bit/update/status")" = 401 ] && curl -skI "https://127.0.0.1:$lport/" 2>/dev/null | grep -qi "^HTTP" ; check $? "no credentials: 401"
curl -sk -D "$t/hdr" -o /dev/null "https://127.0.0.1:$lport/1bit/update/status"; grep -qi '^WWW-Authenticate: Basic' "$t/hdr"; check $? "  ... with a Basic challenge"
[ "$(L -u "root:$PW" "https://127.0.0.1:$lport/1bit/update/status")" = 200 ] && [ "$(j admin)" = True ] && [ "$(j offered)" = v2026.49-rc9 ]
check $? "root + the password: status 200, admin"
[ "$(L -u "spot:$PW" "https://127.0.0.1:$lport/1bit/update/status")" = 401 ]; check $? "another user name with the password: 401"
[ "$(L -u "root:$PW" "https://127.0.0.1:$lport/")" = 200 ] && grep -q "1bit OS - update" "$t/body"; check $? "/ is the update page"
[ "$(L -u "root:$PW" "https://127.0.0.1:$lport/update.js")" = 200 ] && grep -q "X-1bit-Update" "$t/body"; check $? "the page's script is served (the CSP forbids inline)"
curl -sk -u "root:$PW" -D "$t/hdr" -o /dev/null "https://127.0.0.1:$lport/"
grep -qi "^Content-Security-Policy: default-src 'self'" "$t/hdr" && grep -qi '^X-Frame-Options: DENY' "$t/hdr"; check $? "the LAN page sends the same CSP and no-framing headers"
[ "$(L -u "root:$PW" "https://127.0.0.1:$lport/index.html")" = 404 ] && [ "$(L -u "root:$PW" "https://127.0.0.1:$lport/api/v1/models")" = 404 ]
check $? "the chat page and /api/ are NOT served on the LAN port (the key never leaves loopback)"
H='X-1bit-Update: 1'
[ "$(L -u "root:$PW" -X POST -H 'Content-Type: application/json' "https://127.0.0.1:$lport/1bit/update/latest" -d '{"confirm":"v2026.49-rc9"}')" = 403 ]
check $? "a signed-in POST without X-1bit-Update is refused"
[ "$(L -u "root:$PW" -X POST -H "$H" -H 'Origin: https://evil.example' -H 'Content-Type: application/json' "https://127.0.0.1:$lport/1bit/update/latest" -d '{"confirm":"v2026.49-rc9"}')" = 403 ]
check $? "a signed-in POST from another origin is refused"
[ "$(L -u "root:$PW" -X POST -H "$H" -H "Origin: https://127.0.0.1:$lport" -H 'Content-Type: application/json' "https://127.0.0.1:$lport/1bit/update/latest" -d '{"confirm":"v2026.49-rc9"}')" = 202 ]
check $? "a signed-in POST from the page's own origin, with the header and the confirmation, starts the install"
for _ in $(seq 50); do L -u "root:$PW" "https://127.0.0.1:$lport/1bit/update/progress" > /dev/null; [ "$(j job running)" = False ] && break; sleep 0.1; done
[ "$(j job rc)" = 0 ]; check $? "  ... and it finishes (exit 0)"
! grep -qF "$PW" "$t/proxy.log"; check $? "the password never appears in the proxy's log"
[ "$(L -u "root:$PW" -X POST -H "$H" -H "Origin: https://127.0.0.1:$lport" -H 'Content-Type: application/json' "https://127.0.0.1:$lport/1bit/update/rollback" -d '{"confirm":"1bit-alt"}')" = 403 ] && grep -q "console" "$t/body"
check $? "rollback is refused on the network (403: the console's or the box's own page) - a captured password buys no downgrade"
L -u "root:$PW" "https://127.0.0.1:$lport/1bit/update/status" > /dev/null; [ "$(j rollback_to)" = "" ]; check $? "  ... and the LAN status offers no rollback slot"
curl -sk -u "root:$PW" -D "$t/hdr" -o /dev/null "https://127.0.0.1:$lport/1bit/update/status"; grep -qi '^Cache-Control: no-store' "$t/hdr"
check $? "the status JSON is no-store (nothing left in a browser cache after the password is turned off)"
stop

# 3b. per-address backoff that holds under concurrency, and an address the owner signed in from is never locked out
# by others' failures (127.0.0.x are all loopback: each is its own client address here; the 5-min sign-in cache is
# off in this part, so the known-address rule is tested on its own)
export ONEBIT_WEBAPP_KNOWN=$t/known ONEBIT_WEBAPP_TEST_OK_TTL=0
start || { echo "FAIL the proxy did not start"; exit 1; }
for _ in $(seq 30); do [ "$(L "https://127.0.0.1:$lport/")" != 000 ] && break; sleep 0.1; done
: > "$t/checks"; : > "$t/par"
pids=
for i in $(seq 1 16); do curl -sk -o /dev/null -w '%{http_code}\n' --interface 127.0.0.9 -u "root:wrong-$i-password" "https://127.0.0.1:$lport/1bit/update/status" >> "$t/par" & pids="$pids $!"; done
for p in $pids; do wait "$p"; done                  # the curls only: a bare wait would wait for the proxy too
[ "$(wc -l < "$t/checks")" = 4 ] && [ "$(grep -c '^429$' "$t/par")" = 12 ]
check $? "16 concurrent wrong sign-ins from one address make exactly 4 password checks (3 free, then the backoff): $(wc -l < "$t/checks") checks, $(grep -c '^429$' "$t/par") x 429"
curl -sk -u "root:$PW" --interface 127.0.0.9 -D "$t/hdr" -o /dev/null "https://127.0.0.1:$lport/1bit/update/status"
grep -q '^HTTP/1.1 429' "$t/hdr" && grep -qi '^Retry-After: [12]' "$t/hdr"; check $? "inside its backoff the address gets 429 + Retry-After, even with the right password"
sleep 2.2
[ "$(L --interface 127.0.0.9 -u "root:$PW" "https://127.0.0.1:$lport/1bit/update/status")" = 200 ]; check $? "after the wait the right password signs in (a success clears the backoff)"
grep -q "sign-in failed from 127.0.0.9" "$t/proxy.log"; check $? "failed sign-ins are logged by address"
[ "$(L --interface 127.0.0.1 -u "root:$PW" "https://127.0.0.1:$lport/1bit/update/status")" = 200 ]; check $? "the owner signs in from 127.0.0.1"
[ "$(stat -c %a "$t/known")" = 600 ] && grep -qx 127.0.0.1 "$t/known"; check $? "  ... which is now a KNOWN address (a root-only file)"
for a in 2 3 4 5 6; do for i in 1 2 3 4; do L --interface 127.0.0.$a -u "root:x-$a-$i-password" "https://127.0.0.1:$lport/1bit/update/status" > /dev/null; done; done
[ "$(L --interface 127.0.0.7 -u "root:$PW" "https://127.0.0.1:$lport/1bit/update/status")" = 429 ]; check $? "20 failures from other addresses lock sign-ins from addresses never seen before (distributed guessing)"
[ "$(L --interface 127.0.0.1 -u "root:$PW" "https://127.0.0.1:$lport/1bit/update/status")" = 200 ]; check $? "  ... but NOT the owner's known address (no cache here: the known-address rule)"
[ "$(L --interface 127.0.0.9 -u "root:$PW" "https://127.0.0.1:$lport/1bit/update/status")" = 200 ]; check $? "  ... nor another address that signed in before"
stop
start || { echo "FAIL the proxy did not restart"; exit 1; }
for _ in $(seq 30); do [ "$(L "https://127.0.0.1:$lport/")" != 000 ] && break; sleep 0.1; done
for a in 2 3 4 5 6; do for i in 1 2 3 4; do L --interface 127.0.0.$a -u "root:y-$a-$i-password" "https://127.0.0.1:$lport/1bit/update/status" > /dev/null; done; done
[ "$(L --interface 127.0.0.1 -u "root:$PW" "https://127.0.0.1:$lport/1bit/update/status")" = 200 ]; check $? "after a restart the known address still is (kept in the file), under the same flood"
stop
unset ONEBIT_WEBAPP_KNOWN ONEBIT_WEBAPP_TEST_OK_TTL

# 4. no credential: no LAN page at all
rm -f "$t/cred"; start || { echo "FAIL the proxy did not start"; exit 1; }
sleep 0.5
[ "$(L "https://127.0.0.1:$lport/")" = 000 ]; check $? "no credential: nothing listens on the LAN port"
stop

echo "webapp_update_lan_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
