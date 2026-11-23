#!/bin/bash
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
# release_check_hold_cycle_test.sh <release-check.sh> - the `hold` row's live cycle (engine#3350/#3352), run from the
# script itself against a stub curl that answers exactly as Lemonade with 1bit-MONSTER/lemonade#58 does (measured:
# lemond aa6c7520 on ryzen). The v2026.49-rc1 QEMU run FAILed the row with "POST /api/v1/hold answered 200 ": the POST
# helper prints "<status> <Retry-After>", and the row compared that whole string to 200. No network, no device.
set -u
RC=${1:?usage: release_check_hold_cycle_test.sh <path-to-release-check.sh>}
[ -r "$RC" ] || { echo "not readable: $RC"; exit 1; }
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
fail=0
ok() { echo "ok   $*"; }
no() { echo "FAIL $*"; fail=1; }

PWO=$(awk '/^post_watch_once\(\) \{/{f=1} f{print} f&&/^\}$/{exit}' "$RC")
HV=$(awk '/^hold_verdict\(\) \{/{f=1} f{print} f&&/^\}$/{exit}' "$RC")
ROW=$(awk '/^# engine#3350: Lemonade can hold the models for heavy work/{f=1} f{print} f&&/^esac$/{exit}' "$RC")
[ -n "$PWO" ] && [ -n "$HV" ] && [ -n "$ROW" ] || { echo "FAIL could not find post_watch_once / hold_verdict / the hold row in $RC"; exit 1; }

# Lemonade #58's answers (bodies copied from a real lemond aa6c7520): state lives in a file so the cycle can move
cat > "$t/curl" <<'STUB'
#!/bin/bash
st=$STUBDIR/active; out=; hdr=; body=; url=
while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift 2 ;; -D) hdr=$2; shift 2 ;; --data-binary) body=$2; shift 2 ;;
    -m|-H|-w) shift 2 ;; -*) shift ;; *) url=$1; shift ;; esac; done
case "$url" in
  */api/v1/hold/release) rm -f "$st"; r='{"expired":false,"failed":[],"hold_id":"hold-1","reloaded":[]}' ;;
  */api/v1/hold) if [ -n "$body" ]; then : > "$st"; r='{"hold_id":"hold-1","max_seconds":30,"reason":"release-check hold row","unloaded":[]}'
                 elif [ -e "$st" ]; then r='{"active":true,"hold_id":"hold-1","max_seconds":30,"phase":"holding","reason":"release-check hold row","since_s":0.002,"unloaded":[]}'
                 else r='{"active":false}'; fi ;;
  *) r='{}' ;;
esac
[ -n "$hdr" ] && printf 'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n' > "$hdr"
if [ -n "$out" ]; then printf '%s' "$r" > "$out"; printf '200'; else printf '%s' "$r"; fi
STUB
chmod +x "$t/curl"
export STUBDIR=$t
run() {  # the row with the script's own helpers; auth_get/auth_status are the blank-mode (no key) shapes
    PATH=$t:$PATH bash -c '
        KEY= API=http://127.0.0.1:13305 _nd_pup=1
        auth_get() { curl -s "$1"; }
        auth_status() { printf 200; }
        pass() { echo "PASS $1 $2"; }; skip() { echo "SKIP $1 $2"; }; fail() { echo "FAIL $1 $2"; }
        '"$PWO"'
        '"$HV"'
        '"$ROW"'
    ' 2>&1
}
out=$(run)
case "$out" in
    "PASS hold hold -> active -> release -> inactive"*) ok "a Lemonade with the hold endpoint passes the full hold -> release cycle" ;;
    *) no "the hold row on a working Lemonade: $out" ;;
esac
[ ! -e "$t/active" ]; [ $? = 0 ] && ok "the row releases the hold it took (nothing left active)" || no "the row left the hold active"
echo "release_check_hold_cycle_test: $([ $fail = 0 ] && echo ok || echo FAIL)"
exit $fail
