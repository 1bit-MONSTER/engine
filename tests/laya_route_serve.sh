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
# `1bit route --serve` (docs/laya-first.md §8 step 1): the resident decision service. The one-shot
# `1bit route` loads the scorer for every request (about half a second on the CPU scorer), so the
# fork cannot ask it per request on a cold Lemonade; here the scorer is loaded once and one decision
# is answered per stdin line. The test proves:
#   - the service announces {"ready":true,"load_ms":N} before it reads anything,
#   - a decision agrees with `1bit route --classify` for the same request (same class, same device),
#   - the resident decision is cheap: decision_ms < load_ms (that is the whole point),
#   - the same request asked 5 times gets 5 identical decisions (determinism),
#   - a state of kLongDocChars (1024) or more is long_doc with confidence 1.0 without asking the model,
#   - a line that is not {"state": "<text>"} is answered with {"error": ...} and the service keeps serving,
#   - EOF on stdin, and SIGTERM, both end the service with status 0 (a dead daemon is a clean exit,
#     not a hang: the fork's client restarts it on the next request).
#
# usage: tests/laya_route_serve.sh path/to/1bit path/to/laya-model-dir
set -uo pipefail

bin=${1:?usage: laya_route_serve.sh path/to/1bit path/to/laya-model-dir}
laya_model=${2:?usage: laya_route_serve.sh path/to/1bit path/to/laya-model-dir}
here=$(cd "$(dirname "$0")" && pwd)
policy="$here/route-policy-e2e.json"
# the candidate devices serve offers Laya for a .gguf in this build (tests/laya_route_e2e.sh)
devices=cpu
if grep -q "^ONEBIT_HRX:BOOL=ON" "$(dirname "$bin")/CMakeCache.txt" 2>/dev/null; then devices=hrx,cpu; fi
echo "candidates: $devices"

scratch=$(mktemp -d)
out="$scratch/out.jsonl"
err="$scratch/err.log"
one_shot="$scratch/one-shot.txt"
mkfifo "$scratch/in"
pid=
cleanup() {
    exec 3>&- 2>/dev/null
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    rm -rf "$scratch"
}
trap cleanup EXIT

"$bin" route --serve --laya-model "$laya_model" --devices "$devices" --route-policy "$policy" \
    <"$scratch/in" >"$out" 2>"$err" &
pid=$!
exec 3>"$scratch/in"  # hold the write end open: closing it is what EOF means to the service

wait_lines() { for _ in $(seq 1 600); do [ "$(wc -l <"$1")" -ge "$2" ] && return 0; sleep 0.1; done; return 1; }

state="Write a Python function that reverses a linked list."
send() { printf '%s\n' "$1" >&3 2>/dev/null || true; }

wait_lines "$out" 1 || { echo "FAIL the service never announced itself"; cat "$err"; exit 1; }

send "{\"state\": \"$state\"}"
wait_lines "$out" 2
"$bin" route --laya-model "$laya_model" --devices "$devices" --route-policy "$policy" \
    --state "$state" --classify >"$one_shot" 2>/dev/null

# the same request 5 times: determinism, and the resident case the fork actually calls
for _ in 1 2 3 4 5; do send "{\"state\": \"$state\"}"; done
wait_lines "$out" 7
# a malformed line is answered, not fatal
send 'not json'
wait_lines "$out" 8
# kLongDocChars (laya/route.h) or more: long_doc without asking the model
long_state=$(python3 -c 'print("log line " * 200)')
send "{\"state\": \"$long_state\"}"
wait_lines "$out" 9

# a dead daemon comes in two shapes and they are not the same thing: the client closing its pipe
# (EOF: the service exits 0) and a signal (SIGTERM: the process is gone, nothing is written after,
# and a client must not hang on it).
before=$(wc -l <"$out")
exec 3>&-  # EOF on stdin
for _ in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
alive=0; kill -0 "$pid" 2>/dev/null && alive=1
wait "$pid"; eof_rc=$?
pid=
after=$(wc -l <"$out")

fail=0
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; fail=1; fi; }
[ "$alive" = 0 ]; check $? "EOF on stdin ends the service (still alive=$alive)"
[ "$eof_rc" = 0 ]; check $? "a client that closes the pipe gets exit 0 (rc=$eof_rc)"
[ "$before" = "$after" ]; check $? "no line is written after EOF ($before -> $after)"

# A daemon that is killed rather than closed: gone at once, nothing written half-way, and the
# status says SIGTERM (128+15) instead of a hang.
out2="$scratch/out2.jsonl"
mkfifo "$scratch/in2"
"$bin" route --serve --laya-model "$laya_model" --devices "$devices" --route-policy "$policy" \
    <"$scratch/in2" >"$out2" 2>>"$err" &
pid=$!
exec 3>"$scratch/in2"
wait_lines "$out2" 1
send "{\"state\": \"$state\"}"
wait_lines "$out2" 2
before2=$(wc -l <"$out2")
kill -TERM "$pid" 2>/dev/null
for _ in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
alive2=0; kill -0 "$pid" 2>/dev/null && alive2=1
wait "$pid"; term_rc=$?
pid=
exec 3>&-
after2=$(wc -l <"$out2")
[ "$alive2" = 0 ]; check $? "SIGTERM removes the service at once (still alive=$alive2)"
[ "$term_rc" = 143 ] || [ "$term_rc" = 0 ]; check $? "a killed daemon reports SIGTERM, not a hang (rc=$term_rc)"
[ "$before2" = "$after2" ]; check $? "no line is written after SIGTERM ($before2 -> $after2)"

python3 - "$out" "$one_shot" "$devices" <<'PY'
import json, sys

out, one_shot_path, devices_csv = sys.argv[1], sys.argv[2], sys.argv[3]
devices = devices_csv.split(",")
raw = [l for l in open(out) if l.strip()]
msgs = [json.loads(l) for l in raw]  # every line must be one JSON object
fail = 0


def check(ok, label):
    global fail
    print(("ok   " if ok else "FAIL ") + label)
    if not ok:
        fail = 1


ready = msgs[0]
check(ready.get("ready") is True and isinstance(ready.get("load_ms"), (int, float)),
      "first line is the handshake {ready:true, load_ms:%s}" % ready.get("load_ms"))

cls = msgs[1:7]    # the same request, six times
long_doc = msgs[8]  # after the malformed line at index 7
error = msgs[7]

check(all(isinstance(m.get("decision_ms"), (int, float)) for m in cls),
      "a decision carries decision_ms")
check(all(m.get("device") in devices for m in cls),
      "the device is one of the candidates (%s)" % devices_csv)
check(cls[0].get("decision_ms", 0) < ready.get("load_ms", 0),
      "resident decisions are cheaper than the scorer load (%s ms vs %s ms)" %
      (round(cls[0].get("decision_ms", 0), 2), round(ready.get("load_ms", 0), 2)))

want = open(one_shot_path).read().split()
one_shot = "%s %.2f %s" % (want[0], float(want[1]), want[2]) if len(want) == 3 else ""
served = "%s %.2f %s" % (cls[0]["class"], cls[0]["confidence"], cls[0]["device"])
check(served == one_shot, "the served decision matches `1bit route --classify` (%r vs %r)" % (served, one_shot))

keys = ("class", "confidence", "device")
same = all({k: m.get(k) for k in keys} == {k: cls[0].get(k) for k in keys} for m in cls[1:])
check(same, "the same request 5 times gives 5 identical decisions")

check(error.get("error") is not None, "a malformed line is answered with an error (%s)" % str(error.get("error"))[:40])
check(long_doc.get("class") == "long_doc" and long_doc.get("confidence") == 1.0,
      "a 1800-character state is long_doc at confidence 1.0 without the model (%s)" % long_doc.get("class"))

sys.exit(fail)
PY
check $? "all protocol assertions"

if [ "$fail" = 0 ]; then echo "PASS"; else echo "--- stderr of the service:"; cat "$err"; fi
exit "$fail"
