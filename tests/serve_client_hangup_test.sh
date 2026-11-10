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
# Copyright 2026 The 1bit-MONSTER Authors.
# SPDX-License-Identifier: Apache-2.0
#
# serve_client_hangup_test -- engine#1779 fault 1: "a server that stopped accepting connections after clients hung up".
#
# The fault, as FastFlowLM v1.0.12 states it, is that a server stops accepting after clients disconnect mid-response.
# In our tree that would be the per-connection teardown: a handler that keeps waiting on the backend after its client is
# gone holds its thread, and enough of those exhaust the accept capacity. `app/serve.cpp` already stops the backend when
# a NON-streaming client hangs up (the `is_connection_closed()` poll); the streaming path and the teardown after it are
# what this test exercises.
#
# The assertion is deliberately about the LISTENER, not about the interrupted request: a client that disconnects is
# allowed to get nothing. What must not happen is that the NEXT client goes unserved.
#
# Device-free by construction: pass `cpu` and it needs no GPU (that is the point of engine#1779's device rule).
#
# usage: tests/serve_client_hangup_test.sh path/to/1bit <model> <device> [rounds] [extra serve args]
#   exits 0 when a fresh client is served after the hangups, 1 when it is not, 77 when it cannot run here.
set -uo pipefail

bin=${1:?usage: serve_client_hangup_test.sh path/to/1bit <model> <device> [rounds] [args]}
here=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=tests/serve_port.sh
. "$(dirname "$0")/serve_port.sh"   # serve_port_wait: read back the port serve bound (engine #1485)
model=${2:?}
device=${3:?}
rounds=${4:-6}
[ "$rounds" -ge 1 ] 2>/dev/null || rounds=6
shift 3
[ $# -gt 0 ] && [ "${1#-}" = "$1" ] 2>/dev/null && shift   # the optional rounds argument, when given

[ -x "$bin" ] || { echo "skip: no 1bit at $bin"; exit 77; }
[ -f "$model" ] || { echo "skip: no model at $model"; exit 77; }
command -v curl > /dev/null || { echo "skip: no curl"; exit 77; }

if [ "$device" = hrx ]; then
    have=$("$here/scripts/gpu-targets.sh" 2>/dev/null | tr '\n' ' ')
    want=${ONEBIT_HRX_E2E_TARGETS:-gfx1151}
    found=0
    for t in $want; do case " $have " in *" $t "*) found=1 ;; esac; done
    [ $found = 1 ] || { echo "skip: no $want GPU for HRX on this machine (GPUs: ${have:-none})"; exit 77; }
fi

log=$(mktemp)
srv=
cleanup() {
    if [ -n "$srv" ] && kill -0 "$srv" 2> /dev/null; then
        kill "$srv" 2> /dev/null
        for _ in $(seq 1 50); do kill -0 "$srv" 2> /dev/null || break; sleep 0.1; done
        kill -9 "$srv" 2> /dev/null
    fi
    # the child llama-server is PR_SET_PDEATHSIG'd to serve (app/serve.cpp), so it goes with it
    rm -f "$log"
}
trap cleanup EXIT INT TERM

"$bin" serve -m "$model" --device "$device" --host 127.0.0.1 --port 0 "$@" > "$log" 2>&1 &
srv=$!
# serve_port_wait <log> <pid> <tenths> prints the port serve reported; the pid is there so a serve that dies stops the
# wait instead of sleeping out the budget (my first version passed a file path here and broke out on the first poll)
if ! port=$(serve_port_wait "$log" "$srv" 24000); then
    echo "skip: serve did not report a port in 240 s (see below)"; tail -20 "$log"; exit 77
fi
# A build that predates engine #1485's read-back prints `ready on http://127.0.0.1:0` when it was asked for port 0,
# while the backend it spawned prints the port it actually bound. Take that one rather than curling port 0.
if [ "$port" = "0" ]; then
    child_port=$(awk '/llama_server: listening on http:\/\//{ n = split($NF, a, ":"); if (a[n] ~ /^[0-9]+$/ && a[n] != "0") { print a[n]; exit } }' "$log")
    if [ -n "$child_port" ]; then
        echo "note: serve reported port 0 (engine #1485's read-back is not in this build); using the backend's $child_port"
        port=$child_port
    fi
fi
echo "serve on 127.0.0.1:$port, device=$device, $rounds rounds"

payload='{"model":"x","messages":[{"role":"user","content":"Count slowly from one to two hundred."}],"max_tokens":200,"stream":true}'
body_nostream='{"model":"x","messages":[{"role":"user","content":"Count slowly from one to two hundred."}],"max_tokens":200}'

# Interrupt `rounds` streaming requests after a few bytes: `head -c 32` closes the pipe, curl gets EPIPE, the
# connection goes away while the backend is still producing.
interrupted=0
for i in $(seq 1 "$rounds"); do
    curl -sN -m 30 -H 'Content-Type: application/json' -d "$payload" \
        "http://127.0.0.1:$port/v1/chat/completions" 2> /dev/null | head -c 32 > /dev/null
    interrupted=$((interrupted + 1))
done

# ... and two non-streaming ones, cut off by a hard client timeout (the path app/serve.cpp polls for).
for i in 1 2; do
    curl -s -m 1 -H 'Content-Type: application/json' -d "$body_nostream" \
        "http://127.0.0.1:$port/v1/chat/completions" > /dev/null 2>&1
    interrupted=$((interrupted + 1))
done
echo "interrupted $interrupted requests mid-response"

# The assertion: the NEXT client is served. Generous timeout: a fresh prompt on a CPU backend is slow, and a dial
# failure is the fault we are looking for, not slowness.
code=$(curl -s -m 120 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"Say Paris."}],"max_tokens":8,"temperature":0}' \
    "http://127.0.0.1:$port/v1/chat/completions" 2> /dev/null)
echo "fresh client after the hangups: HTTP ${code:-none}"

if [ "$code" = "200" ]; then
    echo "PASS serve_client_hangup: a fresh client is served after $interrupted interrupted requests"
    exit 0
fi
echo "FAIL serve_client_hangup: the listener stopped serving after $interrupted interrupted requests (HTTP ${code:-none})"
echo "--- last 25 lines of the serve log:"
tail -25 "$log"
exit 1
