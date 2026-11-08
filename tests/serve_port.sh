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
# Reading back the port `serve` bound (engine #1485).
#
# The route tests used to pick a front port with bind(0), release it, and launch serve - which binds it a moment later,
# after starting its backends. On a `ctest -j` runner another serve can take the port inside that window, and the test
# then fails for something it does not test (engine #1444). `serve --port 0` removes the window: the kernel picks the
# port and serve reports it, so nothing here picks and releases anything.
#
#   . "$(dirname "$0")/serve_port.sh"
#   "$bin" serve ... --port 0 ... >"$log" 2>&1 &
#   pid=$!
#   port=$(serve_port_wait "$log" "$pid") || { echo "FAIL serve never reported a port"; cat "$log"; exit 1; }
#
# serve prints the ready line in three shapes (app/serve.cpp):
#   ready on http://HOST:PORT                                          (the plain path, no laya)
#   ready on http://HOST:PORT (laya routes each request; ...)          (the plain path, laya on)
#   ready on http://HOST:PORT (laya-models: model auto is routed ...)  (the laya-models path, always suffixed)
# so an end-anchored pattern is wrong: it matches only the first, and a helper meant to be shared would hand the next
# test an empty port, wait out its budget and fail *after* serve had reported one (reviewer's finding on #1523/#1525).
# Take the URL token (everything between "on " and the first space) and the port is the field after its LAST colon, which
# also keeps an IPv6 --host right: http://::1:41235 -> 41235.
serve_port_from_log() {  # <log> -> the port, empty while serve has not said
    awk '/^1bit serve: ready on http:\/\//{ n = split($5, a, ":"); if (a[n] ~ /^[0-9]+$/) { print a[n]; exit } }' "$1" 2>/dev/null
}

serve_port_wait() {  # <log> <pid> [tenths-of-a-second, default 300 = 30 s] -> the port, non-zero if it never came
    local log=$1 pid=$2 tries=${3:-300} port=
    for _ in $(seq 1 "$tries"); do
        port=$(serve_port_from_log "$log")
        if [ -n "$port" ]; then printf '%s\n' "$port"; return 0; fi
        # serve died: no readiness line is ever coming, so stop waiting rather than sleeping out the whole budget
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
    done
    return 1
}
