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

# The default budget is 90 s, not 30: `forward-serve` warms its weight cache before printing `ready on` (engine#1621),
# measured at 44.5 s on a folded 27B pack, and pays it on exactly the configurations these tests exercise. A caller that
# wants the old bound can still pass it, and a log now says `warming` before the wait, so a short budget fails with a
# readable line rather than with "serve never reported a port".
serve_port_wait() {  # <log> <pid> [tenths-of-a-second, default 900 = 90 s] -> the port, non-zero if it never came
    local log=$1 pid=$2 tries=${3:-900} port=
    for _ in $(seq 1 "$tries"); do
        port=$(serve_port_from_log "$log")
        if [ -n "$port" ]; then printf '%s\n' "$port"; return 0; fi
        # serve died: no readiness line is ever coming, so stop waiting rather than sleeping out the whole budget
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
    done
    return 1
}

# Ending a serve is not the same as killing the process the shell started: `1bit serve` is a WRAPPER, and the model —
# with the GTT it holds — lives in the `llama-server` it spawns as a child. Seen live on this box:
#
#   4642  ppid 2510  1bit serve -m … --device hrx
#   4644  ppid 4642  …/hrx/llama/bin/llama-server -m … --device HRX0 -ngl 99
#
# so `kill "$pid"; wait "$pid"` signals the wrapper, reaps the wrapper, and leaves the process that actually holds the
# memory — the orphan shape from 11-08, one level down, produced by a test's own teardown (engine #1653 review). This is
# the same conclusion `scripts/gpu-hold` reached in kill_tree (engine #1624): walk the tree, and snapshot it BEFORE
# signalling, because once the root dies its children are reparented and a `ps --ppid` walk finds only the corpse.
serve_descendants() {  # <pid> -> its descendants and then itself, deepest last
    local k
    for k in $(ps -o pid= --ppid "$1" 2> /dev/null); do
        serve_descendants "$k"
    done
    printf '%s\n' "$1"
}

serve_kill_tree() {  # <pid> [grace-seconds, default 10] -> TERM the whole tree, KILL whatever ignored it
    local root=$1 grace=${2:-10} i=0 p tree alive
    tree=$(serve_descendants "$root")
    for p in $tree; do kill -TERM "$p" 2> /dev/null; done
    while [ "$i" -lt "$grace" ]; do
        alive=0
        for p in $tree; do
            if kill -0 "$p" 2> /dev/null; then alive=1; break; fi
        done
        [ "$alive" = 1 ] || break
        sleep 1
        i=$((i + 1))
    done
    for p in $tree; do kill -KILL "$p" 2> /dev/null; done
}

# `llama-server` -- the vendored llama.cpp, which the offload-parity scripts drive directly rather than through
# `1bit serve` -- prints the same fact in its own words (engine #1485). llama.cpp binds an ephemeral port when
# --port is 0 (`bind_to_any_port`, tools/server/server-http.cpp:444) and logs the address it ended up on:
#   srv  http_bind: listening on http://127.0.0.1:41235
# so the port is the field after the URL token's last colon, and the same IPv6 reasoning applies.
llama_port_from_log() {  # <log> -> the port, empty while llama-server has not said
    awk '/listening on http:\/\//{ for (i = 1; i <= NF; i++) if ($i ~ /^http:\/\//) { n = split($i, a, ":"); if (a[n] ~ /^[0-9]+$/) { print a[n]; exit } } }' "$1" 2>/dev/null
}

llama_port_wait() {  # <log> <pid> [tenths-of-a-second, default 300 = 30 s] -> the port, non-zero if it never came
    local log=$1 pid=$2 tries=${3:-300} port=
    for _ in $(seq 1 "$tries"); do
        port=$(llama_port_from_log "$log")
        if [ -n "$port" ]; then printf '%s\n' "$port"; return 0; fi
        # the server died: no listening line is ever coming, so stop rather than sleeping out the budget
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
    done
    return 1
}
