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
# `1bit serve`'s GET /metrics (docs/serve.md, "Metrics"), without a GPU: tests/fake_backend.py
# stands in for llama-server, writes the log lines serve reads (graph splits, HRX graph-program
# cache flushes, a CPU-fallback report) and answers /metrics with one llamacpp:* series.
#   - what the backend's log said, and nothing for a line that never came,
#   - GTT and Tctl where the box has them,
#   - the backend's own /metrics appended,
#   - starts and watchdog kills kept across serve processes,
#   - no request / TTFT / tok-s series (Lemonade reports those),
#   - every line is valid Prometheus text.
#
# usage: tests/metrics_serve.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: metrics_serve.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
touch "$scratch/tiny.gguf"
pids=()
cleanup() {
    for p in "${pids[@]}"; do pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done
    wait 2>/dev/null
    statefile | xargs -r rm -f
    rm -rf "$scratch"
}
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
newport() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
export FAKE_BACKEND_EVENTS="$scratch/events"
# a model id of its own: the start counts live in ~/.local/state/1bit/serve/, removed at exit
alias=metrics-test-$$-$RANDOM
statefile() { grep -l "\"model\":\"$alias\"" "$(getent passwd "$(id -u)" | cut -d: -f6)"/.local/state/1bit/serve/*.json 2>/dev/null; }

start() {  # <log> [serve args]
    port=$(newport)
    "$bin" serve -m "$scratch/tiny.gguf" --device cpu --port "$port" --alias "$alias" \
        --llama-server "$here/fake_backend.py" "${@:2}" >"$1" 2>&1 &
    pid=$!
    pids+=("$pid")
    api="http://127.0.0.1:$port"
    for _ in $(seq 1 100); do [ "$(curl -s -o /dev/null -w '%{http_code}' "$api/health")" = 200 ] && break; sleep 0.1; done
}
metric() {  # <text file> <series with labels>: its value, or "absent"
    python3 - "$1" "$2" <<'PY'
import sys
want = sys.argv[2]
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    name, _, value = line.rstrip("\n").rpartition(" ")
    if name == want:
        print(value)
        break
else:
    print("absent")
PY
}
valid() {  # every line is a comment or `name{labels} value`
    python3 - "$1" <<'PY'
import re, sys
ok = re.compile(r'^(# (HELP|TYPE) .*|[a-zA-Z_:][a-zA-Z0-9_:]*(\{([a-zA-Z_][a-zA-Z0-9_]*="([^"\\]|\\.)*",?)*\})? [-+0-9.eEInfa]+)$')
bad = [l for l in open(sys.argv[1]).read().splitlines() if not ok.match(l)]
if bad: print("\n".join(bad[:5]))
sys.exit(1 if bad else 0)
PY
}

FAKE_BACKEND_LOG_LINES=1 start "$scratch/a.log"
curl -s -o /dev/null "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "hi"}]}'
type=$(curl -s -o "$scratch/m1.txt" -w '%{content_type}' "$api/metrics")
m=$scratch/m1.txt
check "text/plain exposition ($type)" '[[ "$type" == text/plain* ]]'
check "valid Prometheus text" 'valid "$m"'
check "onebit_up names the device" '[ "$(metric "$m" "onebit_up{device=\"cpu\"}")" = 1 ]'
check "graph splits from the backend log" '[ "$(metric "$m" "onebit_graph_splits{batch=\"prompt\"}")" = 3 ] && [ "$(metric "$m" "onebit_graph_splits{batch=\"1\"}")" = 2 ]'
check "program-cache flushes from the WARN" '[ "$(metric "$m" "onebit_program_cache_flushes_total")" = 2 ]'
check "CPU fallbacks from the report line" '[ "$(metric "$m" "onebit_cpu_fallback_ops{op=\"all\"}")" = 3 ] && [ "$(metric "$m" "onebit_cpu_fallback_ops{op=\"GET_ROWS\"}")" = 2 ]'
check "the backend's own /metrics appended" '[ "$(metric "$m" "llamacpp:prompt_tokens_total")" = 12 ]'
check "first start" '[ "$(metric "$m" "onebit_serve_starts_total")" = 1 ]'
check "no request, TTFT or tok/s series (Lemonade's)" '! grep -q -E "^onebit_(requests|ttft|prompt_tokens|generated|last_)" "$m"'
if ls /sys/class/drm/card*/device/mem_info_gtt_total >/dev/null 2>&1; then
    check "GTT from amdgpu sysfs" '[ "$(metric "$m" "onebit_gtt_bytes{kind=\"total\"}")" -gt 0 ]'
fi
if grep -qs k10temp /sys/class/hwmon/*/name; then
    check "Tctl from k10temp" '[ "$(metric "$m" "onebit_tctl_celsius")" != absent ]'
fi
check "backend log lines still reach serve's stderr" 'grep -q "graph splits = 3" "$scratch/a.log"'
kill "$pid"; wait "$pid" 2>/dev/null

# Lemonade starts serve again: the start counts carry over; no report line, no fallback metric.
start "$scratch/b.log" --watchdog 1
curl -s -o "$scratch/m2.txt" "$api/metrics"
m=$scratch/m2.txt
check "second start counted" '[ "$(metric "$m" "onebit_serve_starts_total")" = 2 ]'
check "no CPU-fallback line, no CPU-fallback metric" '[ "$(metric "$m" "onebit_cpu_fallback_ops{op=\"all\"}")" = absent ]'
check "no splits line, no splits metric" '[ "$(metric "$m" "onebit_graph_splits{batch=\"prompt\"}")" = absent ]'
curl -sN -m 20 -o /dev/null "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"stream": true, "fake_hang": true, "messages": [{"role": "user", "content": "hi"}]}' &
pids+=($!)
for _ in $(seq 1 150); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
check "the watchdog took the hung backend down" '! kill -0 "$pid" 2>/dev/null'

start "$scratch/c.log"
curl -s -o "$scratch/m3.txt" "$api/metrics"
m=$scratch/m3.txt
check "the watchdog kill is counted after the restart" '[ "$(metric "$m" "onebit_watchdog_kills_total")" = 1 ] && [ "$(metric "$m" "onebit_serve_starts_total")" = 3 ]'

if [ $fail -ne 0 ]; then for l in a b c; do echo "--- serve log $l"; cat "$scratch/$l.log"; done; echo "--- metrics"; cat "$scratch"/m*.txt; echo FAIL; exit 1; fi
echo PASS
