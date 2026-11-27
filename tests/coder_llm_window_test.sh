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
# scripts/appliance/coder-llm-window (the PR reviewer's model in box-hold windows on the 1bit OS appliance) with a fake
# `1bit serve` that forks a grandchild and a fake box-hold that only records the ticket. No device, no network.
# Arms: no demand -> no ticket; demand -> one ticket, window opens, idle -> closes and the WHOLE serve chain is gone
# (grandchild included); max hold -> closes; demand stale by the time the ticket comes round -> no server; too little
# memory -> no server.
set -u
WIN=${1:?usage: coder_llm_window_test.sh <scripts/appliance/coder-llm-window>}
T=$(mktemp -d); trap 'kill $(cat "$T"/*.pid 2>/dev/null) 2>/dev/null; rm -rf "$T"' EXIT
fail=0; ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; fail=1; }
PORT=$(( 40000 + RANDOM % 20000 ))   # the window kills anything with "--port $PORT": never the real 8090
cat > "$T/1bit" <<'F'
#!/usr/bin/env bash
# fake `1bit serve`: a child that is the "llama-server" grandchild, then wait
( exec -a fake-llama-server-$FAKE_TAG sleep 600 ) &
echo $! > "$FAKE_DIR/grandchild.pid"
echo "serve $*" >> "$FAKE_DIR/serve.calls"
wait
F
cat > "$T/box-hold" <<'F'
#!/usr/bin/env bash
echo "ticket $1 who=$BOX_HOLD_WHO" >> "$FAKE_DIR/tickets"; shift; "$@"
F
chmod +x "$T/1bit" "$T/box-hold"
export FAKE_DIR=$T FAKE_TAG=$$ ONEBIT_BIN=$T/1bit BOX_HOLD=$T/box-hold CODER_LLM_DEMAND=$T/demand CODER_LLM_PORT=$PORT \
  CODER_LLM_LOGDIR=$T/log CODER_LLM_RUNDIR=$T/run MIN_FREE_GB=0 POLL_S=1 WATCH_S=1 STOP_WAIT_S=3 DEMAND_FRESH_S=3 IDLE_S=4
log=$T/log/window.log
alive() { pgrep -f "fake-llama-server-$FAKE_TAG" > /dev/null; }

# 1. no demand: the watcher takes no ticket
bash "$WIN" watch & echo $! > "$T/watch.pid"; sleep 4
[ ! -s "$T/tickets" ] && ok "no demand -> no ticket" || bad "no demand but a ticket was taken: $(cat "$T/tickets")"

# 2. demand -> one ticket, a window with the server; demand stops -> idle close, and the grandchild is gone too
touch "$T/demand"; for i in $(seq 20); do alive && break; sleep 0.5; done
alive && ok "demand -> window open, server running" || bad "demand but no server (log: $(tail -3 "$log"))"
grep -q "who=coder-llm" "$T/tickets" && grep -q "^ticket gpu" "$T/tickets" && ok "ticket is box-hold gpu, who=coder-llm" || bad "ticket: $(cat "$T/tickets" 2>/dev/null)"
grep -q -- "--port $PORT" "$T/serve.calls" && grep -q -- "--alias qwen3-coder" "$T/serve.calls" && grep -q -- "--device hrx" "$T/serve.calls" \
  && ok "serve line: hrx, alias qwen3-coder, port" || bad "serve line: $(cat "$T/serve.calls")"
for i in $(seq 30); do grep -q "closed.*idle" "$log" && break; sleep 0.5; done
grep -q "closed.*idle" "$log" && ok "idle -> window closed" || bad "window did not close on idle: $(tail -3 "$log")"
sleep 1; alive && bad "grandchild (llama-server) survived the window" || ok "whole serve chain gone after close"
[ "$(wc -l < "$T/tickets")" -eq 1 ] && ok "one ticket for one burst of demand" || bad "tickets: $(cat "$T/tickets")"
kill "$(cat "$T/watch.pid")"; wait 2>/dev/null; rm -f "$T/watch.pid"

# 3. max hold ends a busy window (demand kept fresh)
: > "$log"; ( while :; do touch "$T/demand"; sleep 1; done ) & echo $! > "$T/toucher.pid"
MAX_HOLD_S=3 bash "$WIN" window
grep -q "closed.*max hold" "$log" && ok "max hold -> window closed while demand is fresh" || bad "max hold: $(tail -3 "$log")"
sleep 1; alive && bad "grandchild survived max hold" || ok "chain gone after max hold"
kill "$(cat "$T/toucher.pid")"; rm -f "$T/toucher.pid"

# 4. demand went stale while queued: no server
: > "$log"; : > "$T/serve.calls"; touch -d '-1 hour' "$T/demand"; bash "$WIN" window
grep -q "stale while queued" "$log" && [ ! -s "$T/serve.calls" ] && ok "stale demand at ticket time -> no server" || bad "stale: $(cat "$log")"

# 5. not enough memory: no server
: > "$log"; touch "$T/demand"; MIN_FREE_GB=100000 bash "$WIN" window
grep -q "MemAvailable.*not starting" "$log" && [ ! -s "$T/serve.calls" ] && ok "low memory -> no server" || bad "memory floor: $(cat "$log")"

[ "$fail" = 0 ] && echo "coder_llm_window: PASS" || echo "coder_llm_window: FAIL"
exit "$fail"
