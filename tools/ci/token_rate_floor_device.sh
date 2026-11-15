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
# The caller for `token_rate_floor.py --log`. Without it the tool has a self-test and no consumer, which is
# engine#2418's defect: arms that are green and never applied to a real run (engine#2432's review, item 1).
#
# Judges ONE sequential `1bit serve` generation against TOKEN_RATE_FLOOR (default 0 = record only). Refuses
# with 77 -- CTest's skip code -- when it cannot run, and says which prerequisite is missing: a device row
# that silently passes because there was no model would be exactly the failure this tool exists to catch.
set -u
ROOT=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
FLOOR=${TOKEN_RATE_FLOOR:-0}
PORT=${TOKEN_RATE_PORT:-8191}
SKIP=77

BIN=${TOKEN_RATE_BIN:-}
for cand in "$ROOT/build/1bit" "$ROOT/b2180/1bit"; do [ -n "$BIN" ] || { [ -x "$cand" ] && BIN=$cand; }; done
[ -n "$BIN" ] || BIN=$(command -v 1bit 2>/dev/null || true)
MODEL=${TOKEN_RATE_MODEL:-}
[ -n "$MODEL" ] || MODEL=$(ls "$HOME"/models/Qwen3-0.6B-Q4_K_M.gguf 2>/dev/null | head -1)
HOLD=${TOKEN_RATE_GPU_HOLD:-$(command -v gpu-hold || echo "$HOME/.local/bin/gpu-hold")}

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "SKIPPED: token_rate_floor device row did NOT run -- no 1bit binary (set TOKEN_RATE_BIN)"; exit $SKIP
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "SKIPPED: token_rate_floor device row did NOT run -- no model (set TOKEN_RATE_MODEL)"; exit $SKIP
fi
if [ ! -x "$HOLD" ]; then
    echo "SKIPPED: token_rate_floor device row did NOT run -- gpu-hold not found ($HOLD), and a GPU run outside a ticket is not allowed"; exit $SKIP
fi

LOG=$(mktemp -t token-rate.XXXXXX.log)
cleanup() {
    local p; p=$(pgrep -f "port $PORT" | head -1)
    [ -n "$p" ] && { kill -TERM "$p" 2>/dev/null; for _ in $(seq 1 20); do kill -0 "$p" 2>/dev/null || break; sleep 1; done; }
    rm -f "$LOG"
}
trap cleanup EXIT

# The whole run lives inside the ticket: the process ends when the ticket ends.
"$HOLD" -t 10 -w 900 -- bash -c '
    set -u
    nohup "$1" serve -m "$2" --device hrx --host 127.0.0.1 --port "$3" --ctx-size 2048 --alias p --no-strict-device >/dev/null 2>&1 &
    for _ in $(seq 1 40); do curl -sf -m 3 "http://127.0.0.1:$3/health" >/dev/null 2>&1 && break; sleep 3; done
    curl -sf -m 5 "http://127.0.0.1:$3/health" >/dev/null || { echo "SKIPPED: server never became ready"; exit 77; }
    body=$(curl -s -m 300 "http://127.0.0.1:$3/v1/chat/completions" -H "content-type: application/json" \
      -d "{\"model\":\"p\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about rivers.\"}],\"max_tokens\":64,\"temperature\":0}")
    case "$body" in *'"error"'*) echo "SKIPPED: server returned an error: $(printf %s "$body" | head -c 120)"; exit 77;; esac
    printf %s "$body" | python3 -c "
import json,sys,time
d=json.load(sys.stdin); c=(d.get(\"usage\") or {}).get(\"completion_tokens\") or 0
print(f\"{c} tokens\")" >/dev/null 2>&1
    t0=$(date +%s.%N); body=$(curl -s -m 300 "http://127.0.0.1:$3/v1/chat/completions" -H "content-type: application/json" \
      -d "{\"model\":\"p\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about rivers.\"}],\"max_tokens\":64,\"temperature\":0}"); t1=$(date +%s.%N)
    case "$body" in *'"error"'*) echo "SKIPPED: server returned an error on the measured run"; exit 77;; esac
    printf %s "$body" | python3 -c "
import json,sys
w=float(sys.argv[1]); d=json.load(sys.stdin); c=(d.get(\"usage\") or {}).get(\"completion_tokens\") or 0
print(f\"{c/w:.2f} tokens per second\" if c>0 and w>0 else \"\")" "$(echo "$t1 - $t0" | bc)"
' _ "$BIN" "$MODEL" "$PORT" > "$LOG" 2>&1
rc=$?

if [ $rc -eq $SKIP ] || ! grep -q "tokens per second" "$LOG" 2>/dev/null; then
    echo "SKIPPED: token_rate_floor device row did NOT run -- $(tail -1 "$LOG" 2>/dev/null | head -c 120)"; exit $SKIP
fi

# The rate goes through the tool, which owns the judging, the floor and the subject line.
exec python3 "$(dirname "$0")/token_rate_floor.py" --log "$LOG" --floor "$FLOOR" \
     --subject "1bit serve $(basename "$MODEL") ctx2048"
