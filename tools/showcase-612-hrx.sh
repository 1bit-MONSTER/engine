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

# engine#612, row 2: what `1bit serve --parallel N` actually buys on HRX. One server load per N,
# and each load measures solo (one stream) against N streams at once, so both numbers come from the
# same weights and the same clocks. Wall clock per request, tokens from usage.completion_tokens,
# greedy, distinct prompts, REPEATS runs reported as a median with min and max.
#
#   BIN=... tools/showcase-612-hrx.sh 1 2 4 8        # the parallel counts to measure
#
# Every GPU command in this lane runs under gpu-hold (docs/GPU-TOKEN.md in the fleet tree), so wrap
# the call:  ~/.local/bin/gpu-hold -l "<label>" -t 20 -- tools/showcase-612-hrx.sh 2 4 8
# The servers are killed by pid; nothing here uses pkill.
set -uo pipefail

BIN=${BIN:-${PWD}/build/1bit}
LLAMA_SERVER=${LLAMA_SERVER:-${PWD}/build/hrx/llama/bin/llama-server}
GGUF=${GGUF:-$HOME/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}
CTX=${CTX:-8192}
PORT=${PORT:-18081}
NPRED=${NPRED:-128}
REPEATS=${REPEATS:-5}
TAG=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$HOME/lb/612-hrx-$TAG
mkdir -p "$OUT" || exit 1
LOG=$OUT/run.log
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

[ -x "$BIN" ] || { echo "no binary at $BIN (set BIN)"; exit 1; }
[ -x "$LLAMA_SERVER" ] || { echo "no HRX llama-server at $LLAMA_SERVER (set LLAMA_SERVER)"; exit 1; }
[ -f "$GGUF" ] || { echo "no model at $GGUF (set GGUF)"; exit 1; }

say "pin: $("$BIN" --version 2>&1 | head -1) at $BIN"
say "llama-server: $("$LLAMA_SERVER" --version 2>&1 | grep -o 'version: [0-9]* ([0-9a-f]*)' | head -1)"
say "model: $(basename "$GGUF") $(stat -Lc %s "$GGUF") B sha256 $(sha256sum "$GGUF" | cut -d' ' -f1)"
say "instrument: wall clock per stream + usage.completion_tokens, n_predict=$NPRED temp=0 repeats=$REPEATS port=$PORT"

for NP in "$@"; do
    log=$OUT/server-np$NP.log
    : > "$log"
    say "=== --parallel $NP"
    HSA_USERPTR_FOR_PAGED_MEM=0 "$BIN" serve -m "$GGUF" --device hrx --ctx-size "$CTX" --parallel "$NP" \
        --port "$PORT" --llama-server "$LLAMA_SERVER" >> "$log" 2>&1 &
    srv=$!
    up=0
    for _ in $(seq 1 300); do
        curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && { up=1; break; }
        kill -0 "$srv" 2>/dev/null || break
        sleep 2
    done
    if [ "$up" != 1 ]; then
        say "server did not come up (--parallel $NP); last lines:"
        tail -5 "$log" | tee -a "$LOG"
        kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
        continue
    fi
    say "up: $(grep -o 'n_slots = [0-9]*' "$log" | head -1) $(grep -o "kv_unified = '[a-z]*'" "$log" | head -1)"
    python3 - "$PORT" "$NPRED" "$REPEATS" "$NP" "$OUT" <<'PY' | tee -a "$LOG"
import json, statistics, sys, threading, time, urllib.request

port, ntok, reps, np_, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
PROMPTS = ["Write one sentence about the sea.", "Name three uses for copper.",
           "Explain what a tide is.", "Describe a paperclip in one sentence.",
           "What does a barometer measure?", "Give one reason bridges are expensive.",
           "Describe rain in one sentence.", "What is a sundial?"]


def one(idx, results):
    body = json.dumps({"prompt": PROMPTS[idx % len(PROMPTS)], "max_tokens": ntok,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
        wall = time.time() - t0
        n = (d.get("usage") or {}).get("completion_tokens", 0)
        results[idx] = {"wall": wall, "tokens": n}
    except Exception as ex:                       # a dead stream invalidates the run
        results[idx] = {"error": repr(ex)}


def arm(n):
    res = [None] * n
    ths = [threading.Thread(target=one, args=(i, res)) for i in range(n)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    bad = [r for r in res if r is None or "error" in r]
    if bad:
        raise SystemExit("streams=%d INVALID: %s" % (n, bad[0]))
    tokens = sum(r["tokens"] for r in res)
    return {"streams": n, "wall": wall, "tokens": tokens,
            "aggregate": tokens / wall if wall else 0.0,
            "per_stream": statistics.mean(r["tokens"] / r["wall"] for r in res)}


def med(rows, key):
    v = [r[key] for r in rows]
    return statistics.median(v), min(v), max(v)


for n in (1, np_):
    rows = [arm(n) for _ in range(reps)]
    agg, lo, hi = med(rows, "aggregate")
    ps, _, _ = med(rows, "per_stream")
    wall, _, _ = med(rows, "wall")
    print("parallel=%d streams=%d repeats=%d aggregate_tps=%.2f (min %.2f max %.2f) "
          "per_stream_tps=%.2f wall_s=%.2f timestamp=%s"
          % (np_, n, reps, agg, lo, hi, ps, wall, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
PY
    kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
    sleep 5
done
say "done; $OUT"
