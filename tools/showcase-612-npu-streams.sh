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

# engine#612, row 1: two NPU engines at once, through the supported opt-in (`--npu-streams 2`,
# app/npu_device_lock.h). On Linux both engines are this runtime: MODEL and MODEL_B are two different
# models resident side by side, each measured solo and then together, REPEATS reps of 8 greedy tokens
# each. Every number carries its instrument, its repeat count and its timestamp; rep 1 of each arm is a
# warm-up and is reported but not averaged.
#
#   BIN=... MODEL=... MODEL_B=... tools/showcase-612-npu-streams.sh
#
# It waits for the NPU device lock to be free and never takes it from a running job: while another
# process holds /tmp/1bit-npu-device.lock exclusively (the owner's own serve, the ELF runner) this
# exits 75 and says who holds it. It kills only the servers it started, by pid.
set -uo pipefail

BIN=${BIN:-${PWD}/build/1bit}
MODEL=${MODEL:-$HOME/models/npu/Qwen3-0.6B}
MODEL_B=${MODEL_B:-$HOME/models/npu/Qwen3-4B}
KERNELS=${KERNELS:-${ONEBIT_NPU_KERNELS:-$HOME/lb/lfm2-npu/kernels_m16}}
PORT_A=${PORT_A:-18761}
PORT_B=${PORT_B:-18762}
TOKS=${TOKS:-8}
REPEATS=${REPEATS:-3}
WAIT_MIN=${WAIT_MIN:-240}
LOCK=/tmp/1bit-npu-device.lock
TAG=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$HOME/lb/612-npu-streams-$TAG
mkdir -p "$OUT" || exit 1
LOG=$OUT/run.log
PROMPT="Write one sentence about the sea."
A=$(basename "$MODEL")
B=$(basename "$MODEL_B")

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
die() { say "FAIL: $*"; exit 1; }

[ -x "$BIN" ] || die "no binary at $BIN"
[ -f "$MODEL/model.q4nx" ] || die "no model.q4nx in $MODEL"
[ -f "$MODEL_B/model.q4nx" ] || die "no model.q4nx in $MODEL_B (set MODEL_B)"
[ -d "$KERNELS" ] || die "no kernel (full ELF) directory at $KERNELS"

# The device lock: wait for it to be free, never take it from a live job.
deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
while ! flock -n "$LOCK" true 2>/dev/null; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        # /proc/locks is the authority on who holds it: one line per holder, its pid in field 5.
        ino=$(stat -Lc %i "$LOCK" 2>/dev/null)
        say "the NPU device lock is still held after ${WAIT_MIN} min (WAIT_MIN); holders:"
        awk -v ino="$ino" '$6 ~ (":" ino "$")' /proc/locks 2>/dev/null | while read -r _ _ adv kind pid _; do
            say "  pid $pid (flock $adv $kind): $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)"
        done
        exit 75
    fi
    say "waiting for the NPU device lock (another job owns the NPU)"
    sleep 30
done
say "device lock free; claiming the box for this run"

say "pin: $("$BIN" --version 2>&1 | head -1) at $BIN"
for m in "$MODEL" "$MODEL_B"; do
    say "model: $m/model.q4nx $(stat -Lc %s "$m/model.q4nx") B sha256 $(sha256sum "$m/model.q4nx" | cut -d' ' -f1)"
done
say "kernels: $KERNELS"
say "protocol: two engines of this runtime on $A and $B, max_tokens $TOKS, temperature 0, $REPEATS repeats per arm, rep 1 dropped as warm-up"

PIDS=()
cleanup() {
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null && wait "$p" 2>/dev/null; done
    PIDS=()
}
trap cleanup EXIT

start() {   # start <model dir> <port> <logfile>: echoes the pid once the serve is ready
    local model=$1 port=$2 log=$3
    : > "$log"
    "$BIN" forward-serve -m "$model/model.q4nx" --kernels "$KERNELS" -p "$port" --npu-streams 2 \
        >> "$log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    for _ in $(seq 1 600); do
        grep -q "ready on" "$log" && { echo "$pid"; return 0; }
        kill -0 "$pid" 2>/dev/null || { say "server on $port died:" >&2; tail -5 "$log" >&2; return 1; }
        sleep 1
    done
    say "server on $port never became ready:" >&2; tail -5 "$log" >&2
    return 1
}

stop() {    # stop <pid>
    [ -n "${1:-}" ] || return 0
    kill "$1" 2>/dev/null; wait "$1" 2>/dev/null; PIDS=()
}

ask() {     # ask <port>: "tokens wall_seconds" for one request
    python3 - "$1" "$PROMPT" "$TOKS" <<'PY'
import json, sys, time, urllib.request
port, prompt, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3])
body = json.dumps({"prompt": prompt, "max_tokens": ntok, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port, data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
except Exception as ex:                        # a failed request is a failed row, not a number
    print("ERR %s" % ex)
    sys.exit(1)
wall = time.time() - t0
# The NPU route's answer carries no usage block, so the token count is the server's own
# finish_reason: "length" means it ran to max_tokens (greedy, temperature 0). Anything else is a
# failed row, not a number to divide by.
ch = (d.get("choices") or [{}])[0]
n = (d.get("usage") or {}).get("completion_tokens") or (ntok if ch.get("finish_reason") == "length" else 0)
if not n:
    print("ERR finish_reason=%r with no usage" % ch.get("finish_reason"))
    sys.exit(1)
print("%d %.3f" % (n, wall))
PY
}

rows=$OUT/rows.csv
echo "rep,arm,model,tokens,wall_s,note" > "$rows"

# Solo: each model alone on the device, the other not even loaded -- so "solo" means solo.
for pair in "solo_a $MODEL $PORT_A" "solo_b $MODEL_B $PORT_B"; do
    set -- $pair
    arm=$1 model=$2 port=$3
    srv=$(start "$model" "$port" "$OUT/$arm.log") || die "$arm server did not start"
    for rep in $(seq 1 "$REPEATS"); do
        out=$(ask "$port") || die "$arm request failed at rep $rep"
        read -r n wall <<< "$out"
        note=$([ "$rep" = 1 ] && echo warm-up)
        echo "$rep,$arm,$(basename "$model"),$n,$wall,$note" >> "$rows"
        say "rep $rep $arm ($(basename "$model")): $n tokens in ${wall}s${note:+ ($note)}"
    done
    say "lock mode on $arm: $(grep -o '[a-z]* device lock acquired[^)]*' "$OUT/$arm.log" | head -1)"
    stop "$srv"
done

# Concurrent: two models, two hwctx, one device.
for rep in $(seq 1 "$REPEATS"); do
    pa=$(start "$MODEL" "$PORT_A" "$OUT/conc-a-$rep.log") || die "engine 1 did not start at rep $rep"
    pb=$(start "$MODEL_B" "$PORT_B" "$OUT/conc-b-$rep.log") || die "engine 2 did not start at rep $rep"
    # Both requests in flight at once: start them together, then read both walls.
    before=$(date +%s.%N)
    ask "$PORT_A" > "$OUT/a-$rep.txt" &
    ra=$!
    ask "$PORT_B" > "$OUT/b-$rep.txt" &
    rb=$!
    wait "$ra" || die "engine 1 request failed at rep $rep"
    wait "$rb" || die "engine 2 request failed at rep $rep"
    after=$(date +%s.%N)
    read -r na wa < "$OUT/a-$rep.txt"
    read -r nb wb < "$OUT/b-$rep.txt"
    note=$([ "$rep" = 1 ] && echo warm-up)
    echo "$rep,concurrent_a,$A,$na,$wa,$note" >> "$rows"
    echo "$rep,concurrent_b,$B,$nb,$wb,$note" >> "$rows"
    say "rep $rep concurrent: $A $na tokens in ${wa}s, $B $nb tokens in ${wb}s, both in $(python3 -c "print(round($after-$before,3))")s${note:+ ($note)}"
    stop "$pa"; stop "$pb"
done

python3 - "$rows" "$TOKS" "$A" "$B" <<'PY' | tee -a "$LOG"
import csv, sys, statistics
TOKS, A, B = int(sys.argv[2]), sys.argv[3], sys.argv[4]
walls = {}
for r in csv.DictReader(open(sys.argv[1])):
    if r["note"] == "warm-up":
        continue
    walls.setdefault(r["arm"], []).append(float(r["wall_s"]))
for arm in ("solo_a", "solo_b", "concurrent_a", "concurrent_b"):
    if not walls.get(arm):
        print("no measured rows for %s" % arm)
        raise SystemExit(1)
sa, sb = statistics.mean(walls["solo_a"]), statistics.mean(walls["solo_b"])
ca, cb = statistics.mean(walls["concurrent_a"]), statistics.mean(walls["concurrent_b"])
n = len(walls["concurrent_a"])
print("")
print("| arm | wall (s), mean of %d measured reps | decode (tok/s) |" % n)
print("|---|---|---|")
print("| %s alone | %.2f | %.2f |" % (A, sa, TOKS / sa))
print("| %s alone | %.2f | %.2f |" % (B, sb, TOKS / sb))
print("| %s + %s at once: %s | %.2f | %.2f (%.1f %% of solo) |" % (A, B, A, ca, TOKS / ca, 100 * sa / ca))
print("| %s + %s at once: %s | %.2f | %.2f (%.1f %% of solo) |" % (A, B, B, cb, TOKS / cb, 100 * sb / cb))
print("| **aggregate** | | **%.2fx** |" % ((TOKS / ca + TOKS / cb) / (TOKS / sa + TOKS / sb)))
PY
say "rows in $rows, logs and answers in $OUT"
