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

# engine#612, row 1: two NPU streams through the supported opt-in (`--npu-streams 2`,
# app/npu_device_lock.h) against the same model run solo, REPEATS reps of 8 greedy tokens each.
# Every number carries its instrument, its repeat count and its timestamp; rep 1 is a warm-up and
# is reported but not averaged, because the first request after a load is 2x the steady state.
#
#   BIN=... MODEL=... KERNELS=... tools/showcase-612-npu-streams.sh
#
# It waits for the NPU device lock to be free and never takes it from a running job: while another
# process holds /tmp/1bit-npu-device.lock exclusively (the owner's own serve, the ELF runner) this
# exits 75 and says who holds it. It kills only the two servers it started, by pid.
set -uo pipefail

BIN=${BIN:-${PWD}/build/1bit}
MODEL=${MODEL:-$HOME/models/npu/Qwen3-0.6B}
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

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
die() { say "FAIL: $*"; exit 1; }

[ -x "$BIN" ] || die "no binary at $BIN"
[ -f "$MODEL/model.q4nx" ] || die "no model.q4nx in $MODEL"
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
say "model: $MODEL/model.q4nx $(stat -Lc %s "$MODEL/model.q4nx") B sha256 $(sha256sum "$MODEL/model.q4nx" | cut -d' ' -f1)"
say "kernels: $KERNELS"
say "protocol: max_tokens $TOKS, temperature 0, $REPEATS repeats, rep 1 dropped as warm-up"

PIDS=()
cleanup() {
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null && wait "$p" 2>/dev/null; done
    PIDS=()
}
trap cleanup EXIT

start() {   # start <port> <logfile>
    local port=$1 log=$2
    : > "$log"
    "$BIN" forward-serve -m "$MODEL/model.q4nx" --kernels "$KERNELS" -p "$port" --npu-streams 2 \
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
echo "rep,arm,stream,tokens,wall_s,note" > "$rows"

solo_srv=$(start "$PORT_A" "$OUT/solo.log") || die "solo server did not start"
for rep in $(seq 1 "$REPEATS"); do
    out=$(ask "$PORT_A") || die "solo request failed at rep $rep"
    read -r n wall <<< "$out"
    note=$([ "$rep" = 1 ] && echo warm-up)
    echo "$rep,solo,1,$n,$wall,$note" >> "$rows"
    say "rep $rep solo: $n tokens in ${wall}s${note:+ ($note)}"
done
kill "$solo_srv" 2>/dev/null; wait "$solo_srv" 2>/dev/null
PIDS=()
say "shared lock taken by both streams: $(grep -c 'shared device lock acquired' "$OUT/solo.log") of 1"

for rep in $(seq 1 "$REPEATS"); do
    a=$(start "$PORT_A" "$OUT/solo-$rep.log") || die "stream 1 did not start at rep $rep"
    b=$(start "$PORT_B" "$OUT/stream2-$rep.log") || die "stream 2 did not start at rep $rep"
    # Both requests in flight at once: start them together, then read both walls.
    before=$(date +%s.%N)
    ask "$PORT_A" > "$OUT/a-$rep.txt" &
    pa=$!
    ask "$PORT_B" > "$OUT/b-$rep.txt" &
    pb=$!
    wait "$pa" || die "stream 1 request failed at rep $rep"
    wait "$pb" || die "stream 2 request failed at rep $rep"
    after=$(date +%s.%N)
    read -r na wa < "$OUT/a-$rep.txt"
    read -r nb wb < "$OUT/b-$rep.txt"
    note=$([ "$rep" = 1 ] && echo warm-up)
    echo "$rep,concurrent,1,$na,$wa,$note" >> "$rows"
    echo "$rep,concurrent,2,$nb,$wb,$note" >> "$rows"
    say "rep $rep concurrent: stream 1 $na tokens in ${wa}s, stream 2 $nb tokens in ${wb}s, both in $(python3 -c "print(round($after-$before,3))")s${note:+ ($note)}"
    kill "$a" "$b" 2>/dev/null; wait "$a" "$b" 2>/dev/null
    PIDS=()
done

python3 - "$rows" "$TOKS" <<'PY' | tee -a "$LOG"
import csv, sys, statistics
TOKS = int(sys.argv[2])
solo, conc = {}, {}
for r in csv.DictReader(open(sys.argv[1])):
    if r["note"] == "warm-up":
        continue
    (solo if r["arm"] == "solo" else conc).setdefault(r["arm"] + r["stream"], []).append(float(r["wall_s"]))
if not solo or not conc:
    print("no measured rows (every rep was a warm-up)")
    raise SystemExit(1)
s = statistics.mean(next(iter(solo.values())))
c1 = statistics.mean(conc.get("concurrent1", [0]))
c2 = statistics.mean(conc.get("concurrent2", [0]))
c = statistics.mean([c1, c2])
print("")
print("| arm | wall (s), mean over %d measured reps | per-stream decode (tok/s) |" % len(next(iter(conc.values()))))
print("|---|---|---|")
print("| one stream, solo | %.2f | %.2f |" % (s, TOKS / s))
print("| two streams, stream 1 | %.2f | %.2f (%.1f %% of solo) |" % (c1, TOKS / c1, 100 * s / c1))
print("| two streams, stream 2 | %.2f | %.2f (%.1f %% of solo) |" % (c2, TOKS / c2, 100 * s / c2))
print("| **aggregate** | | **%.2fx** |" % ((TOKS / c1 + TOKS / c2) / (TOKS / s)))
PY
say "rows in $rows, logs and answers in $OUT"
