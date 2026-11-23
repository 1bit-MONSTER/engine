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
# box_inkling.sh OUT -- ONE Inkling-2-Mini measurement on the Loom route, run ON the box (strixhalo) inside
# `box-hold gpu sudo -n timeout ...` by custom_radeon8060s.sh. Never started by hand outside box-hold.
#
# It serves the Loom card (loom-serve, hrx-loom 0.3.0) with the lever settings in its environment, runs the gauntlet
# (gauntlet.py) against it, stops the server and leaves everything in OUT for the caller to fetch:
#   OUT/gauntlet.json    the gauntlet rows (full replies included)
#   OUT/provenance.json  card revision, file hashes, the exact command, power mode, die temperature, RAM low-water
#   OUT/serve.log        loom-serve's own log
#   OUT/status           "ok" or the reason the measurement did not run
#
# Levers (validated by the caller; this script re-checks the shape and refuses anything else):
#   HL_DRAFT_DEPTH    preset draft_depth (DSpark; the card's default is 3)
#   HL_ROWS           preset rows (concurrent sequences)          HL_CONTEXT   preset context_capacity
#   HL_PREFILL_CAP    preset prefill_capacity                     HL_MAX_TOKENS preset max_tokens
#   HL_SPIN_NS        amdgpu completion_spin_ns ("default" = the device's own)
#   HL_WAIT_MODE      --wait_mode (active|blocking, "" = default) HL_POOL_BACKING --pool_backing (fixed|elastic)
#   HL_WORKERS        --workers
# Not levers: HL_RUNS (repetitions, default 3), HL_GAUNTLET_ROWS, HL_RAM_NEED_GB (the job's declared RAM need, default
# 40 GB: 31 GB measured at low-water for the card's 4 x 262144, fixed backing).
set -u
OUT=${1:?usage: box_inkling.sh OUT}
HERE=$(cd "$(dirname "$0")" && pwd)
B=${HL_BOX_ROOT:-/data/dev/cl-hyperloom}
CARD=${HL_CARD:-$B/cards/inkling2-mini-a8354e77}
LS=${HL_LOOM_SERVE:-$B/hrxloom030/hrx_loom-0.3.0.data/scripts/loom-serve}
OWNER=${HL_OUT_OWNER:-bcloud}
mkdir -p "$OUT" || exit 2
status() { echo "$*" > "$OUT/status"; echo "[box_inkling] $*"; }
pid=; guard=
cleanup() {
    if [ -n "$pid" ]; then kill -TERM -"$pid" 2>/dev/null; sleep 3; kill -KILL -"$pid" 2>/dev/null; wait "$pid" 2>/dev/null; fi
    pid=
    [ -n "$guard" ] && kill "$guard" 2>/dev/null
    guard=
    rm -f "$OUT/serve.pid"
    chown -R "$OWNER:$OWNER" "$OUT" 2>/dev/null
}
trap 'status "killed (signal)"; cleanup; exit 143' TERM INT
trap cleanup EXIT

num() { case "$2" in ''|*[!0-9]*) status "refused: $1=$2 is not a whole number"; exit 2 ;; esac; }
DEPTH=${HL_DRAFT_DEPTH:-3};   num HL_DRAFT_DEPTH "$DEPTH"
ROWS=${HL_ROWS:-4};           num HL_ROWS "$ROWS"
CTX=${HL_CONTEXT:-262144};    num HL_CONTEXT "$CTX"
PCAP=${HL_PREFILL_CAP:-4096}; num HL_PREFILL_CAP "$PCAP"
MAXT=${HL_MAX_TOKENS:-4096};  num HL_MAX_TOKENS "$MAXT"
WORKERS=${HL_WORKERS:-8};     num HL_WORKERS "$WORKERS"
SPIN=${HL_SPIN_NS:-4294967295}; [ "$SPIN" = default ] || num HL_SPIN_NS "$SPIN"
WAIT=${HL_WAIT_MODE:-};       case "$WAIT" in ''|active|blocking) ;; *) status "refused: HL_WAIT_MODE=$WAIT"; exit 2 ;; esac
POOL=${HL_POOL_BACKING:-fixed}; case "$POOL" in fixed|elastic) ;; *) status "refused: HL_POOL_BACKING=$POOL"; exit 2 ;; esac
RUNS=${HL_RUNS:-3};           num HL_RUNS "$RUNS"
GROWS=${HL_GAUNTLET_ROWS:-k391,code,reasoning,sustained,needle16k}
case "$GROWS" in *[!a-z0-9,]*) status "refused: HL_GAUNTLET_ROWS=$GROWS"; exit 2 ;; esac
# The KV pool is rows x context; the card's 4 x 262144 is the measured point (cl-loomfun 22 Nov: 31 GB used at
# low-water, fixed backing). Anything above twice that is refused rather than measured into the RAM guard.
if [ $((ROWS * CTX)) -gt $((2 * 4 * 262144)) ]; then status "refused: rows*context $((ROWS * CTX)) > 2x the card's 4x262144"; exit 2; fi
[ -x "$LS" ] || { status "refused: no loom-serve at $LS"; exit 2; }
[ -f "$CARD/model.json" ] || { status "refused: no card at $CARD"; exit 2; }

# --- RAM: the job states its need; refuse below need + 16 GB, kill the job if MemAvailable falls under 16 GB --------
NEED=${HL_RAM_NEED_GB:-40}; num HL_RAM_NEED_GB "$NEED"
FLOOR=${HL_RAM_FLOOR_GB:-16}; num HL_RAM_FLOOR_GB "$FLOOR"   # the fleet's 16 GB; only the device-free test lowers it
mem_avail_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
have=$(mem_avail_gb)
echo "ram: need ${NEED} GB (+${FLOOR} floor), MemAvailable ${have} GB at start" > "$OUT/ram.txt"
if [ "$have" -lt $((NEED + FLOOR)) ]; then status "refused: RAM need $NEED + $FLOOR GB, MemAvailable $have GB"; exit 3; fi
me=$$
( low=999; while kill -0 "$me" 2>/dev/null; do a=$(mem_avail_gb); [ "$a" -lt "$low" ] && { low=$a; echo "$low" > "$OUT/ram.low"; }
    if [ "$a" -lt "$FLOOR" ]; then echo "RAM-GUARD: MemAvailable $a GB < $FLOOR at $(date -u +%T)" >> "$OUT/ram.txt"; kill -TERM "$me"; exit; fi
    sleep 2; done ) &
guard=$!

ulimit -l unlimited 2>/dev/null

# --- the card, with the levers applied: an overlay of links to the pinned card plus a rewritten model.json ---------
OV=$OUT/card; mkdir -p "$OV"
for f in "$CARD"/* "$CARD"/.[!.]*; do
    [ -e "$f" ] || continue
    [ "$(basename "$f")" = model.json ] && continue
    ln -sfn "$f" "$OV/$(basename "$f")"
done
python3 - "$CARD/model.json" "$OV/model.json" "$DEPTH" "$ROWS" "$CTX" "$PCAP" "$MAXT" <<'EOF' || { status "refused: cannot write the overlay model.json"; exit 2; }
import json, sys
src, dst, depth, rows, ctx, pcap, maxt = sys.argv[1:]
doc = json.load(open(src))
ex = doc["presets"]["chat"]["execution"]
ex.update(draft_depth=int(depth), rows=int(rows), context_capacity=int(ctx), prefill_capacity=int(pcap), max_tokens=int(maxt))
json.dump(doc, open(dst, "w"), indent=1)
EOF

DEV='amdgpu://'; [ "$SPIN" = default ] || DEV="amdgpu://?completion_spin_ns=$SPIN"
# A port nothing is bound to (another lane's CPU-side server, or a parallel test, may hold one in the range).
PORT=$(python3 -c '
import socket, sys
for p in range(18400, 18490):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); print(p); sys.exit(0)
    except OSError:
        s.close()
sys.exit(1)') || { status "refused: no free port in 18400-18489"; exit 3; }
CMD=("$LS" "--model=$OV/model.json" --preset=chat "--device=$DEV" "--pool_backing=$POOL" "--workers=$WORKERS" "--listen=127.0.0.1:$PORT")
[ -n "$WAIT" ] && CMD+=("--wait_mode=$WAIT")

hw() { for h in /sys/class/hwmon/hwmon*; do [ "$(cat "$h/name" 2>/dev/null)" = "$1" ] && { echo "$h"; return; }; done; }
K=$(hw k10temp)
tctl() { [ -n "$K" ] && [ -r "$K/temp1_input" ] && echo $(( $(cat "$K/temp1_input") / 1000 )) || echo null; }
sha_cached() {  # sha256 of a big file, cached by device+inode+size+mtime so 17 GB is hashed once, not per measurement
    local f; f=$(readlink -f "$1"); local key; key=$(stat -Lc '%d-%i-%s-%Y' "$f")
    local c=$B/state/sha-cache; mkdir -p "$c"
    [ -f "$c/$key" ] || sha256sum "$f" | cut -c1-64 > "$c/$key"
    cat "$c/$key"
}
T0=$(tctl); L0=$(cut -d' ' -f1-3 /proc/loadavg)
MODE=$(tail -1 /data/dev/orchestrator/perfmode.log 2>/dev/null | tr -d '"\\')

cd "$OV" || exit 2
setsid "${CMD[@]}" > "$OUT/serve.log" 2>&1 &
pid=$!; echo "$pid" > "$OUT/serve.pid"
ok=0; i=0
while [ $i -lt 600 ]; do
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"' && { ok=1; break; }
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/status" | grep -q '"registration":"failed"' && break
    kill -0 "$pid" 2>/dev/null || break
    i=$((i + 1)); sleep 1
done
echo "$i" > "$OUT/register_s"
if [ $ok != 1 ]; then
    kill -TERM -- -"$pid" 2>/dev/null   # its process group: setsid made the server a session leader
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/status" 2>/dev/null | head -c 2000 > "$OUT/status.json"
    status "failed: the model did not register in ${i}s (serve.log)"; exit 1
fi
python3 -u "$HERE/gauntlet.py" --url "http://127.0.0.1:$PORT" --arm hyperloom --runs "$RUNS" --rows "$GROWS" \
    --json "$OUT/gauntlet.json" > "$OUT/gauntlet.log" 2>&1
grc=$?
cleanup
trap - EXIT
T1=$(tctl)
FAULTS=$(journalctl -k --since -60min 2>/dev/null | grep -c -E 'amdgpu.*(fault|reset)|THERMTRIP')
python3 - "$OUT/provenance.json" <<EOF
import json, sys
json.dump({
    "card": "$CARD", "card_revision": open("$CARD/CARD_REVISION").read().strip(),
    "loom_serve": "$LS", "loom_serve_sha256": "$(sha256sum "$LS" | cut -c1-64)",
    "gguf_sha256": {
$(for f in "$CARD"/parameter/*.gguf; do echo "        \"$(basename "$f")\": \"$(sha_cached "$f")\","; done)
    },
    "command": $(python3 -c 'import json,sys;print(json.dumps(sys.argv[1:]))' "${CMD[@]}"),
    "levers": {"draft_depth": $DEPTH, "rows": $ROWS, "context_capacity": $CTX, "prefill_capacity": $PCAP,
               "max_tokens": $MAXT, "completion_spin_ns": "$SPIN", "wait_mode": "$WAIT", "pool_backing": "$POOL",
               "workers": $WORKERS},
    "runs": $RUNS, "gauntlet_rows": "$GROWS", "register_s": $i,
    "kernel": "$(uname -r)", "power_mode_log_last": "$MODE",
    "tctl_c_start": $T0, "tctl_c_end": $T1, "loadavg_start": "$L0",
    "mem_available_gib_start": $have, "mem_available_gib_low": $(cat "$OUT/ram.low" 2>/dev/null || echo null),
    "ram_need_gb": $NEED, "amdgpu_fault_or_reset_lines_60min": $FAULTS,
}, open(sys.argv[1], "w"), indent=1)
EOF
chown -R "$OWNER:$OWNER" "$OUT" 2>/dev/null
if [ $grc != 0 ]; then status "failed: gauntlet rc=$grc (gauntlet.log)"; exit 1; fi
status ok
