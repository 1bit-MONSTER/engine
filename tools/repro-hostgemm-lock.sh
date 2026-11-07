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
# repro-hostgemm-lock.sh -- engine#1353: does a host-GEMM forward-serve hold the exclusive NPU device lock?
#
#   ./repro-hostgemm-lock.sh <1bit binary> [label]
#
# Starts `1bit forward-serve` with NPU_INFER_HOST_GEMM=1 on a private port (never 18097), waits for it to
# answer /health, then reports three things:
#   1. whether the serve process itself has /tmp/1bit-npu-device.lock open (definitive: the lock is held by
#      whoever has the fd, so this is not confused by another lane's serve),
#   2. whether an independent flock(LOCK_EX|LOCK_NB) from this script succeeds while the serve is up,
#   3. whether the serve answers a completion (i.e. it is a working serve, not a process that is about to die).
#
# The array is never opened: host GEMM is the mode that runs the model with host maths and makes no
# xrt::device. Nothing here touches :18097, and the run takes no npu-hold ticket for that reason -- but the
# PRE-FIX binary holds the lock for as long as it runs, so the caller must keep the window short and must not
# run it while a lane is queued for the device.
set -u
BIN=${1:?usage: repro-hostgemm-lock.sh <1bit binary> [label]}
LABEL=${2:-$(basename "$(dirname "$BIN")")}
PORT=${PORT:-18143}
MODEL=${MODEL:-/home/bcloud/models/npu/Qwen3-0.6B}
LOCK=/tmp/1bit-npu-device.lock
LOG=/tmp/hg-lock-$LABEL.log

say() { printf '  %-22s %s\n' "$1" "$2"; }

# Caller's precondition: if someone else holds the lock, this is still a valid *blocked-window* run -- a host-GEMM
# serve that comes up anyway proves it takes no lock, and one that sits in `waiting for the exclusive device lock`
# proves it does. What is refused is the ambiguous case: a lock held by a *host-GEMM* serve that is not ours.
HELD_BY_OTHER=no
if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q "$LOCK"; then
    HELD_BY_OTHER=yes
fi

echo "== $LABEL: $BIN (lock held by another process at start: $HELD_BY_OTHER)"
NPU_INFER_HOST_GEMM=1 setsid "$BIN" forward-serve -m "$MODEL/model.q4nx" --kernels "$MODEL" -p "$PORT" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
up=0
for i in $(seq 1 120); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && { up=1; break; }
    kill -0 $PID 2>/dev/null || break
    sleep 1
done
if [ "$up" != 1 ]; then
    # With the lock held elsewhere, the pre-fix binary never gets past it. Say which of the two this is.
    if grep -q 'waiting for the .* device lock' "$LOG"; then
        say "serve came up" "no -- it is queued on the device lock"
        say "its log says" "$(grep -m1 'waiting for the .* device lock' "$LOG" | cut -c1-90)"
        echo "  verdict: $LABEL takes the device lock in host-GEMM mode (it cannot start while another process holds it)"
        exit 0
    fi
    echo "  serve did not come up (this run says nothing about the lock):"; tail -5 "$LOG" | sed 's/^/    /'
    exit 3
fi
say "serve pid" "$PID (up: $(grep -c 'ready on' "$LOG") ready line[s])"

# 1. the definitive check: does the serve hold the lock fd?
if ls -l /proc/$PID/fd 2>/dev/null | grep -q "$LOCK"; then
    say "serve holds the lock" "YES  <-- the defect (engine#1353)"
    HOLDS=yes
else
    say "serve holds the lock" "no"
    HOLDS=no
fi

# 2. an independent flock while it runs
python3 - "$LOCK" <<'PY'
import fcntl, sys
try:
    f = open(sys.argv[1], 'r')
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(f"  {'flock from outside':22} succeeds (lock is free)")
except OSError as e:
    print(f"  {'flock from outside':22} blocked ({e.strerror})")
PY

# 3. it is a working serve, not a corpse
R=$(curl -s -m 120 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d '{"model":"x","prompt":"The capital of France is","max_tokens":4,"temperature":0}' 2>/dev/null \
    | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print("completion ok:", repr(d["choices"][0]["text"]), d.get("usage",{}).get("completion_tokens"))
except Exception as e: print("completion FAILED:", e)' 2>&1)
say "answers" "$R"
echo "  verdict: $LABEL $([ "$HOLDS" = yes ] && echo 'takes the device lock in host-GEMM mode' || echo 'takes no device lock in host-GEMM mode')"
