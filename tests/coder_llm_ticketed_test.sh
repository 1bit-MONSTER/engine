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
# tests/coder_llm_ticketed_test.sh — the PR-Agent model runs only inside a gpu-hold ticket, in on-demand windows
# (CODER-LLM-TICKETED, 2026-11-15; engine#2494). No GPU: the real scripts/gpu-hold with a private queue, token, die
# temperature and kernel-log fixture, and a fake server whose /slots says busy or idle from a state file.
#
# Usage: coder_llm_ticketed_test.sh [path-to-coder-llm-ticketed] [path-to-gpu-hold]
#
# Arms:
#   A no demand: no window opens, no token is taken;
#   B demand opens a window under the label coder-llm; an idle server and stale demand close it (rc 0), the server is
#     gone and the token released;
#   C a busy server is cut at the max hold (rc 124), the server is gone;
#   D a gfxhub page fault in the kernel log during the window ends it (rc 78) and the server dies with it;
#   E a lane that queues during a window gets the GPU before the next coder-llm window (FIFO; the restart re-queues);
#   F a failing gh (the pr-agent-run fallback) is logged once per wait with its reason, and opens no window;
#   G the default queue wait is long enough to keep the ticket's place; H a grant whose demand went stale releases
#     without starting the server;
#   I a server that needs 12 s to exit after TERM at the max hold: the window's own kill grace (not gpu-hold's 10 s
#     default) lets it exit cleanly, rc 124; with a grace too short for it, gpu-hold's KILL (rc 137) is still reported
#     as the max hold, rc 124 (the first live window, 11-15 10:56Z, ended rc 137); the teardown time is logged, and a
#     grace too short for the stop wait is warned about;
#   J a server KILLed 1 s into a window that queued 9 s behind a lane stays rc 137: the relabel is timed from the
#     window's start, not the queue's, so a real KILL is not reported as a max hold; every outcome line says how long
#     the window was open;
#   K a fault line already in the kernel log makes gpu-hold refuse the grant (rc 78): the journal says "ticket refused
#     before the window", not a fault during one (arm D says "fault during the window"), and no server starts;
#   L CODER_LLM_LABEL names the window's ticket (another resident model, 1bit-canonical, runs in its own windows;
#     engine#2579);
#   M scripts/ticketed-wait asks a waiting unit for the model and returns 0 once it answers: it touches the demand
#     file, the wrapper queues and opens a window, and /v1/models answers;
#   N ticketed-wait on a model that never answers gives up with 75 after its limit and WITHDRAWS the demand it made,
#     so the unit does not take a ticket for a caller that has gone (review of #2588);
#   O ticketed-wait on a model that already answers returns 0 at once and creates no demand at all.
#   P readiness, not liveness (review of #2588): a server that lists /v1/models (200 from bind) but cannot complete yet
#     (503 while "loading") does not count as answering: ticketed-wait waits, and returns 0 only once a 1-token
#     completion is answered.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)   # the defaults work from any directory, not only the repo root
tick=${1:-$root/scripts/coder-llm-ticketed}
hold=${2:-$root/scripts/gpu-hold}
for f in "$tick" "$hold"; do [ -x "$f" ] || { echo "not executable: $f"; exit 2; }; done
tick=$(readlink -f "$tick"); hold=$(readlink -f "$hold")

scratch=$(mktemp -d)
pids=()
cleanup() { for p in "${pids[@]}"; do kill -9 "$p" 2> /dev/null; done; pkill -9 -f "$scratch/fake" 2> /dev/null; rm -rf "$scratch"; }
trap cleanup EXIT
fail=0
echo 40000 > "$scratch/tctl"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')

cat > "$scratch/fake" <<FAKE
#!/usr/bin/env python3
# fake model server: /slots answers busy or idle from the state file; records its pid
import http.server, json, os
open("$scratch/server.pid", "w").write(str(os.getpid()))
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        busy = open("$scratch/state").read().strip() == "busy"
        body = json.dumps([{"id": 0, "is_processing": busy}]).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        # a completion answers only once "loaded": a $scratch/loading file makes it 503 (weights still loading), while
        # GET /v1/models above keeps answering 200 from bind, as 1bit serve's static listing does
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if os.path.exists("$scratch/loading"):
            self.send_response(503); self.end_headers(); return
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", $port), H).serve_forever()
FAKE
chmod +x "$scratch/fake"

# common environment: private queue/token, cool die, clean kernel log unless an arm writes to it
envs=(GPU_QUEUE_DIR="$scratch/q" GPU_TOKEN_FILE="$scratch/TOKEN" GPU_HOLD_LOG="$scratch/hold.log"
      GPU_HOLD_POLL=1 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=3 GPU_HOLD_RESET_WINDOW_MIN=0
      GPU_HOLD_TCTL_START=100 GPU_HOLD_TCTL_FILE="$scratch/tctl" GPU_HOLD_THERMAL_FLAG="$scratch/x"
      GPU_HOLD_KLOG_FILE="$scratch/klog" PATH="$(dirname "$hold"):$PATH"
      CODER_LLM_GPU_HOLD="$hold" CODER_LLM_DEMAND="$scratch/demand" CODER_LLM_URL="http://127.0.0.1:$port"
      CODER_LLM_GH_POLL=0
      CODER_LLM_POLL_S=1 CODER_LLM_DEMAND_TTL_S=60 CODER_LLM_FAULT_BACKOFF_S=1 CODER_LLM_QUEUE_WAIT_S=60)
mkdir -p "$scratch/q"; : > "$scratch/klog"

wait_for() {   # $1 seconds, $2.. a test command
    local n=$1; shift
    for _ in $(seq 1 "$n"); do "$@" && return 0; sleep 1; done
    return 1
}
server_alive() { [ -s "$scratch/server.pid" ] && kill -0 "$(cat "$scratch/server.pid")" 2> /dev/null; }
token_is() { [ "$(cat "$scratch/TOKEN" 2> /dev/null)" = "$1" ] || grep -qs "^$1" "$scratch/TOKEN"; }

# ---- A + B: no demand, then demand; idle closes the window ----------------------------------------------------------
echo idle > "$scratch/state"; rm -f "$scratch/server.pid" "$scratch/demand"
touch -d '-10 min' "$scratch/demand"   # stale demand
env "${envs[@]}" CODER_LLM_IDLE_S=4 CODER_LLM_MAX_HOLD_MIN=5 timeout 120 "$tick" -- "$scratch/fake" > "$scratch/ab.log" 2>&1 &
ab=$!; pids+=($ab)
sleep 6
if [ -e "$scratch/TOKEN" ] || server_alive; then echo "FAIL A: a window opened with only stale demand"; fail=1; fi
touch "$scratch/demand"
if ! wait_for 15 server_alive; then echo "FAIL B: demand did not open a window"; fail=1; fi
grep -qs coder-llm "$scratch/TOKEN" || { echo "FAIL B: the window does not hold the token as coder-llm ($(cat "$scratch/TOKEN" 2> /dev/null))"; fail=1; }
srv=$(cat "$scratch/server.pid" 2> /dev/null)
wait "$ab"; rc=$?
[ "$rc" = 0 ] || { echo "FAIL B: rc $rc after an idle close, want 0"; fail=1; }
grep -q "idle 4s and no fresh demand" "$scratch/ab.log" || { echo "FAIL B: the log does not say the window closed idle"; fail=1; }
[ -n "$srv" ] && kill -0 "$srv" 2> /dev/null && { echo "FAIL B: the server $srv outlived its window"; fail=1; }
[ -e "$scratch/TOKEN" ] && { echo "FAIL B: the token is still held"; fail=1; }

# ---- C: busy, cut at the max hold ----------------------------------------------------------------------------------
echo busy > "$scratch/state"; rm -f "$scratch/server.pid"; touch "$scratch/demand"
t0=$(date +%s)
env "${envs[@]}" CODER_LLM_IDLE_S=2 CODER_LLM_MAX_HOLD_MIN=0.1 timeout 120 "$tick" -- "$scratch/fake" > "$scratch/c.log" 2>&1
rc=$?; el=$(( $(date +%s) - t0 ))
[ "$rc" = 124 ] || { echo "FAIL C: rc $rc for a busy server at the max hold, want 124"; fail=1; }
[ "$el" -lt 40 ] || { echo "FAIL C: took ${el}s for a 6 s max hold"; fail=1; }
server_alive && { echo "FAIL C: the server outlived the max hold"; fail=1; }
[ -e "$scratch/TOKEN" ] && { echo "FAIL C: the token is still held"; fail=1; }

# ---- D: a kernel fault during the window ---------------------------------------------------------------------------
echo busy > "$scratch/state"; rm -f "$scratch/server.pid"; touch "$scratch/demand"; : > "$scratch/klog"
( sleep 5; echo "kernel: amdgpu 0000:c5:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:9 pasid:26332)" >> "$scratch/klog" ) &
pids+=($!)
env "${envs[@]}" GPU_HOLD_FAULT_ABORT=1 CODER_LLM_IDLE_S=60 CODER_LLM_MAX_HOLD_MIN=5 timeout 120 "$tick" -- "$scratch/fake" > "$scratch/d.log" 2>&1
rc=$?
[ "$rc" = 78 ] || { echo "FAIL D: rc $rc after a gfxhub fault, want 78"; fail=1; }
grep -q "backing off" "$scratch/d.log" || { echo "FAIL D: no back-off after rc 78"; fail=1; }
grep -Eq "rc 78: fault during the window after [0-9]+s" "$scratch/d.log" || { echo "FAIL D: rc 78 not reported as a fault during the window"; fail=1; }
server_alive && { echo "FAIL D: the server survived the fault abort"; fail=1; }
[ -e "$scratch/TOKEN" ] && { echo "FAIL D: the token is still held"; fail=1; }
: > "$scratch/klog"

# ---- E: a lane queued during a window runs before the next window --------------------------------------------------
echo busy > "$scratch/state"; rm -f "$scratch/server.pid"; touch "$scratch/demand"; : > "$scratch/hold.log"
( env "${envs[@]}" CODER_LLM_IDLE_S=60 CODER_LLM_MAX_HOLD_MIN=0.1 timeout 120 "$tick" -- "$scratch/fake" > "$scratch/e1.log" 2>&1
  # what systemd's Restart= does: the next invocation, which queues again
  env "${envs[@]}" CODER_LLM_IDLE_S=60 CODER_LLM_MAX_HOLD_MIN=0.05 timeout 120 "$tick" -- "$scratch/fake" > "$scratch/e2.log" 2>&1 ) &
e=$!; pids+=($e)
wait_for 15 server_alive || { echo "FAIL E: the first window did not open"; fail=1; }
env "${envs[@]}" GPU_HOLD_WAIT=60 timeout 120 "$hold" -t 0.05 -l lanejob -- sleep 1 > /dev/null 2>&1
lrc=$?
[ "$lrc" = 0 ] || { echo "FAIL E: the lane ticket did not run (rc $lrc)"; fail=1; }
wait "$e"
order=$(grep -oE "took .*|queued '[a-z-]+'" "$scratch/hold.log" | tr '\n' ' ')
lane_took=$(grep -n "queued 'lanejob'" "$scratch/hold.log" | head -1 | cut -d: -f1)
took_lines=$(grep -n "took " "$scratch/hold.log" | cut -d: -f1 | tr '\n' ' ')
# three holds, in order: coder-llm, lanejob, coder-llm
holders=$(grep -oE "^gpu-hold\[[0-9]+\] took" "$scratch/hold.log" | wc -l)
seq_labels=$(awk '/queued/ {match($0, /queued .[a-z-]+./); q[$1]=substr($0, RSTART+8, RLENGTH-9)} / took / {print q[$1]}' "$scratch/hold.log" | tr '\n' ' ')
[ "$seq_labels" = "coder-llm lanejob coder-llm " ] || { echo "FAIL E: hold order '$seq_labels', want 'coder-llm lanejob coder-llm '"; fail=1; }

# ---- F: the pr-agent-run fallback says so when it cannot ask (a failing gh), once per wait -------------------------
mkdir -p "$scratch/ghbin"
printf '#!/bin/bash\nif [ "$1" = auth ]; then exit 1; fi\necho "HTTP 401: Bad credentials" >&2; exit 1\n' > "$scratch/ghbin/gh"
chmod +x "$scratch/ghbin/gh"
touch -d '-2 hours' "$scratch/demand"
env "${envs[@]}" CODER_LLM_GH_POLL=1 CODER_LLM_GH_POLL_S=1 PATH="$scratch/ghbin:$PATH" \
    timeout 6 "$tick" -- "$scratch/fake" > "$scratch/f.log" 2>&1
[ "$(grep -c 'gh poll failed (HTTP 401' "$scratch/f.log")" = 1 ] || { echo "FAIL F: want exactly one 'gh poll failed' line, got $(grep -c 'gh poll failed' "$scratch/f.log")"; fail=1; }
grep -q 'gh is not logged in' "$scratch/f.log" || { echo "FAIL F: no warning that gh is not logged in"; fail=1; }
[ -e "$scratch/TOKEN" ] && { echo "FAIL F: a failing poll opened a window"; fail=1; }

# ---- H: a ticket granted after its demand went stale releases without starting the server ---------------------------
echo busy > "$scratch/state"; rm -f "$scratch/server.pid"; : > "$scratch/klog"
env "${envs[@]}" GPU_HOLD_WAIT=60 "$hold" -t 1 -l lanehold -- sleep 7 > /dev/null 2>&1 &
lh=$!; pids+=($lh)
wait_for 10 test -e "$scratch/TOKEN" || { echo "FAIL H: the lane did not take the token"; fail=1; }
touch "$scratch/demand"   # fresh when coder-llm queues, stale (TTL 3 s) by the time the lane releases
env "${envs[@]}" CODER_LLM_DEMAND_TTL_S=3 CODER_LLM_IDLE_S=60 CODER_LLM_MAX_HOLD_MIN=5 timeout 60 "$tick" -- "$scratch/fake" > "$scratch/h.log" 2>&1
rc=$?
wait "$lh"
[ "$rc" = 0 ] || { echo "FAIL H: rc $rc for a stale-demand grant, want 0"; fail=1; }
grep -q "ticket granted with no fresh demand" "$scratch/h.log" || { echo "FAIL H: the stale grant did not say it released"; fail=1; }
[ -s "$scratch/server.pid" ] && { echo "FAIL H: the server was started for a stale grant"; fail=1; }

# ---- I: a slow-exiting server at the max hold ---------------------------------------------------------------------
cat > "$scratch/slow" <<SLOW
#!/bin/bash
# the fake behind a TERM handler that takes 12 s, as the 30B HRX server's teardown did
trap 'trap "" TERM; sleep 12; echo clean > "$scratch/slow.exit"; kill \$p 2> /dev/null; exit 0' TERM
"$scratch/fake" & p=\$!
while :; do sleep 1 & wait \$!; done
SLOW
chmod +x "$scratch/slow"
nograce=(); for e in "${envs[@]}"; do case "$e" in GPU_HOLD_KILL_GRACE=*) ;; *) nograce+=("$e") ;; esac; done
echo busy > "$scratch/state"; rm -f "$scratch/server.pid" "$scratch/slow.exit"; touch "$scratch/demand"
env "${nograce[@]}" CODER_LLM_STOP_WAIT_S=20 CODER_LLM_IDLE_S=60 CODER_LLM_MAX_HOLD_MIN=0.1 timeout 120 "$tick" -- "$scratch/slow" > "$scratch/i1.log" 2>&1
rc=$?
[ "$rc" = 124 ] || { echo "FAIL I: rc $rc for a slow-exiting server at the max hold, want 124"; fail=1; }
[ -e "$scratch/slow.exit" ] || { echo "FAIL I: the slow server was KILLed before it could exit (the window's kill grace did not reach gpu-hold)"; fail=1; }
grep -q "window closed (max hold" "$scratch/i1.log" || { echo "FAIL I: the log does not say max hold"; fail=1; }
grep -Eq "server stopped in 1[1-4]s after TERM" "$scratch/i1.log" || { echo "FAIL I: the teardown time is not logged ($(grep -m1 'server stopped\|did not stop' "$scratch/i1.log"))"; fail=1; }
grep -q "WARNING: the kill grace" "$scratch/i1.log" && { echo "FAIL I: warned about a grace that fits the stop wait"; fail=1; }
[ -e "$scratch/TOKEN" ] && { echo "FAIL I: the token is still held"; fail=1; }
echo busy > "$scratch/state"; rm -f "$scratch/server.pid" "$scratch/slow.exit"; touch "$scratch/demand"
env "${envs[@]}" CODER_LLM_IDLE_S=60 CODER_LLM_MAX_HOLD_MIN=0.1 timeout 120 "$tick" -- "$scratch/slow" > "$scratch/i2.log" 2>&1
rc=$?
[ "$rc" = 124 ] || { echo "FAIL I: rc $rc when gpu-hold KILLs the window after the max hold, want 124"; fail=1; }
grep -q "KILLed after the max hold" "$scratch/i2.log" || { echo "FAIL I: the KILL after the max hold was not reported as such"; fail=1; }
grep -q "WARNING: the kill grace 3s is not at least 5 s longer than CODER_LLM_STOP_WAIT_S=30s" "$scratch/i2.log" || { echo "FAIL I: an explicit grace shorter than the stop wait was not warned about"; fail=1; }
[ -e "$scratch/TOKEN" ] && { echo "FAIL I: the token is still held after the KILL"; fail=1; }
pkill -9 -f "$scratch/slow" 2> /dev/null

# ---- J: a KILL shortly after a long queue wait is a real 137 ----------------------------------------------------------
cat > "$scratch/selfkill" <<SK
#!/bin/bash
# the fake, then a SIGKILL of the server 1 s in (the OOM killer's shape)
"$scratch/fake" & sleep 1; kill -9 \$! 2> /dev/null; kill -9 \$\$
SK
chmod +x "$scratch/selfkill"
echo busy > "$scratch/state"; rm -f "$scratch/server.pid"; : > "$scratch/klog"; touch "$scratch/demand"
env "${envs[@]}" GPU_HOLD_WAIT=60 "$hold" -t 1 -l lanehold -- sleep 9 > /dev/null 2>&1 &
lj=$!; pids+=($lj)
wait_for 10 test -e "$scratch/TOKEN" || { echo "FAIL J: the lane did not take the token"; fail=1; }
env "${envs[@]}" CODER_LLM_IDLE_S=60 CODER_LLM_MAX_HOLD_MIN=0.1 timeout 60 "$tick" -- "$scratch/selfkill" > "$scratch/j.log" 2>&1
rc=$?
wait "$lj"
[ "$rc" = 137 ] || { echo "FAIL J: rc $rc for a server KILLed 1 s into its window after a 9 s queue, want 137 (relabelled as a max hold?)"; fail=1; }
grep -q "KILLed before its max hold" "$scratch/j.log" || { echo "FAIL J: the log does not call it a KILL before the max hold"; fail=1; }
grep -Eq "KILLed before its max hold \(rc 137, open [0-9]s of 0.1 min" "$scratch/j.log" || { echo "FAIL J: the 137 line does not say how long the window was open"; fail=1; }

# ---- K: a grant refused on a recent fault line is not a fault during a window ----------------------------------------
echo busy > "$scratch/state"; rm -f "$scratch/server.pid"; touch "$scratch/demand"
echo "$(date '+%b %d %H:%M:%S') strixhalo kernel: amdgpu 0000:c5:00.0: amdgpu: MODE2 reset" > "$scratch/klog"
nowin=(); for e in "${envs[@]}"; do case "$e" in GPU_HOLD_RESET_WINDOW_MIN=*) ;; *) nowin+=("$e") ;; esac; done
env "${nowin[@]}" GPU_HOLD_RESET_WINDOW_MIN=10 CODER_LLM_IDLE_S=60 CODER_LLM_MAX_HOLD_MIN=1 timeout 60 "$tick" -- "$scratch/fake" > "$scratch/k.log" 2>&1
rc=$?
: > "$scratch/klog"
[ "$rc" = 78 ] || { echo "FAIL K: rc $rc for a refused grant, want 78 ($(grep -m1 'refus' "$scratch/k.log"))"; fail=1; }
grep -q "rc 78: ticket refused before the window" "$scratch/k.log" || { echo "FAIL K: a refused grant is not reported as such ($(grep -m1 'rc 78' "$scratch/k.log"))"; fail=1; }
[ -s "$scratch/server.pid" ] && { echo "FAIL K: a server started on a refused grant"; fail=1; }

# ---- L: the window holds the token under its own label ----------------------------------------------------------
echo idle > "$scratch/state"; rm -f "$scratch/server.pid"; : > "$scratch/klog"; touch "$scratch/demand"
env "${envs[@]}" CODER_LLM_LABEL=1bit-canonical CODER_LLM_IDLE_S=4 CODER_LLM_MAX_HOLD_MIN=1 timeout 60 "$tick" -- "$scratch/fake" > "$scratch/l.log" 2>&1 &
ll=$!; pids+=($ll)
wait_for 15 server_alive || { echo "FAIL L: no window opened"; fail=1; }
grep -qs '1bit-canonical' "$scratch/TOKEN" || { echo "FAIL L: the token is not held as 1bit-canonical ($(head -c 60 "$scratch/TOKEN" 2> /dev/null))"; fail=1; }
grep -qs 'coder-llm' "$scratch/TOKEN" && { echo "FAIL L: the token still says coder-llm"; fail=1; }
wait "$ll"

# ---- M: ticketed-wait asks for the model and returns once it answers ---------------------------------------------
waiter=$root/scripts/ticketed-wait
echo idle > "$scratch/state"; rm -f "$scratch/server.pid" "$scratch/demand"; : > "$scratch/klog"
touch -d '-10 min' "$scratch/demand"
env "${envs[@]}" CODER_LLM_IDLE_S=6 CODER_LLM_MAX_HOLD_MIN=1 timeout 90 "$tick" -- "$scratch/fake" > "$scratch/m.log" 2>&1 &
mm=$!; pids+=($mm)
sleep 3
if [ -e "$scratch/TOKEN" ] || server_alive; then echo "FAIL M: a window opened before anyone asked"; fail=1; fi
t0=$(date +%s)
env TICKETED_WAIT_DEMAND="$scratch/demand" TICKETED_WAIT_POLL_S=1 GPU_TOKEN_FILE="$scratch/TOKEN" \
    "$waiter" test-unit "http://127.0.0.1:$port" 40 2> "$scratch/m.wait"; rc=$?
[ "$rc" = 0 ] || { echo "FAIL M: ticketed-wait rc $rc, want 0 ($(tail -1 "$scratch/m.wait"))"; fail=1; }
[ $(( $(date +%s) - t0 )) -lt 40 ] || { echo "FAIL M: ticketed-wait took the whole limit"; fail=1; }
grep -q 'answers after' "$scratch/m.wait" || { echo "FAIL M: ticketed-wait returned without the model answering"; fail=1; }
grep -qs coder-llm "$scratch/TOKEN" || { echo "FAIL M: the model answered outside a ticket"; fail=1; }
wait "$mm"

# ---- N: a model that never answers: 75 after the limit, demand touched -------------------------------------------
dead=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
rm -f "$scratch/n.demand"; t0=$(date +%s)
env TICKETED_WAIT_DEMAND="$scratch/n.demand" TICKETED_WAIT_POLL_S=1 "$waiter" test-unit "http://127.0.0.1:$dead" 3 2> "$scratch/n.wait"; rc=$?
[ "$rc" = 75 ] || { echo "FAIL N: ticketed-wait rc $rc for a model that never answers, want 75"; fail=1; }
grep -q 'demand withdrawn' "$scratch/n.wait" || { echo "FAIL N: ticketed-wait did not say it withdrew the demand"; fail=1; }
[ -e "$scratch/n.demand" ] && { echo "FAIL N: the demand outlived its caller: the unit would take a ticket for nobody"; fail=1; }

# ---- O: a model that already answers: 0 at once, no demand created --------------------------------------------------
echo idle > "$scratch/state"; rm -f "$scratch/o.demand"
"$scratch/fake" > /dev/null 2>&1 & of=$!; pids+=($of)
wait_for 10 curl -sf -o /dev/null "http://127.0.0.1:$port/v1/models" || { echo "FAIL O: the fake did not come up"; fail=1; }
env TICKETED_WAIT_DEMAND="$scratch/o.demand" TICKETED_WAIT_POLL_S=1 "$waiter" test-unit "http://127.0.0.1:$port" 5 2> "$scratch/o.wait"; rc=$?
[ "$rc" = 0 ] || { echo "FAIL O: ticketed-wait rc $rc on a model that already answers, want 0"; fail=1; }
[ -e "$scratch/o.demand" ] && { echo "FAIL O: ticketed-wait created a demand for a model that already answers"; fail=1; }
kill "$of" 2> /dev/null; wait "$of" 2> /dev/null
[ $(( $(date +%s) - t0 )) -le 10 ] || { echo "FAIL N: ticketed-wait overran its 3 s limit"; fail=1; }

# ---- P: bound but still loading is not "answers" ---------------------------------------------------------------------
echo idle > "$scratch/state"; rm -f "$scratch/p.demand"; touch "$scratch/loading"
"$scratch/fake" > /dev/null 2>&1 & pf=$!; pids+=($pf)
wait_for 10 curl -sf -o /dev/null "http://127.0.0.1:$port/v1/models" || { echo "FAIL P: the fake did not come up"; fail=1; }
env TICKETED_WAIT_DEMAND="$scratch/p.demand" TICKETED_WAIT_POLL_S=1 "$waiter" test-unit "http://127.0.0.1:$port" 3 2> "$scratch/p.wait"; rc=$?
[ "$rc" = 75 ] || { echo "FAIL P: ticketed-wait rc $rc while the model was still loading (only /v1/models answered), want 75"; fail=1; }
( sleep 2; rm -f "$scratch/loading" ) &
env TICKETED_WAIT_DEMAND="$scratch/p.demand" TICKETED_WAIT_POLL_S=1 "$waiter" test-unit "http://127.0.0.1:$port" 20 2> "$scratch/p2.wait"; rc=$?
[ "$rc" = 0 ] && grep -q 'answers after' "$scratch/p2.wait" || { echo "FAIL P: rc $rc once loaded, want 0 after a wait ($(tail -1 "$scratch/p2.wait"))"; fail=1; }
kill "$pf" 2> /dev/null; wait "$pf" 2> /dev/null; rm -f "$scratch/loading"

# ---- G: the default queue wait keeps the ticket's place on a busy box (a give-up re-queues at the back) ------------
sed -n 's/^QUEUE_WAIT=\${CODER_LLM_QUEUE_WAIT_S:-\([0-9]*\)}$/\1/p' "$tick" > "$scratch/qw"
[ "$(cat "$scratch/qw")" -ge 14400 ] 2> /dev/null || { echo "FAIL G: default queue wait $(cat "$scratch/qw")s; under 4 h a busy queue makes it re-queue at the back and starve"; fail=1; }

[ "$fail" = 0 ] && echo "coder_llm_ticketed: ok (no demand -> no ticket; idle -> rc 0; busy -> 124 at max hold; fault -> 78 + back-off; FIFO: coder-llm, lane, coder-llm; a failing gh poll is logged once, no window; default queue wait keeps its place; a stale grant releases without a load; a slow-exiting server gets its grace, a KILL after the max hold is 124, one after a queue wait stays 137; a refused grant is told from a fault during a window; a window takes its label; ticketed-wait asks, waits, gives up with 75 and withdraws its demand, and asks for nothing when the model is up; bound-but-loading is not an answer)"
exit "$fail"
