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
# gpu_hold_nested_test.sh: a payload that re-invokes gpu-hold on the SAME queue refuses instead of deadlocking
# (engine#2727).
#
# Measured 2026-11-16: a payload re-invoked gpu-hold on its own queue, the inner call queued behind the ticket its
# parent already held, and the outer ticket burned its whole 14-minute bound with the GPU idle -- coder-llm was second
# in a 7-deep queue behind it, so every lane's PR check paid for one script's idiom. The payload already receives
# GPU_HOLD_TICKET/GPU_HOLD_LABEL (engine#1640's "exact, not a string guess"); the queue is exported now as well, which
# is what makes "same queue" distinguishable from npu-hold's documented different-queue case.
#
# Arms:
#   1. a nested invocation on the SAME queue exits 2 and names the holder -- and the payload never runs;
#   2. a nested invocation on a DIFFERENT queue still runs (the npu-hold shape: a gpu-hold inside an NPU ticket must
#      still queue on the GPU), because a guard that refused every nested call would break that;
#   3. a plain unticketed run sees no marker at all (the control: the check must be about being inside a ticket, not
#      about an environment variable someone happens to export);
#   4. GPU_HOLD_ALLOW_NESTED=1 keeps a deliberate nesting possible.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
GH="$ROOT/scripts/gpu-hold"
ok=0
bad=0
chk() {  # <expected rc> <actual rc> <what>
    if [ "$2" = "$1" ]; then printf 'ok   %s\n' "$3"; ok=$((ok + 1))
    else printf 'FAIL %s (expected rc %s, got %s)\n' "$3" "$1" "$2"; bad=$((bad + 1)); fi
}
tmp=$(mktemp -d)
fake_home="$tmp/home"
mkdir -p "$fake_home"
trap 'rm -rf "$tmp"' EXIT
# HERMETIC: every arm runs with HOME pointed at a scratch directory. Without it the arms queue against the FLEET's
# token (`$HOME/GPU-TOKEN`, held by coder-llm on a busy box), so this test would both fail and contend for the real
# GPU -- measured on the first version: three arms sat in the fleet queue and timed out at 3 s.
RUN=( env HOME="$fake_home" ONEBIT_HOLD_DIR="$tmp/hold" GPU_HOLD_TCTL_START=999 GPU_HOLD_TCTL_ABORT=999 )

# A stand-in for the fleet's gpu-hold: the test only needs the marker path, so the child runs the real script with the
# queue in a scratch directory. `-t 0.05` is the shortest documented hold (3 s) and never has to expire here.
run() {  # <env assignments...> -- <args...>
    env "$@" > "$tmp/out" 2> "$tmp/err"
}

# 1. nested on the SAME queue: refuse, name the holder, do not run the payload
out=$("${RUN[@]}" GPU_HOLD_TICKET="gpu-hold outer pid 999 $(date -Is)" GPU_HOLD_LABEL="outer" \
          GPU_HOLD_QUEUE="$fake_home/GPU-QUEUE" \
          "$GH" -l inner -t 0.05 -w 3 -- touch "$tmp/payload-ran" 2>&1)
rc=$?
chk 2 "$rc" "a nested hold on the same queue exits 2"
case $out in
    *"already inside a ticket on this queue (outer:"*) chk 0 0 "... and names the holder label and its ticket line" ;;
    *) chk 0 1 "... and names the holder label and its ticket line   [got: $out]" ;;
esac
[ ! -e "$tmp/payload-ran" ] && chk 0 0 "... and the payload never ran" || chk 0 1 "... and the payload never ran"

# 2. nested on a DIFFERENT queue: must still run (npu-hold's documented case)
"${RUN[@]}" GPU_HOLD_TICKET="gpu-hold outer pid 999 $(date -Is)" GPU_HOLD_LABEL="outer" \
    GPU_HOLD_QUEUE="$tmp/npu-queue" \
    "$GH" -l gpu-inside-npu -t 0.05 -w 3 -- touch "$tmp/other-ran" > "$tmp/o2" 2> "$tmp/e2"
rc=$?
if [ -e "$tmp/other-ran" ]; then chk 0 0 "a nested hold on a DIFFERENT queue still runs (rc $rc)"
else chk 0 1 "a nested hold on a DIFFERENT queue still runs   [rc $rc; stderr: $(tail -2 "$tmp/e2")]"; fi

# 3. control: a plain run has no marker, so a payload that probes for one sees nothing
"${RUN[@]}" GPU_HOLD_MINUTES=0.05 "$GH" -l plain -t 0.05 -w 3 -- \
    bash -c 'printf "%s|%s\n" "${GPU_HOLD_TICKET:-}" "${GPU_HOLD_QUEUE:-}" > "$1"' _ "$tmp/marker" \
    > "$tmp/o3" 2> "$tmp/e3"
if grep -q '^gpu-hold plain pid ' "$tmp/marker" 2>/dev/null && grep -qv '|$' "$tmp/marker" 2>/dev/null; then
    chk 0 0 "a plain run's payload receives the marker and the queue"
else
    chk 0 1 "a plain run's payload receives the marker and the queue   [got: $(cat "$tmp/marker" 2>/dev/null)]"
fi

# 4. the override keeps a deliberate nesting possible
"${RUN[@]}" GPU_HOLD_TICKET="gpu-hold outer pid 999 $(date -Is)" GPU_HOLD_LABEL="outer" \
    GPU_HOLD_QUEUE="$tmp/q4" GPU_HOLD_ALLOW_NESTED=1 GPU_HOLD_MINUTES=0.05 \
    "$GH" -l deliberate -t 0.05 -w 3 -- touch "$tmp/deliberate-ran" > "$tmp/o4" 2> "$tmp/e4"
[ -e "$tmp/deliberate-ran" ] && chk 0 0 "GPU_HOLD_ALLOW_NESTED=1 still permits a deliberate nesting" \
    || chk 0 1 "GPU_HOLD_ALLOW_NESTED=1 still permits a deliberate nesting   [stderr: $(tail -2 "$tmp/e4")]"

printf '\n%s ok, %s failed\n' "$ok" "$bad"
[ "$bad" -eq 0 ]
