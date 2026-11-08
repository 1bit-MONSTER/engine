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
# spec_decode_acceptance_hrx.sh -- acceptance measured ON THE DEVICE, for a runtime whose speculative path does not report it.
# The HRX llama.cpp build echoes neither `draft_n_accepted` nor an acceptance-rate line, so `--spec-draft-n-max`-style
# speculative runs cannot report acceptance there. It does not have to: acceptance is a property of the two models, and this
# driver measures it with two ordinary servers on HRX0 and tools/spec_decode_acceptance.py, which compares the draft's greedy
# proposals with the TARGET's own greedy continuation, token by token -- the same accept-prefix rule a greedy verifier applies.
#
# Run it inside ONE gpu-hold ticket. Both servers are started here and killed by a trap on every exit path
# (EXIT/INT/TERM/HUP): an orphaned server held 51 GiB of GTT on 2026-11-08 and knocked the PR-agent LLM off the iGPU, so nothing
# started here outlives the ticket.
#
# Both servers are started here and killed by a trap on every exit path (EXIT/INT/TERM/HUP): an orphaned server held 51 GiB of
# GTT on 2026-11-08 and knocked the PR-agent LLM off the iGPU, so nothing started here outlives the ticket.
#
#   gpu-hold -t 9 -l "..." -- tools/spec_decode_acceptance_hrx.sh <target.gguf> <draft.gguf> [tokens] [k]
#
# The prompt is PROMPT below (the same text tools/spec_decode_speedup.sh uses, so the ratio and the acceptance describe one
# setup). Report it with the sample size: a 24-token budget truncates the last round, so quote rounds and the untruncated
# tokens/round = (accepted + rounds) / rounds, not emitted / rounds.
set -u
TARGET=${1:?usage: spec_decode_acceptance_hrx.sh <target.gguf> <draft.gguf> [tokens] [k]}
DRAFT=${2:?usage: spec_decode_acceptance_hrx.sh <target.gguf> <draft.gguf> [tokens] [k]}
N=${3:-24}
K=${4:-4}
BIN=${BIN:-/home/bcloud/wt/i1251/pincur/llama-server}
LIB=${LIB:-/home/bcloud/wt/i1251/pincur:/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib}
HARNESS=${HARNESS:-$(dirname "$(readlink -f "$0")")/spec_decode_acceptance.py}
PROMPT=${PROMPT:-Q: Explain in plain words why the sky appears blue during the day. A:}
PORT_T=${PORT_T:-18130}
PORT_D=${PORT_D:-18131}
WAITMAX=${WAITMAX:-240}
PIDS=""

cleanup() { for p in $PIDS; do kill -9 "$p" 2>/dev/null; done; PIDS=""; }
trap 'cleanup; exit 130' INT TERM HUP
trap cleanup EXIT

export LD_LIBRARY_PATH="$LIB"
[ -f "$HARNESS" ] || { echo "harness not found: $HARNESS"; exit 2; }

start() {   # start <port> <model> <log>
    $BIN -m "$2" --port "$1" -dev HRX0 -ngl 99 -fa on -c 2048 --no-warmup > "$3" 2>&1 &
    PIDS="$PIDS $!"
}
waitup() {
    local port=$1 t0=$(date +%s)
    while :; do
        [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null)" = 200 ] && return 0
        [ $(( $(date +%s) - t0 )) -ge "$WAITMAX" ] && { echo "  :$port did not come up in ${WAITMAX}s"; \
            tail -4 "/tmp/hrxacc-$port.log" 2>/dev/null | sed 's/^/    /'; return 1; }
        sleep 3
    done
}

echo "== target :$PORT_T   draft :$PORT_D   tokens=$N k=$K   bin=$BIN"
start "$PORT_T" "$TARGET" "/tmp/hrxacc-$PORT_T.log"
start "$PORT_D" "$DRAFT"  "/tmp/hrxacc-$PORT_D.log"
waitup "$PORT_T" || exit 3
waitup "$PORT_D" || exit 3
printf '%s\n' "$PROMPT" > /tmp/hrxacc-prompt.txt

echo "== acceptance (token-exact, target's own greedy continuation as the verifier)"
echo "== report rounds and the UNTRUNCATED tokens/round = (accepted + rounds) / rounds -- a --tokens budget cuts the"
echo "== last round, so emitted/rounds understates it (3.048 vs 3.095 at 64 tokens in 21 rounds)"
python3 "$HARNESS" --target "http://127.0.0.1:$PORT_T" --draft "http://127.0.0.1:$PORT_D" \
    --prompt-file /tmp/hrxacc-prompt.txt --tokens "$N" --k "$K"
rc=$?
echo "== harness rc=$rc; stopping both servers"
exit $rc
