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
# spec_decode_speedup.sh -- does a DRAFT model pay, in ONE runtime?  Two arms, one server process each, and the answer is
# only ever the RATIO of that one run: two runs of this box differ by 2.5x in absolute tok/s (the same target measured
# 7.7 tok/s and 18.9 tok/s an hour apart), so an absolute number here is not a measurement of anything.
#
#   A: the target alone            (baseline)
#   B: the target + the draft       (the runtime's own speculative decoding: -md / --draft-max or --spec-draft-n-max)
#
# Both arms run the same greedy requests back to back, the first request of each being discarded as warmup -- on a loaded
# box that request can be 10-14x slower than the next one, which is how a first sweep of this measurement came out void.
# Judge it by: the two medians, the ratio, and the runtime's own acceptance accounting from the server log.
#
# THREE THINGS THIS HARNESS LEARNED THE HARD WAY (each cost a gpu-hold ticket):
#   1. the HRX build is not the one on PATH: use ~/wt/i1251/pincur/llama-server with
#      LD_LIBRARY_PATH=/home/bcloud/wt/i1251/pincur:/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib
#      (read off a live process's /proc/<pid>/environ); the c22 package binary and a plain worktree build both report
#      "Available devices: (none)" and reject -dev HRX0;
#   2. `--draft-max` no longer exists on the pinned build ("the argument has been removed. use --spec-draft-n-max"); -md
#      survives, --spec-draft-n-min replaces --draft-min. Hence DRAFT_MAX_FLAG / DRAFT_MIN_FLAG;
#   3. WAITMAX is short (150 s) so a wrong invocation fails inside a ticket instead of eating the whole hold.
#
#   spec_decode_speedup.sh <target.gguf> <draft.gguf> [tokens] [k]
#   BIN=... SRV_ARGS='-dev HRX0 -ngl 99 -fa on -devd HRX0' DRAFT_MAX_FLAG=--spec-draft-n-max spec_decode_speedup.sh ...
set -u
TARGET=${1:?usage: spec_decode_speedup.sh <target.gguf> <draft.gguf> [tokens] [k]}
DRAFT=${2:?usage: spec_decode_speedup.sh <target.gguf> <draft.gguf> [tokens] [k]}
N=${3:-64}
K=${4:-4}
PROMPT=${PROMPT:-Q: Explain in plain words why the sky appears blue during the day. A:}
BIN=${BIN:-/usr/bin/llama-server}
SRV_ARGS=${SRV_ARGS:--ngl 0 -t 4}
DRAFT_MAX_FLAG=${DRAFT_MAX_FLAG:---draft-max}
DRAFT_MIN_FLAG=${DRAFT_MIN_FLAG:---draft-min}
REQS=${REQS:-3}
WAITMAX=${WAITMAX:-150}
PORT_A=${PORT_A:-18120}
PORT_B=${PORT_B:-18121}

waitup() {
    local port=$1 label=$2 t0=$(date +%s)
    while :; do
        [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null)" = 200 ] && return 0
        if [ $(( $(date +%s) - t0 )) -ge "$WAITMAX" ]; then
            echo "   $label: server did not come up in ${WAITMAX}s; last log lines:"
            tail -6 "/tmp/specdec-$label.log" 2>/dev/null | sed 's/^/     /'
            return 1
        fi
        sleep 3
    done
}

ask() {  # ask <port> -> "tok/s accepted drafted"
    curl -s -m 900 -X POST "http://127.0.0.1:$1/completion" -H 'Content-Type: application/json' \
      -d "{\"prompt\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$PROMPT"),\"n_predict\":$N,\
\"temperature\":0,\"top_k\":1,\"top_p\":1,\"repeat_penalty\":1,\"seed\":0,\"cache_prompt\":false,\"stream\":false}" \
    | python3 -c 'import json,sys
d=json.load(sys.stdin); t=d.get("timings",{})
print("%.3f %s %s" % (t.get("predicted_per_second",0.0), t.get("draft_n_accepted","-"), t.get("draft_n","-")))'
}

arm() {  # arm <label> <port> <server args...>
    local label=$1 port=$2; shift 2
    : > "/tmp/specdec-$label.acc"
    # shellcheck disable=SC2086
    $BIN -m "$TARGET" --port "$port" $SRV_ARGS -c 2048 --no-warmup "$@" > "/tmp/specdec-$label.log" 2>&1 &
    local pid=$!
    waitup "$port" "$label" || { kill -9 $pid 2>/dev/null; return 1; }
    local i
    for i in $(seq 1 $(( REQS + 1 ))); do
        read -r tps acc gen < <(ask "$port")
        if [ "$i" = 1 ]; then echo "  $label warmup: ${tps} tok/s (discarded)"; continue; fi
        echo "  $label: ${tps} tok/s   accepted ${acc:-0}/${gen:-0}"
        echo "$tps" >> "/tmp/specdec-$label.acc"
    done
    kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
}

echo "== bin $BIN   srv args: $SRV_ARGS   tokens=$N k=$K"
arm A "$PORT_A" || exit 3
arm B "$PORT_B" -md "$DRAFT" "$DRAFT_MAX_FLAG" "$K" "$DRAFT_MIN_FLAG" 1 || exit 3
echo
python3 - <<'PY'
import os, statistics
def med(lab):
    f = f"/tmp/specdec-{lab}.acc"
    if not os.path.exists(f):
        return None
    v = [float(x) for x in open(f).read().split()]
    return statistics.median(v) if v else None
a, b = med("A"), med("B")
if a is None or b is None:
    print("== ratio n/a: an arm produced no timing (see the log above)")
else:
    print(f"== target alone {a:.3f} tok/s | target+draft {b:.3f} tok/s | ratio {b/a:.3f}x "
          f"({'draft pays' if b > a else 'draft costs'})")
PY
grep -hoE 'draft acceptance rate = [0-9.]+' /tmp/specdec-B.log 2>/dev/null | tail -3 | sed 's/^/== /'
echo "== note: on a CPU-only backend this ratio cannot exceed 1 no matter how good the draft is -- a batched verify costs"
echo "== ~M token-equivalents while a round returns 1 + accepted <= 1 + drafted. See engine#1423."
