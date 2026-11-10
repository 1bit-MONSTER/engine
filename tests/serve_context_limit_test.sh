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
# Copyright 2026 The 1bit-MONSTER Authors.
# SPDX-License-Identifier: Apache-2.0
#
# serve_context_limit_test -- engine#1779 fault 2: "Qwen3.5 answering `//////` after hitting the context limit".
#
# The fault as FastFlowLM v1.0.12 states it is a degenerate answer once the context is full. What must happen instead,
# per ~/briefs/TNPU-ALL.md §9: a request whose prompt fills the context, then one more turn, and the response must be
# **a refusal or a truncated-but-clean answer, never a degenerate repeat of one token**.
#
# So the assertion is on the SHAPE of the answer, not on its wording: a 4xx/5xx with an error body passes (that is a
# refusal), a 200 with ordinary text passes (truncated-but-clean), an empty 200 with finish_reason "length" passes, and
# a 200 whose content is one character repeated — `//////` being the reported instance — FAILS.
#
# Device-free by construction: `--device cpu`, per the issue's rule for these four.
#
# usage: tests/serve_context_limit_test.sh path/to/1bit <model> <device> [ctx] [extra serve args]
#   exits 0 when the over-limit turn is a refusal or a clean answer, 1 on a degenerate answer, 77 when it cannot run.
set -uo pipefail

bin=${1:?usage: serve_context_limit_test.sh path/to/1bit <model> <device> [ctx] [args]}
here=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=tests/serve_port.sh
. "$(dirname "$0")/serve_port.sh"
model=${2:?}
device=${3:?}
ctx=${4:-512}
[ "$ctx" -ge 64 ] 2>/dev/null || ctx=512
shift 3
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then shift; fi   # the optional ctx argument, when given

[ -x "$bin" ] || { echo "skip: no 1bit at $bin"; exit 77; }
[ -f "$model" ] || { echo "skip: no model at $model"; exit 77; }
command -v curl > /dev/null || { echo "skip: no curl"; exit 77; }
command -v python3 > /dev/null || { echo "skip: no python3"; exit 77; }

if [ "$device" = hrx ]; then
    have=$("$here/scripts/gpu-targets.sh" 2>/dev/null | tr '\n' ' ')
    want=${ONEBIT_HRX_E2E_TARGETS:-gfx1151}
    found=0
    for t in $want; do case " $have " in *" $t "*) found=1 ;; esac; done
    [ $found = 1 ] || { echo "skip: no $want GPU for HRX on this machine (GPUs: ${have:-none})"; exit 77; }
fi

log=$(mktemp)
srv=
cleanup() {
    if [ -n "$srv" ] && kill -0 "$srv" 2> /dev/null; then
        kill "$srv" 2> /dev/null
        for _ in $(seq 1 50); do kill -0 "$srv" 2> /dev/null || break; sleep 0.1; done
        kill -9 "$srv" 2> /dev/null
    fi
    rm -f "$log"
}
trap cleanup EXIT INT TERM

"$bin" serve -m "$model" --device "$device" --ctx-size "$ctx" --host 127.0.0.1 --port 0 "$@" > "$log" 2>&1 &
srv=$!
if ! port=$(serve_port_wait_any "$log" "$srv" 24000); then
    echo "skip: serve did not report a port in 240 s (see below)"; tail -20 "$log"; exit 77
fi
echo "serve on 127.0.0.1:$port, device=$device, ctx=$ctx"

# A prompt comfortably longer than the context. Repetitive on purpose: the fault is about what the SAMPLER does when
# there is no room left, so the content of the prompt must not be what the answer degenerates from.
# Sized to FIT: the issue's shape is a prompt that fills the context and is SERVED, then one more turn that cannot be.
filler=$(python3 - "$ctx" <<'PY'
import sys
words = ["the", "river", "runs", "past", "the", "mill", "and", "under", "the", "old", "stone", "bridge"]
n = max(32, int(int(sys.argv[1]) * 0.72))      # ~0.72 tokens per word leaves room for the template and the answer
print(" ".join(words[(i * 7) % len(words)] for i in range(n)))
PY
)
req() {  # <json-file> -> "<http-code>\t<response-body>"
    curl -s -m 300 -o "$2.body" -w '%{http_code}' -H 'Content-Type: application/json' -d @"$1" \
        "http://127.0.0.1:$port/v1/chat/completions" 2> /dev/null
    printf '%s\t%s' "$([ -s "$2.body" ] && echo ok)" "$2.body"
}

python3 - "$filler" > "$log.first.json" <<'PY'
import json, sys
print(json.dumps({"model": "x", "messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": 16, "temperature": 0}))
PY
code1=$(curl -s -m 300 -o "$log.first.body" -w '%{http_code}' -H 'Content-Type: application/json' \
    -d @"$log.first.json" "http://127.0.0.1:$port/v1/chat/completions" 2> /dev/null)
echo "fill turn:          HTTP $code1 $(head -c 120 "$log.first.body" 2> /dev/null | tr '\n' ' ')"

# The over-limit turn: same conversation, plus more text, so the request cannot fit what is left.
python3 - "$filler" > "$log.over.json" <<'PY'
import json, sys
print(json.dumps({"model": "x", "messages": [
    {"role": "user", "content": sys.argv[1]},
    {"role": "assistant", "content": "ok"},
    {"role": "user", "content": sys.argv[1]}],
    "max_tokens": 16, "temperature": 0}))
PY
code2=$(curl -s -m 300 -o "$log.over.body" -w '%{http_code}' -H 'Content-Type: application/json' \
    -d @"$log.over.json" "http://127.0.0.1:$port/v1/chat/completions" 2> /dev/null)
echo "over-limit turn:    HTTP $code2 $(head -c 160 "$log.over.body" 2> /dev/null | tr '\n' ' ')"

# Classify the over-limit answer: refusal (non-2xx with a body), clean text, empty-with-length, or degenerate.
verdict=$(python3 - "$code2" "$log.over.body" <<'PY'
import json, sys, re
code, path = sys.argv[1], sys.argv[2]
try:
    body = open(path, "rb").read().decode("utf-8", "replace")
except OSError:
    body = ""
if not code.startswith("2"):
    print("refusal" if body.strip() else "refusal-empty")
    raise SystemExit
try:
    j = json.loads(body)
    choice = (j.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason")
except Exception:
    text, finish = body.strip(), None
core = text.strip()
if re.fullmatch(r"(.)\1{7,}", core, re.S):
    print("degenerate")
elif core == "" and finish in ("length", None):
    print("clean-empty")
elif core == "":
    print("clean-empty")
else:
    print("clean")
PY
)
echo "verdict:            $verdict"

case "$verdict" in
    refusal|refusal-empty|clean|clean-empty)
        echo "PASS serve_context_limit: an over-limit turn is $verdict, not a degenerate repeat"
        exit 0 ;;
    degenerate)
        echo "FAIL serve_context_limit: the over-limit turn degenerated to a repeated character: $(head -c 80 "$log.over.body")"
        tail -15 "$log"
        exit 1 ;;
    *)
        echo "FAIL serve_context_limit: could not classify the over-limit answer ($verdict)"
        tail -15 "$log"
        exit 1 ;;
esac
