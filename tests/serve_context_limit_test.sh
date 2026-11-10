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
# So the assertion is on the SHAPE of the answer, not on its wording: a 4xx passes (the server refused), a 200 with
# ordinary text passes (truncated-but-clean), an empty 200 with finish_reason "length" passes, and a 200 whose content
# is one character repeated (the reported instance was `//////`) FAILS.
#
# engine#1844: `000` (no response at all, i.e. the server died mid-request) and `5xx` are NOT refusals -- they are the
# server failing the very property this row exists for, so they have their own verdicts and FAIL. The first version
# accepted anything that was not 2xx, so a crash passed. The classifier and its pass set are exercised without a
# device by `--self-test`, which is also a ctest row: the bug was never the one wrong line, it was that nothing ever
# ran the classifier on the codes it has to reject.
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
classify_over_limit() {   # <http code> <body file> -> refusal|refusal-empty|died|server-error|degenerate|clean-empty|clean
    python3 - "$1" "$2" <<'PY'
import json, sys, re
code, path = sys.argv[1], sys.argv[2]
try:
    body = open(path, "rb").read().decode("utf-8", "replace")
except OSError:
    body = ""
if code in ("", "000"):
    print("died")                      # curl's 000: no response at all, so the server went away mid-request
    raise SystemExit
if code.startswith("5"):
    print("server-error")              # it answered, but with its own failure -- not the client's
    raise SystemExit
if code.startswith("4"):
    print("refusal" if body.strip() else "refusal-empty")
    raise SystemExit
if not code.startswith("2"):
    print("unclassified-%s" % code)    # e.g. a 3xx redirect: neither outcome, so say so rather than pass
    raise SystemExit
try:
    j = json.loads(body)
    choice = (j.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason")
except Exception:
    text, finish = body.strip(), None
core = text.strip()
packed = re.sub(r"\s+", "", core)
if len(packed) >= 6 and len(set(packed)) == 1:
    # engine#1791's fix, which main's version did not carry: the reported symptom is "//////" (SIX), and the regex
    # above needed eight, so a server answering exactly the reported string classified as 'clean'. One character
    # repeated >= 6 (whitespace stripped, so "////// //////" is caught too) is the rule; six rather than four because
    # a real completion can legitimately be a short repeated token.
    print("degenerate")
elif core == "" and finish in ("length", None):
    print("clean-empty")
elif core == "":
    print("clean-empty")
else:
    print("clean")
PY
}

# The accepted outcomes, in one place, so the self-test asserts the same set the run uses instead of a copied `case`.
verdict_is_pass() {   # <verdict> -> 0 when that verdict is an acceptable over-limit answer
    case "$1" in
        refusal|refusal-empty|clean|clean-empty) return 0 ;;
        *) return 1 ;;
    esac
}

# engine#1844: the classifier's job is to REJECT codes, so the cases that must fail are the ones worth running. Device-
# free by construction (no model, no server): this is a ctest row of its own, and it fails if either class above is put
# back into the "refusal" bucket.
self_test() {
    local tmp fails=0 got
    tmp=$(mktemp -d) || { echo "FAIL self_test: mktemp -d"; return 1; }
    printf '%s' '{"choices":[{"message":{"content":"////////////////"},"finish_reason":"stop"}]}' > "$tmp/degenerate.json"
    # the EXACT reported symptom is six slashes, not sixteen: a detector with an eight-character floor passed this string
    # (engine#1791's first version) and only a six-character case proves the fix covers what was filed (engine#1779).
    printf '%s' '{"choices":[{"message":{"content":"//////"},"finish_reason":"stop"}]}' > "$tmp/degenerate6.json"
    printf '%s' '{"choices":[{"message":{"content":"The answer is 4."},"finish_reason":"stop"}]}' > "$tmp/clean.json"
    printf '%s' '{"choices":[{"message":{"content":""},"finish_reason":"length"}]}' > "$tmp/empty.json"
    printf '%s' '{"error":{"message":"context length exceeded"}}' > "$tmp/err.json"
    : > "$tmp/none.body"

    expect() {   # <label> <code> <body file> <wanted verdict>
        got=$(classify_over_limit "$2" "$3")
        if [ "$got" != "$4" ]; then
            echo "FAIL self_test: $1 -> '$got', wanted '$4'"
            fails=$((fails + 1))
        fi
    }
    expect_pass() {   # <label> <verdict> <should it be accepted: yes|no>
        if verdict_is_pass "$2"; then
            [ "$3" = yes ] || { echo "FAIL self_test: $1 ('$2') is accepted but must not be"; fails=$((fails + 1)); }
        else
            [ "$3" = no ] || { echo "FAIL self_test: $1 ('$2') is rejected but must be accepted"; fails=$((fails + 1)); }
        fi
    }

    # the two the first version got wrong, and the reason this self-test exists
    expect      "no response at all (the server died)" 000  "$tmp/none.body"   died
    expect_pass "no response at all"                    died                   no
    expect      "the server's own 500"                 500  "$tmp/err.json"    server-error
    expect_pass "the server's own 500"                  server-error           no
    # the accepted outcomes must stay accepted, or the fix would just invert the bug
    expect      "a 400 refusal"                        400  "$tmp/err.json"    refusal
    expect_pass "a 400 refusal"                         refusal                yes
    expect      "an ordinary answer"                   200  "$tmp/clean.json"  clean
    expect_pass "an ordinary answer"                    clean                  yes
    expect      "an empty answer with finish_reason length" 200 "$tmp/empty.json" clean-empty
    expect_pass "an empty answer with finish_reason length" clean-empty        yes
    # and the fault itself still has to fail
    expect      "the reported degenerate answer"       200  "$tmp/degenerate.json" degenerate
    expect      "the reported answer at its real length (six)" 200 "$tmp/degenerate6.json" degenerate
    expect_pass "the reported answer at its real length (six)" degenerate               no
    expect_pass "the reported degenerate answer"       degenerate             no
    # neither outcome: say so rather than pass
    expect      "a redirect nobody follows"            302  "$tmp/none.body"   unclassified-302
    expect_pass "a redirect nobody follows"            unclassified-302       no

    rm -rf "$tmp"
    if [ "$fails" -ne 0 ]; then
        echo "self-test: $fails failure(s)"
        return 1
    fi
    echo "self-test: every classification case behaves, including the two that used to pass"
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi
model=${2:-}   # empty in CI unless ONEBIT_SERVE_TEST_GGUF is set: that is a SKIP, not a failure
device=${3:?}
ctx=${4:-512}
[ "$ctx" -ge 64 ] 2>/dev/null || ctx=512
shift 3
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then shift; fi   # the optional ctx argument, when given

[ -x "$bin" ] || { echo "skip: no 1bit at $bin"; exit 77; }
[ -n "$model" ] || { echo "skip: no model given (set ONEBIT_SERVE_TEST_GGUF)"; exit 77; }
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

# engine#1844: one classifier for (http code, body) and one predicate for "is that verdict acceptable", so the
# self-test below exercises exactly the pair the run above uses.

# engine#1844: the fill turn is a PRECONDITION -- if it did not answer 2xx the over-limit assertion below is made
# against a server that had already failed, so it is checked rather than only printed.
case "$code1" in
    2[0-9][0-9]) ;;
    *)
        echo "FAIL serve_context_limit: the fill turn returned HTTP $code1, so the over-limit turn proves nothing"
        tail -15 "$log"
        exit 1 ;;
esac

verdict=$(classify_over_limit "$code2" "$log.over.body")
echo "verdict:            $verdict"

case "$verdict" in
    died)
        echo "FAIL serve_context_limit: no response at all from the over-limit turn (curl 000) -- the server went away, which is the property this row exists for"
        tail -15 "$log"
        exit 1 ;;
    server-error)
        echo "FAIL serve_context_limit: the over-limit turn returned HTTP $code2 -- the server failed on its own, not a refusal"
        tail -15 "$log"
        exit 1 ;;
esac

if verdict_is_pass "$verdict"; then
    echo "PASS serve_context_limit: an over-limit turn is $verdict, not a degenerate repeat"
    exit 0
fi

case "$verdict" in
    degenerate)
        echo "FAIL serve_context_limit: the over-limit turn degenerated to a repeated character: $(head -c 80 "$log.over.body")"
        tail -15 "$log"
        exit 1 ;;
    *)
        echo "FAIL serve_context_limit: could not classify the over-limit answer ($verdict)"
        tail -15 "$log"
        exit 1 ;;
esac
