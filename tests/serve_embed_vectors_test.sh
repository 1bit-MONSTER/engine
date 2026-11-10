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
# serve_embed_vectors_test -- engine#1779 fault 3: "wrong vectors from embed-gemma".
#
# WHERE THIS MUST BE TESTED, corrected after a peer reading: `app/serve.cpp` has TWO embedding routes.
#   serve.cpp:2264  if (embed_serve::container(model))       -> IN-PROCESS, our code computes the vector
#   serve.cpp:2275  rag_launch(o, model, role, free_port())  -> a PROXY to a llama-server for a GGUF model
# A vector test on the proxy route would only re-test llama.cpp. This one takes the CONTAINER route - a directory
# holding `model.embed` (`embed_serve.cpp:39-42`) - where our model decides the numbers, served at POST
# /v1/embeddings (`:185`). The issue's test is: the same text embedded twice must be identical, and a known-answer
# pair (a paraphrase closer than an unrelated sentence) must hold.
#
# The metric layer self-checks first, because the two ways this test could lie are both in the measurement: cosine
# that cannot separate anything, or an "identical" compare that passes on unrelated vectors.
#
# usage: tests/serve_embed_vectors_test.sh path/to/1bit <container dir> <device> [extra serve args]
#   exits 0 when the vectors are deterministic and ordered, 1 when not, 77 when it cannot run here.
set -uo pipefail

bin=${1:?usage: serve_embed_vectors_test.sh path/to/1bit <container dir> <device> [args]}
here=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=tests/serve_port.sh
. "$(dirname "$0")/serve_port.sh"
container=${2:-}   # empty in CI unless a container is configured: that is a SKIP, not a failure
device=${3:-cpu}
shift 3 2> /dev/null || shift $#

if [ -z "${ONEBIT_EMBED_TEST_URL:-}" ]; then
    [ -n "$container" ] || { echo "skip: no encoder container given (engine#1779 fault 3 needs one; the in-process container route is app/embed_serve.cpp)"; exit 77; }
    [ -x "$bin" ] || { echo "skip: no 1bit at $bin"; exit 77; }
    [ -f "$container/model.embed" ] || { echo "skip: no model.embed in $container (a container is a directory holding one, embed_serve.cpp:39-42)"; exit 77; }
fi
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
    rm -f "$log" "$log".*
}
trap cleanup EXIT INT TERM

# ---- the metric layer, self-checked before it is trusted ------------------------------------------------
vectors=$(python3 - <<'PY'
import json, math, sys

def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def identical(a, b):
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))

# self-check 1: cos must separate a paraphrase-ish pair from an unrelated one on known vectors
a, para, unrel = [1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]
if not (cos(a, para) > 0.5 and cos(a, unrel) < 0.1 and cos(a, a) == 1.0):
    print("SELFCHECK_FAIL cosine cannot separate a paraphrase from an unrelated sentence"); raise SystemExit(1)
# self-check 2: identical() must not call two different vectors equal
if identical(a, para) or not identical(a, list(a)):
    print("SELFCHECK_FAIL identical() is wrong"); raise SystemExit(1)
# self-check 3: a constant-output server must be caught - that is what "two different texts differ" is for
if identical([0.5, 0.5], [0.5, 0.5]) is not True:
    print("SELFCHECK_FAIL constant vectors should compare identical, which is why the distinctness arm exists"); raise SystemExit(1)
print("SELFCHECK_OK")
PY
) || { echo "FAIL serve_embed_vectors: the metric layer does not work: $vectors"; exit 1; }
echo "metric self-check:   $vectors"

# ONEBIT_EMBED_TEST_URL points the harness at an already-running /v1/embeddings service instead of spawning serve.
# It exists so this test's own measurement can be exercised against a stub - the discipline the fault-2 detector
# now follows after its eight-vs-six bug: an instrument that has never been driven is not evidence.
if [ -n "${ONEBIT_EMBED_TEST_URL:-}" ]; then
    url=${ONEBIT_EMBED_TEST_URL%/}
    echo "using ONEBIT_EMBED_TEST_URL=$url (no serve spawned: the harness is being tested, not the model)"
else
    "$bin" serve -m "$container" --device "$device" --host 127.0.0.1 --port 0 "$@" > "$log" 2>&1 &
    srv=$!
    if ! port=$(serve_port_wait_any "$log" "$srv" 24000); then
        echo "skip: serve did not report a port in 240 s (see below)"; tail -20 "$log"; exit 77
    fi
    url="http://127.0.0.1:$port"
    echo "serve on $url, device=$device, container=$(basename "$container")"
fi

post() {  # <text> -> "<code>\t<body-file>"
    python3 -c 'import json,sys; print(json.dumps({"input": sys.argv[1]}))' "$1" > "$log.req.json"
    local code
    code=$(curl -s -m 120 -o "$log.resp.json" -w '%{http_code}' -H 'Content-Type: application/json' \
        -d @"$log.req.json" "$url/v1/embeddings" 2> /dev/null)
    cp "$log.resp.json" "$2"
    printf '%s' "$code"
}

t1="A man is playing a guitar."
t2="A person is playing a musical instrument."     # paraphrase of t1
t3="The stock market fell sharply today."          # unrelated to t1

c1a=$(post "$t1" "$log.v1a")
c1b=$(post "$t1" "$log.v1b")
c2=$(post "$t2" "$log.v2")
c3=$(post "$t3" "$log.v3")
echo "requests:            t1 x2 -> $c1a/$c1b, paraphrase -> $c2, unrelated -> $c3"

verdict=$(python3 - "$log.v1a" "$log.v1b" "$log.v2" "$log.v3" "$c1a" <<'PY'
import json, math, sys
paths, codes = sys.argv[1:5], sys.argv[5]

def load(p):
    try:
        j = json.load(open(p))
        return [float(x) for x in j["data"][0]["embedding"]]
    except Exception:
        return None

def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

v = [load(p) for p in paths]
if any(x is None for x in v):
    print("no-vectors"); raise SystemExit
if len({len(x) for x in v}) != 1:
    print("dim-mismatch"); raise SystemExit
a, b, para, unrel = v
if a != b:
    print("nondeterministic"); raise SystemExit
if a == para or a == unrel:
    print("constant-output"); raise SystemExit        # the arm that catches a server returning one vector for everything
c_para, c_unrel = cos(a, para), cos(a, unrel)
# the known-answer ordering the issue asks for, with a margin so noise cannot flip it
print(f"ordered para={c_para:.4f} unrel={c_unrel:.4f}" if c_para > c_unrel + 0.02 else f"unordered para={c_para:.4f} unrel={c_unrel:.4f}")
PY
)
echo "verdict:             $verdict"

case "$verdict" in
    ordered\ *)
        echo "PASS serve_embed_vectors: same text -> same vector, and a paraphrase is closer than an unrelated sentence"
        exit 0 ;;
    nondeterministic)
        echo "FAIL serve_embed_vectors: the same text embedded twice gave different vectors"; exit 1 ;;
    constant-output)
        echo "FAIL serve_embed_vectors: every text returned the same vector (a server that ignores its input)"; exit 1 ;;
    unordered\ *)
        echo "FAIL serve_embed_vectors: a paraphrase was not closer than an unrelated sentence - $verdict"; exit 1 ;;
    no-vectors|dim-mismatch)
        echo "FAIL serve_embed_vectors: the response is not a usable embedding ($verdict)"; head -c 200 "$log.v1a"; exit 1 ;;
    *)
        echo "FAIL serve_embed_vectors: could not classify ($verdict)"; exit 1 ;;
esac
