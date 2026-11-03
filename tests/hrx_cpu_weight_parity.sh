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
# Placement must not change the answer (engine #674).
#
# #674 was opened as "any graph whose weight ops all run on the CPU computes garbage". The garbage itself was only ever
# produced by a do-not-merge placement patch, but the *class* is real and nothing in this repo tests for it: today
# `serve_e2e.sh` asserts each device answers Paris, and no row compares one placement's answer with another's. This row
# does, on the same binary and the same model, three ways:
#
#   cpu        -dev none        every node on the CPU, no device at all
#   weightscpu -dev HRX0 -ngl 0 the device present, every layer assigned to the CPU: the device still claims the nodes it
#                               supports and materializes the weights they read, i.e. the partial-offload path #674 names
#   offload    -dev HRX0 -ngl 99  the shipped serve placement
#
# WHAT IS COMPARED, and why it is not text (2026-11-03). The first version of this row required the three greedy texts to
# be byte-identical. That is too strict to be true: the CPU path and the device paths are different arithmetic (f32 CPU
# kernels vs HRX's kernels over an f16 KV cache), so their logits differ by ~0.1 nats, and at temperature 0 a *coin-flip*
# between two candidates then flips one token. Measured on the c21 binary (`ca1d42f`), 24 greedy tokens, `n_probs 5`:
#
#   pairs                                  top-5 overlap   median |dlogprob|
#   cpu vs weightscpu  (-ngl 0)                 0.63             0.094 nats
#   cpu vs offload     (-ngl 99, SHIPPED)       0.78             0.084 nats    <- the shipped arm flips a token too
#   weightscpu vs offload                       0.63             0.038 nats
#
# i.e. the text mismatch c21 hit ("also a city called" vs "also a capital city") is a near-tie at step 14 where the CPU's
# own top-1 margin is 0.17 nats, and the *shipped* placement flips its own coin three tokens later. Byte-identical text
# is therefore not achievable across backends, and a row that demands it reports coin flips as placement bugs.
#
# So the row compares the DISTRIBUTION: each device arm against the CPU arm, and the two device arms against each other,
# must still overlap and still agree. Calibration from the same two stacks (see the table above and below):
#
#   correct stack                               overlap 0.63-0.78, median |dlogprob| 0.038-0.094 nats
#   engine #674's corruption (garbage output)   overlap 0.00,      median undefined (nothing shared)
#
# Thresholds: overlap >= 0.30 and median |dlogprob| <= 0.50 nats. Those sit an order of magnitude clear of a correct
# stack in both metrics and cannot survive the corruption, which shares no top-5 candidate at all. The texts are still
# printed, with the step they fork at, because a human reading a failure wants them — they just do not decide the verdict.
#
# KNOWN LIMIT, stated rather than implied: this row detects placement changing the *distribution* (the #674 class: wrong
# numbers, garbage, a collapsed or shifted top-5). It does not certify bit-identical arithmetic across backends, and it
# would not catch a bug that perturbs the answer by less than the cross-backend spread measured above.
#
# The device arm is also the only row that notices the silent no-device trap: on this box the #715 trio is necessary but
# not sufficient (`IREE_HAL_AMDGPU_LIBHSA_PATH` decides whether the HAL finds a runtime that can open the device at all),
# and a missing variable used to look exactly like a null result. So when the machine has a GPU this test FAILS on "no
# HRX0", rather than skipping, and says which variable to look at.
#
# usage: tests/hrx_cpu_weight_parity.sh path/to/llama-server <model.gguf>
# Skips (exit 77) on a machine without a GPU HRX has kernels for: ONEBIT_HRX_E2E_TARGETS (default gfx1151), looked up in
# sysfs by scripts/gpu-targets.sh, which opens no GPU.
set -uo pipefail

server=${1:?usage: hrx_cpu_weight_parity.sh path/to/llama-server <model.gguf>}
model=${2:?}
root=$(cd "$(dirname "$0")/.." && pwd)

have=$("$root/scripts/gpu-targets.sh" 2>/dev/null | tr '\n' ' ')
want=${ONEBIT_HRX_E2E_TARGETS:-gfx1151}
found=0
for t in $want; do case " $have " in *" $t "*) found=1 ;; esac; done
[ $found = 1 ] || { echo "skip: no $want GPU for HRX on this machine (GPUs: ${have:-none})"; exit 77; }

# The recipe every lane uses for a tool the engine does not launch itself (engine#715 trio, engine#315/#674 runtime).
# shellcheck source=../scripts/hrx-env.sh
. "$root/scripts/hrx-env.sh"

fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

devices=$("$server" --list-devices 2>&1)
if [[ "$devices" != *HRX0* ]]; then
    echo "FAIL the GPU is present but $server lists no HRX0"
    echo "$devices" | sed 's/^/     /'
    echo "     HRX needs a runtime it can open the device through: set IREE_HAL_AMDGPU_LIBHSA_PATH"
    echo "     (source scripts/hrx-env.sh sets it when a TheRock library is installed)."
    exit 1
fi

free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# One greedy completion from a fresh server, JSON saved to $work/$tag.json; logs go to stderr.
arm() { # $1 tag, rest = llama-server args
    local tag=$1; shift
    local port log pid body
    port=$(free_port)
    log=$(mktemp)
    "$server" -m "$model" --host 127.0.0.1 --port "$port" --alias parity "$@" >"$log" 2>&1 &
    pid=$!
    local up=0
    for _ in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then up=1; break; fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if [ $up = 0 ]; then
        echo "     $tag: server did not come up" >&2
        tail -5 "$log" >&2
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -f "$log"
        return 4
    fi
    # n_probs 5: the comparison needs each step's top-5 logprobs, not only the sampled token
    body='{"prompt": "What is the capital of France?", "n_predict": 24, "temperature": 0, "cache_prompt": false, "n_probs": 5}'
    curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' -d "$body" >"$work/$tag.json"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -f "$log"
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("content",""))' "$work/$tag.json" 2>/dev/null
}

cpu=$(arm cpu -dev none -ngl 0 --jinja) || { echo "FAIL cpu arm did not run"; exit 1; }
weightscpu=$(arm weightscpu -dev HRX0 -ngl 0 --jinja) || { echo "FAIL weightscpu arm did not run"; exit 1; }
offload=$(arm offload -dev HRX0 -ngl 99 --jinja) || { echo "FAIL offload arm did not run"; exit 1; }

echo "     cpu        said: $cpu"
echo "     weightscpu said: $weightscpu"
echo "     offload    said: $offload"
check "the CPU answers at all" '[ -n "$cpu" ]'

# Distribution agreement, per arm pair. Prints the numbers it judged on, so a failure is readable without rerunning.
compare() { # $1 left tag, $2 right tag; prints "<overlap> <median|d|>"
    python3 - "$work/$1.json" "$work/$2.json" <<'PY'
import json, sys
def load(p):
    cp = json.load(open(p)).get("completion_probabilities", [])
    return [[(t["token"], t["logprob"]) for t in s.get("top_logprobs", [])] for s in cp]
a, b = load(sys.argv[1]), load(sys.argv[2])
ds, shared, possible = [], 0, 0
for i in range(min(len(a), len(b))):
    right = dict(b[i])
    for tok, lp in a[i]:
        possible += 1
        if tok in right:
            shared += 1
            ds.append(abs(lp - right[tok]))
ds.sort()
overlap = shared / possible if possible else 0.0
median = ds[len(ds) // 2] if ds else float("nan")
print(f"{overlap:.2f} {median:.4f}")
PY
}
agree() { # $1 description, $2 left, $3 right
    local overlap median
    read -r overlap median <<<"$(compare "$2" "$3")"
    echo "     $2 vs $3: top-5 overlap $overlap, median |dlogprob| $median nats"
    if python3 -c "import sys; sys.exit(0 if $overlap >= 0.30 and $median <= 0.50 else 1)"; then
        echo "ok   $1"
    else
        echo "FAIL $1"
        fail=1
    fi
}

agree "placement onto the CPU keeps the CPU's distribution"      cpu weightscpu
agree "full offload keeps the CPU's distribution"                 cpu offload
agree "the two device placements agree with each other"           weightscpu offload

[ $fail = 0 ] && echo "PASS placement does not change the answer" || echo "FAIL placement changed the answer"
exit $fail
