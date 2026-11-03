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
# must still overlap and still agree. Calibration from the same two stacks, scored the way this row scores them (see
# "shared history" below):
#
#   correct stack (c21 `ca1d42f`, cpu vs weightscpu / offload / weightscpu vs offload)
#                                                      overlap 0.96-0.97, median |dlogprob| 0.036-0.085 nats
#   engine #674's corruption (garbage output)          overlap 0.00,      median undefined (nothing shared, 1 shared step)
#
# Thresholds: overlap >= 0.30 and median |dlogprob| <= 0.50 nats, judged on the FULL-PRECISION values (the printed
# numbers are rounded for reading only, so 0.50004 cannot pass a 0.50 limit). They sit ~3x clear of a correct stack in
# both metrics and cannot survive the corruption, which shares no top-5 candidate at all. The texts are still printed,
# with the step they fork at and the number of steps compared, because a human reading a failure wants them — they just
# do not decide the verdict.
#
# SHARED HISTORY, which is what makes those numbers mean anything. At temperature 0 the arms produce the same tokens
# until one step flips, and from the NEXT step on each arm is conditioning on its own different prefix: comparing their
# distributions there compares two different computations, and it drags a correct stack's overlap down to 0.63-0.78 for
# no reason. So a pair is scored only over the steps whose input history is the same for both arms — every step up to and
# including the first fork (the fork step itself is comparable: identical prefix, different sample) — and it is judged
# only when that shared prefix is at least kMinSteps long. A pair that forks before then fails, and the message says so
# explicitly rather than blaming truncation: after a fork the distributions are not comparable at all.
#
# KNOWN LIMIT, stated rather than implied: this row detects placement changing the *distribution* (the #674 class: wrong
# numbers, garbage, a collapsed or shifted top-5). It does not certify bit-identical arithmetic across backends, and it
# would not catch a bug that perturbs the answer by less than the cross-backend spread measured above. It also cannot
# judge a pair that diverges inside the first kMinSteps steps, where it fails closed (correct stack forks at step 14 and
# 18, so the margin is four steps).
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
# How many steps a pair must share before its numbers mean anything. An arm that ends early (EOS, a truncated
# response, or a placement bug that stops the graph) leaves only a short comparable prefix, and a short prefix always
# agrees: one step of two identical top-5 lists is overlap 1.00 and median 0.0000. That is the failure this row exists
# to catch, so a pair is judged only when it has at least this many steps, and only over steps the two arms share a
# history for (see SHARED HISTORY above).
kMinSteps=12

compare() { # $1 left tag, $2 right tag; prints "<overlap> <median> <steps> <scored> <fork> <verdict>"
    python3 - "$work/$1.json" "$work/$2.json" <<'PY'
import json, sys
def load(p):
    cp = json.load(open(p)).get("completion_probabilities", [])
    return [{"id": s.get("id"),
             "top": [(t["token"], t["logprob"]) for t in s.get("top_logprobs", [])]} for s in cp]
a, b = load(sys.argv[1]), load(sys.argv[2])
steps = min(len(a), len(b))
# First step whose sampled token differs. Every step before it had an identical history, and that step had one too (same
# prefix, different sample), so the comparable prefix is [0, fork]. After it the arms condition on different tokens and
# their distributions describe different computations.
fork = next((i for i in range(steps) if a[i]["id"] != b[i]["id"]), -1)
scored = steps if fork < 0 else fork + 1
ds, shared, possible = [], 0, 0
for i in range(scored):
    right = dict(b[i]["top"])
    for tok, lp in a[i]["top"]:
        possible += 1
        if tok in right:
            shared += 1
            ds.append(abs(lp - right[tok]))
ds.sort()
overlap = shared / possible if possible else 0.0
n = len(ds)
# True median: an even sample averages its two middle values (the upper-middle element is not the median). nan when
# nothing is shared, so the printed number says "undefined" instead of a 0.0 that looks like agreement.
median = float("nan") if not n else (ds[n // 2] if n % 2 else 0.5 * (ds[n // 2 - 1] + ds[n // 2]))
# The verdict is decided HERE, on the full-precision values; the two printed numbers are rounded for reading. Comparing
# the rounded strings let 0.50004 pass a <= 0.50 limit by exactly the amount that was hidden.
verdict = "ok" if (overlap >= 0.30 and median == median and median <= 0.50) else "limit"
print(f"{overlap:.2f} {median:.4f} {steps} {scored} {fork} {verdict}")
PY
}
agree() { # $1 description, $2 left, $3 right
    local overlap median steps scored fork verdict where
    read -r overlap median steps scored fork verdict <<<"$(compare "$2" "$3")"
    if [ "$fork" -lt 0 ]; then
        if [ "$steps" -eq 0 ]; then where="n/a (one arm returned no comparable steps)"
        else where="none (the texts are identical over the $steps compared steps)"; fi
    else where="step $fork"; fi
    echo "     $2 vs $3: top-5 overlap $overlap, median |dlogprob| $median nats over $scored shared-history step(s) of $steps, texts fork at $where"
    if [ "$steps" -lt $kMinSteps ]; then
        echo "FAIL $1"
        echo "     only $steps comparable step(s), the floor is $kMinSteps: an arm that stops early is not evidence of agreement"
        fail=1
        return
    fi
    if [ "$scored" -lt $kMinSteps ]; then
        echo "FAIL $1"
        echo "     the texts fork at $where, so only $scored step(s) share a history, below the $kMinSteps floor. After a"
        echo "     fork the two arms condition on different tokens, so comparing their distributions past it compares two"
        echo "     different computations — not evidence either way."
        fail=1
        return
    fi
    if [ "$verdict" = ok ]; then
        echo "ok   $1"
    else
        echo "FAIL $1"
        echo "     thresholds are top-5 overlap >= 0.30 and median |dlogprob| <= 0.50 nats, judged on the full-precision"
        echo "     values ($overlap and $median here, rounded for printing)."
        fail=1
    fi
}

agree "placement onto the CPU keeps the CPU's distribution"      cpu weightscpu
agree "full offload keeps the CPU's distribution"                 cpu offload
agree "the two device placements agree with each other"           weightscpu offload

[ $fail = 0 ] && echo "PASS placement does not change the answer" || echo "FAIL placement changed the answer"
exit $fail
