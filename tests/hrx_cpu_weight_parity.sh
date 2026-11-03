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
# `serve_e2e.sh` asserts each device answers Paris, and no row compares one placement's text with another's. This row
# does, on the same binary and the same model, three ways:
#
#   cpu        -dev none        every node on the CPU, no device at all
#   weightscpu -dev HRX0 -ngl 0 the device present, every layer assigned to the CPU: the device still claims the nodes it
#                               supports and materializes the weights they read, i.e. the partial-offload path #674 names
#   offload    -dev HRX0 -ngl 99  the shipped serve placement
#
# Greedy decoding at temperature 0 must be byte-identical across all three, because placement changes where a number is
# computed, never what it is. A regression here is exactly the failure #674 reported: plausible-looking text that is not
# the CPU's answer.
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

# One greedy completion from a fresh server. Prints the reply's content on stdout; logs go to stderr.
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
    body='{"prompt": "What is the capital of France?", "n_predict": 16, "temperature": 0, "cache_prompt": false}'
    curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' -d "$body" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content", ""))' 2>/dev/null
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -f "$log"
}

cpu=$(arm cpu -dev none -ngl 0 --jinja) || { echo "FAIL cpu arm did not run"; exit 1; }
weightscpu=$(arm weightscpu -dev HRX0 -ngl 0 --jinja) || { echo "FAIL weightscpu arm did not run"; exit 1; }
offload=$(arm offload -dev HRX0 -ngl 99 --jinja) || { echo "FAIL offload arm did not run"; exit 1; }

echo "     cpu        said: $cpu"
echo "     weightscpu said: $weightscpu"
echo "     offload    said: $offload"
check "the CPU answers at all" '[ -n "$cpu" ]'
check "device with every layer on the CPU answers the CPU's text" '[ "$weightscpu" = "$cpu" ]'
check "device with every layer offloaded answers the CPU's text" '[ "$offload" = "$cpu" ]'

[ $fail = 0 ] && echo "PASS placement does not change the answer" || echo "FAIL placement changed the answer"
exit $fail
