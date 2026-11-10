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
# gpu_hold_queue_test.sh: the GPU hold queue is the FLEET's when it is installed, whoever runs (engine #1504).
#
# A runner under a different account, a tmux lane and a `systemd --user` service must all queue on the same
# token file. The failure this guards against is silent: a caller whose `$HOME` is somewhere else, and which
# never sourced a profile file, takes its OWN token and runs GPU work beside the fleet.
#
# Arms:
#   1. the fleet state exists -> a claim with NO environment at all lands in the fleet queue and uses the
#      fleet token file;
#   2. the fleet state is absent -> the old ~/GPU-* paths are used, so a box that has not been set up is
#      unchanged (that is what every caller did before).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
hold=$(mktemp -d); fake_home=$(mktemp -d)
trap 'rm -rf "$hold" "$fake_home"' EXIT
fail=0

claim() {   # claim: run one very short claim with the given HOME and hold dir, nothing else in the environment
    # the output is kept: gpu-hold logs "took <token>", which is the only evidence of WHICH token a claim used
    # after it released it (release is `rm -f "$TOKEN"`, by design)
    HOME=$fake_home ONEBIT_HOLD_DIR=$hold \
    GPU_HOLD_TCTL_START=${GPU_HOLD_TCTL_START:-999} GPU_HOLD_RELEASE_WAIT=0 \
        bash "$here/../scripts/gpu-hold" -l gpu-hold-queue-test -t 0.05 -w 10 true 2>&1
}

# 1. the fleet's state is installed: the queue and the token are the fleet's, with no environment to say so
mkdir -p "$hold/gpu-queue"   # and NOT the token: its existence means "held", so the holder must create it
if out=$(claim); then
    if printf '%s' "$out" | grep -q "took $hold/gpu-token" &&
       [ -e "$hold/gpu-queue/.released-at.gpu-hold-queue-test" ] && [ ! -d "$fake_home/GPU-QUEUE" ]; then
        echo "PASS  a claim with no environment queues on the fleet's state and takes the fleet token"
    else
        echo "FAIL  the claim did not use the fleet's state (fleet queue: $(ls -A "$hold/gpu-queue" 2>/dev/null | tr '\n' ' '); ~/GPU-QUEUE: $([ -d "$fake_home/GPU-QUEUE" ] && echo present || echo absent)); gpu-hold said:"
        printf '%s\n' "$out" | tail -3 | sed 's/^/      /'
        fail=$((fail + 1))
    fi
else
    echo "FAIL  the claim itself failed (rc $?); see the gpu-hold output above"
    fail=$((fail + 1))
fi

# 2. nothing installed: the old paths, so this change is inert on a box without the shared state
hold2=$(mktemp -d); rm -rf "$hold2"; mkdir -p "$fake_home"
if HOME=$fake_home ONEBIT_HOLD_DIR=$hold2 GPU_HOLD_TCTL_START=${GPU_HOLD_TCTL_START:-999} GPU_HOLD_RELEASE_WAIT=0 \
        bash "$here/../scripts/gpu-hold" -l gpu-hold-queue-test2 -t 0.05 -w 10 true >/dev/null 2>&1; then
    if [ -e "$fake_home/GPU-QUEUE/.released-at.gpu-hold-queue-test2" ]; then
        echo "PASS  with no fleet state the claim falls back to ~/GPU-QUEUE, as before"
    else
        echo "FAIL  with no fleet state the claim did not use ~/GPU-QUEUE"; fail=$((fail + 1))
    fi
else
    echo "FAIL  the fallback claim failed (rc $?)"; fail=$((fail + 1))
fi
r=$([ "$fail" = 0 ] && echo ok || echo no); rm -rf "$hold2"
[ "$r" = ok ] && echo "gpu_hold_queue: all rows PASS"
[ "$fail" = 0 ]
