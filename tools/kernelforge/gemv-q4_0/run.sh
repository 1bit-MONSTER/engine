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
# run.sh WORKSPACE [MAX_HOURS]: a KernelForge campaign on the Q4_0 x Q8_1 few-column matmul (speculative verification), gfx1151
# (tools/kernelforge/README.md). Copies the task into WORKSPACE as a fresh git repository
# (forge-loop edits the kernel in place) and runs `kernelforge forge-loop` there.
# Needs: Hyperloom installed with `.[forge]` in the Python on PATH (a ROCm torch), and a
# `claude` CLI logged in; the campaign bills that account.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
W=${1:?usage: run.sh WORKSPACE [MAX_HOURS]}
MAX_HOURS=${2:-2}
[ -e "$W" ] && { echo "run.sh: $W exists; pick a new workspace"; exit 1; }
command -v kernelforge >/dev/null || { echo "run.sh: kernelforge not on PATH (pip install -e '.[forge]' in a Hyperloom checkout)"; exit 1; }
command -v claude >/dev/null || { echo "run.sh: no claude CLI on PATH; KernelForge drives its agent through it"; exit 1; }

mkdir -p "$W"
cp "$here/gemv_q4_0_kernel.py" "$here/driver.py" "$here/program.md" "$W/"
cd "$W"
printf 'forge_experiments/\n__pycache__/\n' > .gitignore
git init -q && git add -A && git -c user.name=forge -c user.email=forge@localhost commit -qm "task: gemv q4_0 (gfx1151)"

export GPU_TARGET=gfx1151
# a subscription login: keep sessions off --bare (a local switch in our Hyperloom checkout, README.md)
export FORGE_CLAUDE_OAUTH=${FORGE_CLAUDE_OAUTH:-1}
exec kernelforge forge-loop \
    --kernel "$W/gemv_q4_0_kernel.py" \
    --driver "$W/driver.py" \
    --workspace "$W" \
    --experiments-dir "$W/forge_experiments" \
    --result-json "$W/forge_experiments/forge_result.json" \
    --program-md-file "$W/program.md" \
    --kernel-backend hip \
    --gpu-target gfx1151 \
    --snr-threshold 90.0 \
    --max-hours "$MAX_HOURS" \
    --git-branch forge-optimize \
    --target-functions "gemv_q4_0"
