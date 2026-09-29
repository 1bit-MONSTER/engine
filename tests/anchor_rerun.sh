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
# Re-run the anchor completion on `1bit serve` for the target models.
# usage: anchor_rerun.sh <1bit> <outdir> <device> <alias>:<gguf> [<alias>:<gguf> ...]
set -uo pipefail

bin=${1:?usage: anchor_rerun.sh <1bit> <outdir> <device> <alias>:<gguf> ...}
outdir=${2:?}
device=${3:?}
shift 3
mkdir -p "$outdir"

for spec in "$@"; do
    alias=${spec%%:*}
    gguf=${spec#*:}
    if [ ! -f "$gguf" ]; then
        echo "SKIP $alias: gguf missing: $gguf"
        continue
    fi
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    log=$(mktemp /tmp/serve-rerun-${alias}.XXXX.log)
    t0=$(date +%s)
    "$bin" serve -m "$gguf" --device "$device" --port "$port" --alias "$alias" >"$log" 2>&1 &
    pid=$!
    code=000
    for _ in $(seq 1 3600); do
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null)
        [ "$code" = 200 ] && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if [ "$code" != 200 ]; then
        echo "FAIL $alias: /health never 200 (code=$code); log=$log"
        echo "--- tail ---"; tail -25 "$log"
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        continue
    fi
    reply=$(curl -s "http://127.0.0.1:$port/v1/completions" \
        -d "{\"model\":\"$alias\",\"prompt\":\"The capital of France is\",\"max_tokens\":1,\"temperature\":0}")
    t1=$(date +%s)
    echo "$reply" > "$outdir/rerun_${alias}.json"
    text=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["choices"][0]["text"])' "$reply" 2>/dev/null)
    if [[ "$text" == *"Paris"* ]]; then
        echo "PASS $alias: text='$text' ($((t1-t0))s) -> $outdir/rerun_${alias}.json"
    else
        echo "FAIL $alias: text='$text' (full: $(head -c 300 <<<"$reply"))"
    fi
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
done
echo "done"
