#!/bin/bash
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
# Lemonade's standard benchmark (`lemonade bench`, the same call as its nightly
# benchmark-regression CI: chat-short + code-short, 1 warmup, 5 runs, ctx 4096) run against a
# llama-server build of our fork. The fork's server is routed in the way Lemonade's CI routes
# forks: llamacpp.vulkan_bin = the server file. Lemonade reserves -dev/--device, so the device
# goes in through llama-server's own LLAMA_ARG_DEVICE environment variable (default HRX0).
# A private lemond (own cache/config dirs, free port, no broadcast) so nothing touches the
# user's Lemonade install.
#
# usage: [LLAMA_ARG_DEVICE=HRX0] lemonade_bench.sh LLAMA_SERVER GGUF OUT_JSON [SERVER_ARGS]
set -uo pipefail
SERVER=$(readlink -f "$1"); GGUF=$(readlink -f "$2"); OUT=$3; ARGS=${4:-}
export LLAMA_ARG_DEVICE=${LLAMA_ARG_DEVICE:-HRX0}
L=${LEMONADE_BIN_DIR:-$HOME/.cache/lemonade-pin/build}   # lemond reads resources/ next to itself
W=$(mktemp -d "${TMPDIR:-/tmp}/lembench.XXXXXX")
mkdir -p "$W/cache" "$W/config" "$W/models"
ln -s "$GGUF" "$W/models/$(basename "$GGUF")"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$L/lemond" --host 127.0.0.1 --port "$PORT" --no-broadcast --log-file "$W/lemond.log" "$W/cache" "$W/config" > "$W/lemond.out" 2>&1 &
LP=$!
C=("$L/lemonade" --host 127.0.0.1 --port "$PORT" --no-discovery)
cleanup() { "${C[@]}" unload > /dev/null 2>&1; kill $LP 2>/dev/null; wait $LP 2>/dev/null
  mkdir -p "$OUT.logs"; cp "$W"/*.out "$W"/*.log "$OUT.logs/" 2>/dev/null; rm -rf "$W"; }
trap cleanup EXIT
for _ in $(seq 60); do "${C[@]}" status > /dev/null 2>&1 && break; sleep 1; done
"${C[@]}" config set llamacpp.backend=vulkan llamacpp.prefer_system=false \
  "llamacpp.vulkan_bin=$SERVER" "llamacpp.vulkan_args=$ARGS" "extra_models_dir=$W/models" > "$W/config.out" 2>&1 \
  || { cat "$W/config.out" >&2; exit 1; }
STEM=$(basename "$GGUF" .gguf)
MODEL=$("${C[@]}" list 2>/dev/null | awk -v s="$STEM" '$1 ~ s {print $1; exit}')
[ -n "$MODEL" ] || { echo "lemonade_bench: local model not registered" >&2; "${C[@]}" list >&2; exit 1; }
"${C[@]}" bench "$MODEL" --backend vulkan --scenarios chat-short --scenarios code-short \
  --runs 5 --warmup 1 --json --output "$OUT" > "$W/bench.out" 2>&1
rc=$?
# lemonade bench exits 0 even when every run failed
[ $rc -eq 0 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(any(sc.get('all_runs_failed') for m in d['models'] for r in m['results'] for sc in r['scenarios']))" "$OUT" || rc=${rc/#0/2}
[ $rc -eq 0 ] || { tail -30 "$W/bench.out" >&2; tail -30 "$W/lemond.log" >&2; }
exit $rc
