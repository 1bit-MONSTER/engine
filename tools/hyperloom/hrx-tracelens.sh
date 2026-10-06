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
# hrx-tracelens.sh <model.gguf> <prompt|decode> [VAR=value ...]
#   Profile one HRX run (prompt: llama-bench -p 512 -n 0; decode: llama-server, 8 generated tokens),
#   convert it with hrx2kineto.py (shapes from GGML_HRX_DISPATCH_SHAPE_LOG), and run TraceLens with the
#   hrx extension + gfx1151 arch JSON. Output: ~/lb/tracelens/<ts>-<model>-<mode>[-TAG]/ (summary.txt =
#   converter roofline table, report.xlsx + csvs/ = TraceLens, tracediff_top.txt with COMPARE). VAR=value
#   pairs are exported for the run (e.g. GGML_HRX_DISABLE_DISPATCH=hip.). Env: HRX_BIN (build/bin with the
#   shape log), TAG, COMPARE=<other
#   run dir> (adds TraceDiff vs that run's trace), PROMPT_TOKENS (512), DECODE_TOKENS (8), OUT_ROOT
#   (~/lb/tracelens), BOX_LOCK (flock file shared with other GPU jobs; empty = none), IREE_PROFILE, H2K.
#   Needs a llama.cpp build with GGML_HRX_DISPATCH_SHAPE_LOG and TraceLens in ~/.cache/tracelens-venv.
#   Aborts the run at 93 C Tctl; needs MemAvailable >= model size + 11 GB.
set -uo pipefail
[ $# -ge 2 ] || { sed -n 17,28p "$0"; exit 2; }
MODEL=$(readlink -f "$1"); MODE=$2; shift 2
for kv in "$@"; do export "$kv"; done
BIN=${HRX_BIN:-$HOME/wt/hrx2kineto-src/build/bin}
# iree-profile must match the IREE the HRX build wrote its profile with: a stale tool rejects the file with
# "unsupported IREE HAL profile file version". The deps build a given HRX checkout uses sits beside its
# bin/, so derive the tool from HRX_BIN; IREE_PROFILE overrides, then PATH.
IP=${IREE_PROFILE:-}
if [ -z "$IP" ] || [ ! -x "$IP" ]; then
  IP=$(dirname "$(readlink -f "$BIN")")/../ggml/src/ggml-hrx/hrx/src/ggml-hrx-deps-build/runtime/src/iree/tools/iree-profile/iree-profile
  [ -x "$IP" ] || IP=$(command -v iree-profile 2>/dev/null || echo "$IP")
fi
H2K=${H2K:-$(dirname "$(readlink -f "$0")")/hrx2kineto.py}
TL=${TRACELENS_BIN:-$HOME/.cache/tracelens-venv/bin}
ROOT=${OUT_ROOT:-$HOME/lb/tracelens}
O=$ROOT/$(date +%Y%m%d-%H%M%S)-$(basename "$MODEL" .gguf)-$MODE${TAG:+-$TAG}
mkdir -p "$O"; ln -sfn "$O" "$ROOT/latest"
export LD_LIBRARY_PATH=$BIN
export IREE_HAL_AMDGPU_LIBHSA_PATH=${IREE_HAL_AMDGPU_LIBHSA_PATH:-/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1}
export HRX_PROFILE_MODE=dispatch HRX_PROFILE_FILE=$O/run.prof GGML_HRX_DISPATCH_SHAPE_LOG=$O/shapes.jsonl
K=$(for h in /sys/class/hwmon/*; do [ "$(cat $h/name)" = k10temp ] && echo $h; done)
avail_gb() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
need=$(( $(stat -c %s "$MODEL") / 1073741824 + 11 ))
[ "$(avail_gb)" -ge $need ] || { echo "MemAvailable $(avail_gb) GB < $need GB, not running" | tee "$O/run.log"
                                 exit 1; }
printf '%s\n' "model=$MODEL mode=$MODE bin=$BIN env: $*" > "$O/run.log"

LOCK=${BOX_LOCK-$HOME/.cache/lax-decode/box.lock}
[ -n "$LOCK" ] && { exec 9>"$LOCK"; flock 9; }
( while :; do
    t=$(( $(cat $K/temp1_input) / 1000 ))
    [ $t -ge 93 ] && { echo "$(date +%T) THERMAL ABORT $t C" >> "$O/run.log"; pkill -f "$BIN/llama-"; }
    sleep 1
  done ) & TG=$!
trap 'kill $TG 2>/dev/null' EXIT
if [ "$MODE" = prompt ]; then
  "$BIN/llama-bench" -m "$MODEL" -dev HRX0 -ngl 99 -fa 1 -p "${PROMPT_TOKENS:-512}" -n 0 -r 1 \
    > "$O/bench.txt" 2> "$O/bench.err"
  PHASE=prefill
else
  P=$((18500 + RANDOM % 400))
  "$BIN/llama-server" -m "$MODEL" -dev HRX0 -ngl 99 -fa on --host 127.0.0.1 --port $P -c 2048 -np 1 \
    --no-webui > "$O/server.log" 2>&1 & S=$!
  for i in $(seq 300); do
    curl -sf localhost:$P/health >/dev/null && break; kill -0 $S 2>/dev/null || break; sleep 1
  done
  curl -s localhost:$P/completion -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"The history of the city of Paris\",\"n_predict\":${DECODE_TOKENS:-8},\"temperature\":0}" \
    > "$O/completion.json"
  kill $S; wait $S 2>/dev/null
  PHASE=decode
fi
kill $TG 2>/dev/null; [ -n "$LOCK" ] && { flock -u 9; exec 9>&-; }
echo "Tctl $(( $(cat $K/temp1_input) / 1000 )) C after the run" >> "$O/run.log"

"$IP" dispatch --format=jsonl --dispatch_events "$O/run.prof" > "$O/events.jsonl"
"$IP" command --format=jsonl "$O/run.prof" > "$O/commands.jsonl"
python3 "$H2K" "$O/events.jsonl" "$O/trace.json" --shapes "$O/shapes.jsonl" --commands "$O/commands.jsonl" \
  --phase $PHASE --skip 1 --write-arch "$O/gfx1151.json" --summary | tee "$O/summary.txt"
# FALLBACK: op-level conversion needs each command buffer's dispatch order to equal the logged program's build
# order. The runtime can reorder dispatches inside a command buffer (multisets match, orders do not), in which
# case map_command_buffers finds no candidate and writes no trace -- which used to fail the Hyperloom profile
# phase outright. A kernel-level trace (raw dispatch keys; op names land in "other") is still accepted by
# TraceLens with the extension and arch, so the phase produces real data instead of failing. Removing this needs
# an order-independent matcher (see README "Known limitations").
if [ ! -s "$O/trace.json" ]; then
  echo "hrx-tracelens: op-level match found no program (runtime reorders command buffers vs the logged programs); falling back to a kernel-level trace" >&2
  python3 "$H2K" "$O/events.jsonl" "$O/trace.json" 2>&1 | tee -a "$O/summary.txt" >&2
fi
CMP=()
[ -n "${COMPARE:-}" ] && CMP=(--comparison_json_path "$COMPARE/trace.json")
"$TL/TraceLens_generate_perf_report_pytorch" --profile_json_path "$O/trace.json" \
  --output_xlsx_path "$O/report.xlsx" --output_csvs_dir "$O/csvs" --extension_file "$H2K" \
  --gpu_arch_json_path "$O/gfx1151.json" "${CMP[@]}" > "$O/tracelens.log" 2>&1 \
  || echo "TraceLens failed, see $O/tracelens.log"
# TraceDiff (this run = trace1, COMPARE = trace2): largest per-step time differences by aligned subtree
if [ -n "${COMPARE:-}" ] && [ -f "$O/csvs/diff_stats.csv" ]; then
  python3 "$H2K" --tracediff-top "$O/csvs/diff_stats.csv" --steps-from "$O/trace.json" | tee "$O/tracediff_top.txt"
fi
echo "$O"
