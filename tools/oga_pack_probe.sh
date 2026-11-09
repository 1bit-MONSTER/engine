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
# oga_pack_probe.sh -- how far does a Ryzen AI OGA pack get in OUR runtime? (engine#1371's OGA-pack arm)
#
# #1371 asks whether a drafter pays on the OGA-pack path. Answering that needs the pack to *serve* first, and the road to that
# has three separate gates, each of which fails differently:
#
#   1. import       `1bit npu-pack import <pack>` -- needs the .fconst layer blocks to be locatable (a 1.9 pack renamed them,
#                   engine#1410) and reports the tensors it could not import;
#   2. forward init `1bit forward-serve -m <dir>/model.q4nx` -- needs the quantized *embedding* to dequantise (the OGA import
#                   writes q4_1 in the vendor's layout; npu/forward/src/q4nx_forward.cpp:181 accepts only our 5120-byte q4_1
#                   tile / Q4_K / Q8 chunks), and the derived designs to have kernel ELFs;
#   3. serving      then, and only then, a drafter measurement means anything.
#
# This script runs gates 1 and 2 and says which one bit, with the exact line. Gate 2 needs the NPU, so run the whole thing
# inside ONE npu-hold ticket; the server is started here and killed by a trap on every exit path (an orphaned llama-server took
# 51 GiB of GTT on 11-08, so nothing this script starts outlives it).
#
#   npu-hold -t 4 -l "oga probe" -- tools/oga_pack_probe.sh models/oga/<pack dir> [port]
#
# Exit: 0 = the pack served (gate 3 is then worth doing), 3 = import refused/incomplete-warned, 4 = forward init failed,
#       5 = no health, 2 = usage.
set -u
PACK=${1:?usage: oga_pack_probe.sh <pack dir> [port]}
PORT=${2:-18160}
BIN=${BIN:-${ONEBIT_BIN:-}}
OUT=${OUT:-/tmp/oga-probe-$(basename "$PACK")}

[ -d "$PACK" ] || { echo "oga_pack_probe: no such pack dir: $PACK" >&2; exit 2; }
if [ -z "$BIN" ]; then
    for c in /home/bcloud/wt/serve-bin/1bit-t2s2-20261102 /home/bcloud/.local/share/1bit-embed-c22/*/1bit/1bit; do
        [ -x "$c" ] && { BIN=$c; break; }
    done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "oga_pack_probe: no 1bit binary found; set BIN=" >&2; exit 2; }

echo "== pack: $PACK"
echo "== bin:  $BIN"
echo "== gate 1: import -> $OUT"
rm -rf "$OUT"
imp=$($BIN npu-pack import "$PACK" -o "$OUT" --allow-incomplete 2>&1)
rc=$?
printf '%s\n' "$imp" | grep -E 'fconst|attention|parameters|container|incomplete' | sed 's/^/   /'
if [ $rc -ne 0 ] || [ ! -f "$OUT/model.q4nx" ]; then
    echo "== verdict: import produced no container (rc=$rc) -- gate 1 is where this pack stops"
    exit 3
fi

echo "== gate 2: forward init on the NPU (this needs the ticket you are in)"
"$BIN" forward-serve -m "$OUT/model.q4nx" -p "$PORT" > /tmp/oga-probe-serve.log 2>&1 &
pid=$!
trap 'kill -9 $pid 2>/dev/null' EXIT INT TERM HUP
for _ in $(seq 1 30); do
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$code" = 200 ] && break
    kill -0 $pid 2>/dev/null || break
    sleep 2
done
if [ "${code:-}" != 200 ]; then
    echo "   server log:"
    tail -8 /tmp/oga-probe-serve.log | sed 's/^/     /'
    if grep -q 'dequant failed' /tmp/oga-probe-serve.log; then
        echo "== verdict: forward init stopped on the quantized EMBEDDING (q4_1 in the vendor layout);"
        echo "            npu/forward/src/q4nx_forward.cpp:181 takes only our 5120-byte q4_1 tile / Q4_K / Q8 chunks"
    elif grep -q 'missing .*ELF' /tmp/oga-probe-serve.log; then
        echo "== verdict: forward init stopped on kernel ELFs -- this pack's shapes have no compiled kernels here"
    else
        echo "== verdict: forward init failed (see the log above)"
    fi
    exit 4
fi

echo "== verdict: HEALTH 200 -- the pack serves; gate 3 (a drafter measurement) is worth doing"
curl -s -m 120 -X POST "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d '{"prompt":"Q: Explain in plain words why the sky appears blue. A:","max_tokens":12,"temperature":0}' | head -c 300
echo
exit 0
