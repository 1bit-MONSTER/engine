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
#                   writes q4_1 in the vendor's layout; the forward's `dequant_any()` accepts only our 5120-byte q4_1 tile, a
#                   Q4_K chunk or an Q8 chunk, and the embedding branch raises "embedding dtype … but dequant failed" when the
#                   tensor is none of those), and the derived designs to have kernel ELFs;
#   3. serving      then, and only then, a drafter measurement means anything.
#
# This script runs gates 1 and 2 and says which one bit, with the exact line. Gate 2 needs the NPU, so run the whole thing
# inside ONE npu-hold ticket; the server is started here and killed by a trap on every exit path (an orphaned llama-server took
# 51 GiB of GTT on 11-08, so nothing this script starts outlives it).
#
#   npu-hold -t 4 -l "oga probe" -- tools/oga_pack_probe.sh models/oga/<pack dir> [port]
#
# Exit: 0 = the pack served (gate 3 is then worth doing), 3 = gate 1 only (no device ticket detected, or import refused),
#       4 = forward init failed inside a ticket, 5 = no health, 2 = usage / REQUIRE_TICKET refusal.
set -u
PACK=${1:?usage: oga_pack_probe.sh <pack dir> [port]}
PORT=${2:-18160}
BIN=${BIN:-${ONEBIT_BIN:-$(command -v 1bit 2>/dev/null || command -v onebit 2>/dev/null)}}
# Scratch goes to a DISK-backed dir, never /tmp: an imported container is hundreds of MB to several GB, and this box's tmpfs hit
# 0 MB free on 2026-11-08 and broke every lane's tool output. KEEP=1 leaves it behind, otherwise it is removed on exit.
SCRATCH=${SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/oga-probe}
pack_base=$(basename "$PACK")
# `rm -rf "$OUT"` runs before gate 1, and OUT defaults to $SCRATCH/<basename of the pack>: a pack given as "." or ".." would make
# that the scratch root itself and wipe every other probe's container. Refuse rather than tidy up after it.
case "$pack_base" in ''|.|..|/) echo "oga_pack_probe: refusing a pack path whose basename is '${pack_base}' -- set OUT= explicitly" >&2; exit 2;; esac
OUT=${OUT:-$SCRATCH/$pack_base}
KEEP=${KEEP:-0}
LOG=$OUT/serve.log

[ -d "$PACK" ] || { echo "oga_pack_probe: no such pack dir: $PACK" >&2; exit 2; }

# Gate 2 touches the NPU, so this script is only correct inside a device ticket. The check is "is an ancestor of mine the pid
# the token file names?", which is what npu-kernels#32 settled on after the GPU_TOKEN_FILE idea failed: the helpers are bash
# scripts, so a running npu-hold's argv[0] is the INTERPRETER and its script path is argv[1] -- matching argv[0] (the first
# version of this guard) can never match, and matching the whole command line let a shell whose text merely mentioned
# npu-hold pass the guard on its own grep. The token file is the holder's own statement:
#   gpu-hold <label> pid <PID> <date>
TOKEN_FILES=${TOKEN_FILES:-${NPU_TOKEN_FILE:-$HOME/NPU-TOKEN}:${GPU_TOKEN_FILE:-$HOME/GPU-TOKEN}}
token_pids() {
    local f
    IFS=':' read -r -a _files <<< "$TOKEN_FILES"
    for f in "${_files[@]}"; do
        [ -r "$f" ] || continue
        sed -n 's/.* pid \([0-9][0-9]*\).*/\1/p' "$f" | head -1
    done
}
ticketed() {
    local pids mine pid ppid depth=0
    pids=$(token_pids | tr '\n' ' ')
    [ -z "${pids// /}" ] && return 1
    pid=$$
    while [ -n "$pid" ] && [ "$pid" -gt 1 ] && [ "$depth" -lt 20 ]; do
        for mine in $pids; do [ "$pid" = "$mine" ] && return 0; done
        ppid=$(sed -n 's/^PPid:[[:space:]]*//p' "/proc/$pid/status" 2>/dev/null)
        case "$ppid" in ''|*[!0-9]*) return 1;; esac
        pid=$ppid
        depth=$((depth + 1))
    done
    return 1
}
if ticketed; then TICKETED=1; else TICKETED=0; fi
GATE1_ONLY=$(( 1 - TICKETED ))
if [ "$TICKETED" = 0 ]; then
    echo "oga_pack_probe: WARNING: no device ticket detected (no ancestor of this process is the pid in $TOKEN_FILES)," >&2
    echo "                so GATE 2 WILL NOT RUN. The invocation that runs both gates is:" >&2
    echo "                  npu-hold -t 4 -l \"oga probe\" -- $0 $PACK $PORT" >&2
    if [ "${REQUIRE_TICKET:-0}" = 1 ]; then
        echo "                REQUIRE_TICKET=1, so this is a refusal." >&2
        exit 2
    fi
fi

echo "== pack: $PACK"
echo "== bin:  $BIN"
mkdir -p "$SCRATCH" || { echo "oga_pack_probe: cannot write scratch dir $SCRATCH (set OUT= or SCRATCH=)" >&2; exit 2; }
AVAIL_K=$(df -Pk "$SCRATCH" | awk 'NR == 2 {print int($4/1048576)}')
[ "${AVAIL_K:-0}" -ge 2 ] || { echo "oga_pack_probe: $SCRATCH has ${AVAIL_K}G free; an imported container needs ~1-3G" >&2; exit 2; }
echo "== gate 1: import -> $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"
imp=$($BIN npu-pack import "$PACK" -o "$OUT" --allow-incomplete 2>&1)
rc=$?
printf '%s\n' "$imp" | grep -E 'fconst|attention|parameters|container|incomplete' | sed 's/^/   /'
if [ $rc -ne 0 ] || [ ! -f "$OUT/model.q4nx" ]; then
    echo "== verdict: import produced no container (rc=$rc) -- gate 1 is where this pack stops"
    exit 3
fi

if [ "$GATE1_ONLY" = 1 ]; then
    echo "== no ticket detected: stopping after gate 1 -- gate 2 touches the NPU and needs an npu-hold ticket."
    echo "== verdict: import OK (container at $OUT); the forward gate was not attempted"
    exit 3
fi

echo "== gate 2: forward init on the NPU (this needs the ticket you are in)"
"$BIN" forward-serve -m "$OUT/model.q4nx" -p "$PORT" > "$LOG" 2>&1 &
pid=$!
cleanup() { kill -9 "$pid" 2>/dev/null; [ "${KEEP:-0}" = 1 ] || rm -rf "$OUT"; }
trap 'cleanup; exit 130' INT TERM HUP
trap cleanup EXIT
for _ in $(seq 1 30); do
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$code" = 200 ] && break
    kill -0 $pid 2>/dev/null || break
    sleep 2
done
if [ "${code:-}" != 200 ]; then
    echo "   server log:"
    tail -8 "$LOG" | sed 's/^/     /'
    if grep -q 'dequant failed' "$LOG"; then
        echo "== verdict: forward init stopped on the quantized EMBEDDING (q4_1 in the vendor layout);"
        echo "            the forward's dequant_any() takes only our 5120-byte q4_1 tile, a Q4_K chunk or a Q8 chunk"
    elif grep -q 'missing .*ELF' "$LOG"; then
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
