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
BIN=${BIN:-${ONEBIT_BIN:-}}
# Scratch goes to a DISK-backed dir, never /tmp: an imported container is hundreds of MB to several GB, and this box's tmpfs hit
# 0 MB free on 2026-11-08 and broke every lane's tool output. KEEP=1 leaves it behind, otherwise it is removed on exit.
SCRATCH=${SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/oga-probe}
OUT=${OUT:-$SCRATCH/$(basename "$PACK")}
KEEP=${KEEP:-0}
LOG=$OUT/serve.log

[ -d "$PACK" ] || { echo "oga_pack_probe: no such pack dir: $PACK" >&2; exit 2; }

# Gate 2 touches the NPU, so this script is only correct inside a device ticket. Walk the ancestors and refuse otherwise -- a
# probe that ran unticketed is the shape of every collision this fleet has had. ALLOW_UNTICKETED=1 is the documented escape for a
# gate-1-only run on a box where no ticket is available.
ticketed() {
    # PPid comes from /proc/<pid>/status, not from field 4 of /proc/<pid>/stat: a comm containing a space or a parenthesis
    # shifts stat's fields and silently walks the wrong chain. Depth-capped so a malformed chain cannot loop.
    local pid=$$ argv0 ppid depth=0
    while [ -n "$pid" ] && [ "$pid" -gt 1 ] && [ "$depth" -lt 20 ]; do
        # argv[0] only, never the whole command line: a shell whose command *text* mentions npu-hold (a patch script, an editor,
        # a grep) made the first version of this walk match its own search string and wave an unticketed run through.
        argv0=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | head -1)
        [ "${DEBUG_TICKET:-0}" = 1 ] && echo "   ticket-walk: pid=$pid argv0=$argv0" >&2
        case "$argv0" in
            npu-hold|gpu-hold|*/npu-hold|*/gpu-hold)
                [ "${DEBUG_TICKET:-0}" = 1 ] && echo "   ticket-walk MATCH pid=$pid: $argv0" >&2
                return 0;;
        esac
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
    # Known limitation, stated rather than papered over: this fleet's harness (`npu-hold`) EXECs the command it is given, so the
    # hold process is not in an ancestor's argv[0] by the time this script runs and the walk above cannot see it. A first version
    # matched the whole command line instead, which made a shell whose *text* mentioned npu-hold (a patch script, an editor, a
    # grep) wave an unticketed run through -- a guard that passes on its own search string. So the default is a loud warning plus
    # the exact command, and REQUIRE_TICKET=1 turns it into a refusal for a provisioning script that knows better.
    echo "oga_pack_probe: WARNING: no npu-hold/gpu-hold ticket detected in this process tree, so GATE 2 WILL NOT RUN." >&2
    echo "                the invocation that runs both gates: npu-hold -t 4 -l \"oga probe\" -- $0 $PACK $PORT" >&2
    echo "                (REQUIRE_TICKET=1 refuses outright instead of running gate 1 only)" >&2
    if [ "${REQUIRE_TICKET:-0}" = 1 ]; then
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
