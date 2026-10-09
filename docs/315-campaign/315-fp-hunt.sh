#!/usr/bin/env bash
# engine#315: repeat in-regime instrumented arms until one faults, so the diagnostic record exists
# for a faulting event.
#
# The fault rate in-regime is ~5%/arm, so a single flag-on arm is unlikely to fault (a clean arm is
# the norm and is still useful: it gives the normal record pattern). This driver runs arms
# sequentially, each in its own tag/dir so every log is kept, and stops at the first arm with a
# fault on either face. Re-composition happens per arm because 315-instr-capture.sh owns its regime;
# that costs a few minutes per arm and keeps each arm independently valid.
#
# Usage: 315-fp-hunt.sh <bin_dir> <arms> [tagbase]
set -uo pipefail
BIN=${1:?bin dir}; N=${2:-15}; TAGB=${3:-315on}
M=/home/bcloud/models/GLM-4.7-Flash-Q4_K_M.gguf
HERE=$(dirname "$0")
LOG=$HOME/wt/$TAGB-hunt.log

echo "=== hunt start $(date -Is) bin=$BIN arms=$N tagbase=$TAGB ===" | tee -a "$LOG"
for i in $(seq 1 "$N"); do
    TAG="$TAGB-$i"
    echo "=== hunt arm $i/$N tag=$TAG $(date -Is) ===" | tee -a "$LOG"
    bash "$HERE/315-instr-capture.sh" "$BIN" 1 2048 "$TAG" 1 2>&1 | tee -a "$LOG"
    O=$HOME/wt/rt-det/$TAG-$(basename "$M" .gguf)
    nan=$(grep -c "HRX returned NaN logits" "$O/server.log" 2>/dev/null || echo 0)
    hsa=$(grep -iE "HSA_STATUS_ERROR|Queue error|wait for HRX graph replay commands failed" \
          "$O/server.log" 2>/dev/null | grep -vcE "hrx_allocator_allocate_buffer|allocate HRX host staging buffer" || echo 0)
    utf8=$(python3 -c "import json;print(json.load(open('$O/detector.json'))['utf8_failures'])" 2>/dev/null || echo 0)
    parse=$(python3 -c "import json;print(json.load(open('$O/detector.json'))['parse_failures'])" 2>/dev/null || echo 0)
    fp=$(grep -c "\[hrx-wb-fp\]" "$O/server.log" 2>/dev/null || echo 0)
    gtt=$(grep -o "gtt_start [0-9]*" "$O/run.log" 2>/dev/null | head -1)
    echo "  arm $i verdict: $gtt nan=$nan hsa=$hsa utf8=$utf8 parse=$parse wb_records=$fp" | tee -a "$LOG"
    if [ "${nan:-0}" != "0" ] || [ "${hsa:-0}" != "0" ] || [ "${utf8:-0}" != "0" ] || [ "${parse:-0}" != "0" ]; then
        echo "FAULT CAUGHT on arm $i — stopping the hunt. Records: $O/server.log" | tee -a "$LOG"
        echo "315HUNT_FAULT arm=$i out=$O" | tee -a "$LOG"
        exit 0
    fi
    if [ "${fp:-0}" = "0" ]; then
        echo "WARNING: arm $i produced no [hrx-wb-fp] records — the diagnostic may not be reaching the" | tee -a "$LOG"
        echo "         server (check --setenv GGML_HRX_WRITEBACK_FINGERPRINT); continuing anyway." | tee -a "$LOG"
    fi
done
echo "315HUNT_DONE arms=$N no fault caught" | tee -a "$LOG"
