#!/usr/bin/env bash
# engine#315: wait for a free box, then run the alternating differential that decides the mechanism.
#
# Two questions, one campaign:
#   * does an in-regime arm fault at all (task-2 wants the instrumentation record of a faulting arm)?
#   * does coherence remove the fault (the causal half of task-3: the structural half is already
#     confirmed — GGML_HRX_USE_UNIFIED_MEMORY flips HOST_COHERENT on exactly the logits buffer).
#
# So arms alternate: A = default configuration, B = same binary/regime with
# GGML_HRX_USE_UNIFIED_MEMORY=1. Paired arms in one session share the box state, so the comparison is
# not confounded by drift between sessions.
#
# The mechanism signature is "A faulted, B clean". A is expected to fault at roughly the recorded rate
# (~5%/arm), so a handful of pairs is evidence, not a rate; the acceptance campaign (>=60 clean arms) is
# a separate, longer run.
#
# Usage: PAIRS=6 MAXWAIT=18000 bash 315-differential.sh
set -uo pipefail
MAXWAIT=${MAXWAIT:-18000}
PAIRS=${PAIRS:-6}
TAGB=${TAGB:-315diff}
BIN=${BIN:-/home/bcloud/wt/315instr3-build/bin}
G=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
gtt() { echo $(( $(cat "$G") / 1073741824 )); }
FLOOR=${FLOOR:-30}
LOG=$HOME/wt/$TAGB.log

arm() {   # arm <tag> <extra-env>
    local tag=$1 extra=${2:-}
    echo "--- arm $tag extra_env='${extra:-none}' $(date -Is)" | tee -a "$LOG"
    if [ -n "$extra" ]; then
        EXTRA_ENV="$extra" NOLOCK=0 bash "$HOME/wt/315-instr-capture.sh" "$BIN" 1 2048 "$tag" 1 2>&1 | tail -14 | tee -a "$LOG"
    else
        NOLOCK=0 bash "$HOME/wt/315-instr-capture.sh" "$BIN" 1 2048 "$tag" 1 2>&1 | tail -14 | tee -a "$LOG"
    fi
    local O="$HOME/wt/rt-det/$tag-$(basename /home/bcloud/models/GLM-4.7-Flash-Q4_K_M.gguf .gguf)"
    local nan hsa utf8 parse req g0 fp
    nan=$(grep -c "HRX returned NaN logits" "$O/server.log" 2>/dev/null || echo 0)
    hsa=$(grep -iE "HSA_STATUS_ERROR|Queue error|wait for HRX graph replay commands failed" "$O/server.log" 2>/dev/null \
          | grep -vcE "hrx_allocator_allocate_buffer|allocate HRX host staging buffer" || echo 0)
    utf8=$(python3 -c "import json;print(json.load(open('$O/detector.json'))['utf8_failures'])" 2>/dev/null || echo 0)
    parse=$(python3 -c "import json;print(json.load(open('$O/detector.json'))['parse_failures'])" 2>/dev/null || echo 0)
    req=$(python3 -c "import json;print(json.load(open('$O/detector.json'))['requests'])" 2>/dev/null || echo 0)
    g0=$(grep -o "gtt_start [0-9]*G" "$O/run.log" 2>/dev/null | head -1)
    fp=$(grep -c "\[hrx-buf\]" "$O/server.log" 2>/dev/null || echo 0)
    local verdict=clean
    { [ "${nan:-0}" != "0" ] || [ "${hsa:-0}" != "0" ] || [ "${utf8:-0}" != "0" ] || [ "${parse:-0}" != "0" ]; } && verdict=FAULT
    [ "${req:-0}" -lt 66 ] && [ "$verdict" = clean ] && verdict=INCOMPLETE
    echo "  RESULT tag=$tag $g0 verdict=$verdict nan=$nan hsa=$hsa utf8=$utf8 parse=$parse req=$req buf_trace=$fp" | tee -a "$LOG"
    echo "$verdict" > "$HOME/wt/$tag.verdict"
}

echo "=== waiting for a free box (floor=${FLOOR} GiB, max ${MAXWAIT}s, at $(date -Is)) ===" | tee -a "$LOG"
waited=0; streak=0
while [ "$waited" -lt "$MAXWAIT" ]; do
    cur=$(gtt); peer=$(pgrep -cf "v315-next-fixed" 2>/dev/null || echo 0)
    avail=$(free -g | awk '/^Mem:/ {print $7}')
    if [ "$cur" -lt "$FLOOR" ] && [ "${peer:-0}" = "0" ] && [ "${avail:-0}" -ge 25 ]; then streak=$((streak+1)); else streak=0; fi
    echo "$(date +%H:%M:%S) gtt=${cur}G peer=${peer:-0} avail=${avail:-?}G streak=$streak" | tee -a "$LOG"
    [ "$streak" -ge 2 ] && break
    sleep 60; waited=$((waited+60))
done
[ "$streak" -lt 2 ] && { echo "BOXWAIT_TIMEOUT after ${waited}s" | tee -a "$LOG"; exit 3; }

echo "=== BOX FREE (gtt=$(gtt)G) — alternating differential, $PAIRS pairs ===" | tee -a "$LOG"
for i in $(seq 1 "$PAIRS"); do
    arm "${TAGB}-d$i" ""
    arm "${TAGB}-u$i" "GGML_HRX_USE_UNIFIED_MEMORY=1"
    a=$(cat "$HOME/wt/${TAGB}-d$i.verdict" 2>/dev/null || echo "?")
    b=$(cat "$HOME/wt/${TAGB}-u$i.verdict" 2>/dev/null || echo "?")
    echo "  PAIR $i: default=$a unified=$b" | tee -a "$LOG"
    if [ "$a" = "FAULT" ] && [ "$b" = "clean" ]; then
        echo "MECHANISM SIGNATURE at pair $i: default faulted while coherent arm was clean" | tee -a "$LOG"
        break
    fi
done
echo "315DIFF_DONE $(date -Is)" | tee -a "$LOG"
tail -20 "$LOG"
