#!/usr/bin/env bash
# engine#315: wait for the peer session to release the box, then run the instrumented fault hunt.
#
# Why this exists: the peer #315 session (~/wt/v315-next-fixed.sh) holds the in-regime band for hours at a
# time (measured: GTT 108 GiB with 7 GB RAM free, so a detector server cannot even load, let alone compose
# a regime). Polling that from an agent costs context and turns; instead this waits on the box itself and
# starts the hunt the moment the machine is genuinely free.
#
# Free means: GTT below a floor AND no peer process, sustained for two consecutive checks (so a gap
# between the peer's arms is not mistaken for the box being free).
#
# Usage: MAXWAIT=18000 HUNT_ARMS=12 bash 315-when-free.sh
set -uo pipefail
MAXWAIT=${MAXWAIT:-18000}
HUNT_ARMS=${HUNT_ARMS:-12}
TAGB=${TAGB:-315v3hunt}
BIN=${BIN:-/home/bcloud/wt/315instr3-build/bin}
G=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
gtt() { echo $(( $(cat "$G") / 1073741824 )); }
FLOOR=${FLOOR:-30}

echo "=== waiting for a free box (floor=${FLOOR} GiB, max ${MAXWAIT}s, at $(date -Is)) ==="
waited=0; streak=0
while [ "$waited" -lt "$MAXWAIT" ]; do
    cur=$(gtt)
    peer=$(pgrep -cf "v315-next-fixed" 2>/dev/null || echo 0)
    avail=$(free -g | awk '/^Mem:/ {print $7}')
    if [ "$cur" -lt "$FLOOR" ] && [ "${peer:-0}" = "0" ] && [ "${avail:-0}" -ge 25 ]; then
        streak=$((streak + 1))
    else
        streak=0
    fi
    echo "$(date +%H:%M:%S) gtt=${cur}G peer=${peer:-0} avail=${avail:-?}G streak=$streak waited=${waited}s"
    [ "$streak" -ge 2 ] && break
    sleep 60
    waited=$((waited + 60))
done

if [ "$streak" -lt 2 ]; then
    echo "BOXWAIT_TIMEOUT after ${waited}s — box never became free; hunt not started"
    exit 3
fi

echo "=== BOX FREE (gtt=$(gtt)G avail=$(free -g | awk '/^Mem:/ {print $7}')G) — starting the hunt ==="
# EXTRA_ENV (e.g. GGML_HRX_BUFFER_TRACE=1) is inherited by 315-fp-hunt.sh -> 315-instr-capture.sh.
HUNT_ARMS=$HUNT_ARMS bash /home/bcloud/wt/315-fp-hunt.sh "$BIN" "$HUNT_ARMS" "$TAGB"
echo "315WHENFREE_DONE"
