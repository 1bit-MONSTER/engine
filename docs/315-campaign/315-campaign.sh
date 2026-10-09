#!/usr/bin/env bash
# engine#315: acceptance-campaign launcher — waits for a free box, runs the pre-registered paired
# campaign (control vs fix cells), then applies the pre-registered analyser.
#
# Exists because the box is shared and its regime band is a single-occupancy resource: a campaign that
# starts into a polluted box would burn hours of arms that the analyser must then discard as
# out-of-regime. So the launcher waits for the machine itself instead of the agent polling it.
#
# Acceptance (docs/315-preregistration.md §7): >=60 in-regime clean arms in the fix cell, zero faults on
# either face; the analyser exits 0 only when that holds.
#
# Usage: CBIN=~/wt/<control>/bin FBIN=~/wt/<fix>/bin [MAXPAIRS=80] [MINCLEAN=60] bash 315-campaign.sh
set -uo pipefail
CBIN=${CBIN:?control bin dir}
FBIN=${FBIN:?fix bin dir}
MAXPAIRS=${MAXPAIRS:-80}
MINCLEAN=${MINCLEAN:-60}
TAG=${TAG:-315camp}
MAXWAIT=${MAXWAIT:-18000}
FLOOR=${FLOOR:-30}
G=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
gtt() { echo $(( $(cat "$G") / 1073741824 )); }
HERE=$(dirname "$0")

echo "=== waiting for a free box before the acceptance campaign (max ${MAXWAIT}s) ==="
waited=0; streak=0
while [ "$waited" -lt "$MAXWAIT" ]; do
    cur=$(gtt)
    peer=$(pgrep -cf "v315-next-fixed" 2>/dev/null); peer=${peer:-0}
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
    echo "BOXWAIT_TIMEOUT after ${waited}s — campaign not started (no out-of-regime arms were burned)"
    exit 3
fi

echo "=== box free (gtt=$(gtt)G) — paired campaign: cap $MAXPAIRS pairs, target $MINCLEAN clean fix arms ==="
MINCLEAN=$MINCLEAN PAIRS=$MAXPAIRS bash "$HERE/315-paired.sh" "$MAXPAIRS" 1 2048 "$CBIN" "$FBIN" "$TAG"

echo "=== pre-registered analysis ==="
python3 "$HERE/315-analyse.py" "$HOME/wt/$TAG-console.log"
rc=$?
echo "analyser exit=$rc (0 = acceptance met)"
echo "315CAMPAIGN_DONE rc=$rc"
exit $rc
