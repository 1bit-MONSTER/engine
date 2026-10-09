#!/usr/bin/env bash
# engine#315 shared regime helper — adaptive pressure composition.
#
# The regime is defined by total GTT (gate: gtt_start >= 95 GiB, invalid above ~110), not by a
# neighbour count. A fixed count missed the band twice: 3 default neighbours overshot to 115.8 GiB
# (detector died allocating) and 2 small-context neighbours undershot to 88 GiB — both because the
# box's base GTT moves with other sessions' servers (another #315 session was queued on the box lock
# during the second attempt). So start neighbours one at a time until the *projected* arm start lands
# in band.
#
# Callers must:
#   - set M (model), BIN (binary dir), TAG
#   - export the box lock discipline themselves (rt-det.sh takes it; this helper does not)
#   - call regime_neighbours <min> <max> <ctx>
#   - call regime_teardown on exit (a trap is installed here, but calling it explicitly is fine)
#
# Projection: the detector server itself adds ~15 GiB once loaded, so neighbours are started until
# gtt_now reaches LO (default 80 GiB) and never beyond HI (default 92 GiB), which projects to
# ~95-107 GiB gtt_start.

regime_gtt() { echo $(( $(cat "$REGIME_G_FILE") / 1073741824 )); }

regime_neighbours() {
    local min_nb=${1:-1} max_nb=${2:-6} ctx=${3:-2048}
    REGIME_NB=0
    REGIME_G_FILE=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
    REGIME_LO=${REGIME_LO:-80}; REGIME_HI=${REGIME_HI:-92}
    while [ "$REGIME_NB" -lt "$max_nb" ]; do
        local now; now=$(regime_gtt)
        if [ "$REGIME_NB" -ge "$min_nb" ] && [ "$now" -ge "$REGIME_LO" ]; then
            break
        fi
        if [ "$now" -ge "$REGIME_HI" ]; then
            echo "regime: gtt=$now GiB already at/above HI=$REGIME_HI, starting no further neighbours" >&2
            break
        fi
        REGIME_NB=$((REGIME_NB + 1))
        systemd-run --user --scope -q -p MemoryMax=40G --unit=${TAG}-neigh-$REGIME_NB \
            "$BIN/llama-server" -m "$M" -dev HRX0 -ngl 99 -fa on -c "$ctx" \
            --host 127.0.0.1 --port $((20250 + REGIME_NB)) \
            > "$HOME/wt/$TAG-neigh-$REGIME_NB.log" 2>&1 &
        for _t in $(seq 1 300); do
            curl -sf "localhost:$((20250 + REGIME_NB))/health" >/dev/null 2>&1 && break
            sleep 1
        done
        sleep 4   # let GTT settle before deciding whether another neighbour is needed
        echo "regime: neighbour $REGIME_NB up, gtt=$(regime_gtt)G"
    done
    echo "regime: composition neighbours=$REGIME_NB ctx=$ctx gtt_before_arm=$(regime_gtt)G (band LO=$REGIME_LO HI=$REGIME_HI, projects to >=95 after the detector loads)"
}

regime_teardown() {
    local i
    for i in $(seq 1 "${REGIME_NB:-0}"); do
        systemctl --user stop "${TAG}-neigh-$i" 2>/dev/null
        systemctl --user kill -s TERM "${TAG}-neigh-$i" 2>/dev/null
    done
    sleep 3
    for P in $(pgrep -f "llama-server" 2>/dev/null); do
        local C; C=$(tr '\0' ' ' < "/proc/$P/cmdline" 2>/dev/null)
        case "$C" in *"port 2025"*) kill -TERM "$P" 2>/dev/null ;; esac
        if [ -n "${REGIME_ARM_PORT:-}" ]; then
            case "$C" in *"port $REGIME_ARM_PORT"*) kill -TERM "$P" 2>/dev/null ;; esac
        fi
    done
    sleep 5
    echo "regime: teardown done servers_left=$(pgrep -c llama-server || echo 0) gtt_after=$(regime_gtt)G"
}
