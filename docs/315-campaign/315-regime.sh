#!/usr/bin/env bash
# engine#315 shared regime helper — adaptive pressure composition with hard preflight.
#
# Two failures this helper exists to prevent, both observed on the box:
#
#  1. A neighbour started without IREE_HAL_AMDGPU_LIBHSA_PATH dies immediately with
#     "error while handling argument \"-dev\": invalid device: HRX0" — because llama-server then
#     reports "Available devices: (none)". Neighbours are therefore started with the HRX env
#     explicitly, and regime_preflight refuses to proceed if HRX0 is not visible at all.
#  2. A fixed neighbour count misses the regime band (3 default neighbours -> 115.8 GiB, detector
#     dies allocating; 2 at -c 2048 -> 80.3 GiB, out of regime), because the box's base GTT moves
#     with other sessions. Neighbours are therefore started one at a time against live GTT.
#
# Callers must set BIN, M, TAG and IREE_HAL_AMDGPU_LIBHSA_PATH, then:
#     regime_preflight || exit 1
#     regime_neighbours 1 6 2048 || exit 1        # non-zero = no usable composition
#     trap regime_teardown EXIT

regime_gtt() { echo $(( $(cat "$REGIME_G_FILE") / 1073741824 )); }

regime_preflight() {
    REGIME_G_FILE=${REGIME_G_FILE:-$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)}
    : "${REGIME_G_FILE:?no GTT sysfs node found}"
    if [ -z "${IREE_HAL_AMDGPU_LIBHSA_PATH:-}" ]; then
        echo "regime: ABORT — IREE_HAL_AMDGPU_LIBHSA_PATH is unset; llama-server will see no devices"
        return 1
    fi
    local devs
    devs=$(IREE_HAL_AMDGPU_LIBHSA_PATH="$IREE_HAL_AMDGPU_LIBHSA_PATH" LD_LIBRARY_PATH="$BIN" \
           "$BIN/llama-server" --list-devices 2>&1 | head -8)
    case "$devs" in
        *HRX0*)
            echo "regime: preflight ok — $(printf '%s' "$devs" | grep -m1 HRX0 | sed 's/^ *//' | cut -c1-72)"
            ;;
        *)
            echo "regime: ABORT — HRX0 not visible to $BIN/llama-server; --list-devices said:"
            printf '%s\n' "$devs" | tail -3 | sed 's/^/    /'
            return 1
            ;;
    esac
}

# Start neighbours until the live GTT reaches LO (default 80 GiB), never past HI (default 92 GiB).
# Each start is bounded: if the neighbour does not serve /health in REGIME_UP_TIMEOUT seconds, or
# the process dies, it is recorded as a failed start, and two consecutive failures abort the
# composition instead of grinding through further multi-minute waits.
regime_neighbours() {
    local min_nb=${1:-1} max_nb=${2:-6} ctx=${3:-2048}
    local up_timeout=${REGIME_UP_TIMEOUT:-150}
    REGIME_NB=0; local failures=0
    REGIME_LO=${REGIME_LO:-80}; REGIME_HI=${REGIME_HI:-92}
    while [ "$REGIME_NB" -lt "$max_nb" ]; do
        local now; now=$(regime_gtt)
        if [ "$REGIME_NB" -ge "$min_nb" ] && [ "$now" -ge "$REGIME_LO" ]; then break; fi
        if [ "$now" -ge "$REGIME_HI" ]; then
            echo "regime: gtt=$now GiB at/above HI=$REGIME_HI, no further neighbours"
            break
        fi
        REGIME_NB=$((REGIME_NB + 1))
        local port=$((20250 + REGIME_NB))
        systemd-run --user --scope -q -p MemoryMax=40G --unit=${TAG}-neigh-$REGIME_NB \
            --setenv=IREE_HAL_AMDGPU_LIBHSA_PATH="$IREE_HAL_AMDGPU_LIBHSA_PATH" \
            --setenv=LD_LIBRARY_PATH="$BIN" \
            "$BIN/llama-server" -m "$M" -dev HRX0 -ngl 99 -fa on -c "$ctx" \
            --host 127.0.0.1 --port "$port" \
            > "$HOME/wt/$TAG-neigh-$REGIME_NB.log" 2>&1 &
        local up=0 t=0
        while [ "$t" -lt "$up_timeout" ]; do
            curl -sf "localhost:$port/health" >/dev/null 2>&1 && { up=1; break; }
            if ! systemctl --user is-active "${TAG}-neigh-$REGIME_NB" >/dev/null 2>&1 \
               && ! pgrep -f "port $port" >/dev/null 2>&1; then
                break                      # process is gone: no point waiting out the timeout
            fi
            sleep 2; t=$((t + 2))
        done
        if [ "$up" != "1" ]; then
            failures=$((failures + 1))
            echo "regime: neighbour $REGIME_NB FAILED to come up within ${up_timeout}s — last log lines:"
            tail -3 "$HOME/wt/$TAG-neigh-$REGIME_NB.log" 2>/dev/null | sed 's/^/    /'
            REGIME_NB=$((REGIME_NB - 1))
            systemctl --user stop "${TAG}-neigh-$((REGIME_NB + 1))" 2>/dev/null
            [ "$failures" -ge 2 ] && { echo "regime: ABORT after 2 failed neighbour starts"; return 1; }
            continue
        fi
        sleep 4
        echo "regime: neighbour $REGIME_NB up, gtt=$(regime_gtt)G"
    done
    if [ "$REGIME_NB" -lt 1 ]; then echo "regime: ABORT — no neighbour could be started"; return 1; fi
    echo "regime: composition neighbours=$REGIME_NB ctx=$ctx gtt_before_arm=$(regime_gtt)G (LO=$REGIME_LO HI=$REGIME_HI, projects to >=95 after the detector loads)"
    return 0
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
