#!/bin/sh
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
# 1bit-board: fans and power profile on the boards this image knows.
#
# The firmware's own fan curve is conservative, so on a known board this sets the EC's fan curve plus a
# predictive boost (level 4 on load, back to the curve when quiet) and applies the power profile. All of it
# goes through the EC driver (ec_su_axb35), never through debug registers. On a board it does not know it
# writes nothing at all and says so: the firmware curve stays.
#
# Usage (root):  1bit-board known | curve | boost | status | power | power-status
#   known        exit 0 when this board's EC is one we drive (DMI match), 1 otherwise
#   curve        write the EC fan curve (mode=curve + ramp up/down) to fan1/fan2
#   boost        the 0.5 s predictive-boost loop (init starts it in the background)
#   status       one line: fans: controlled (fan1 N rpm, fan2 N rpm) / not controlled (reason)
#   power        apply the power profile (EC power mode, SMU limits if the tool is present, clock pins)
#   power-status one line: power: <mode>, <limits> / not applied (reason)
set -u

EC=${EC_DIR:-/sys/class/ec_su_axb35}
POWER_CONF=${POWER_CONF:-/etc/1bit/power}
UP=${FAN_UP:-45,54,62,70,76}          # level 1..5 engage at these EC temperatures (C); level 5 from 82 C
DOWN=${FAN_DOWN:-41,50,58,66,72}      # release points: 4 C hysteresis, so the curve does not hunt
FANS=${FAN_NAMES:-"fan1 fan2"}         # fan3 (system fan) is left alone: it reads 0 rpm on this board
BOOST_LEVEL=${BOOST_LEVEL:-4}         # 80 %: level 5 is never the boost level
BOOST_HOT5_C=${BOOST_HOT5_C:-82}      # above this Tctl the boost uses level 5 (100 %) and stays there to 76 C
BOOST_W=${BOOST_W:-35}                # package watts that count as load
BOOST_BUSY=${BOOST_BUSY:-40}          # GPU busy % that counts as load
BOOST_HOT_C=${BOOST_HOT_C:-70}        # Tctl that keeps the boost on regardless of power
BOOST_SLOPE=${BOOST_SLOPE:-3}         # Tctl rise per poll that counts as load
BOOST_HOLD=${BOOST_HOLD:-20}          # quiet seconds before releasing to the curve
POWER_MODE=${POWER_MODE:-performance} # EC power mode: quiet | balanced | performance
STAPM_W=${STAPM_W:-90}                # 90 W sustained, the profile the development box runs
SLOW_W=${SLOW_W:-90}
FAST_W=${FAST_W:-105}
TCTL_C=${TCTL_C:-95}
FCLK=${FCLK:-7}                       # amdgpu pp_dpm_fclk index, "auto" to leave it alone
SOCCLK=${SOCCLK:-6}                   # amdgpu pp_dpm_socclk index
SKIP_POWER=${SKIP_POWER:-0}
FALLBACK_MODE=${FALLBACK_MODE:-balanced}  # EC mode when the SMU limits cannot be written
RYZENADJ=${RYZENADJ:-}

say() { printf '%s\n' "$*"; }
die() { printf '1bit-board: %s\n' "$*" >&2; exit 1; }
# BusyBox sleep may not do fractions; a one-second poll still beats the firmware curve.
nap() { sleep 0.5 2>/dev/null || sleep 1; }
hwmon() { for h in /sys/class/hwmon/*; do [ "$(cat "$h/name" 2>/dev/null)" = "$1" ] && echo "$h"; done; }

dmi() { cat "/sys/class/dmi/id/$1" 2>/dev/null | tr '[:upper:]' '[:lower:]'; }
known_board() {
    # Sixunited AXB35-02, sold as BOSGAME M5 "BeyondMax Series": the one board whose EC we drive today.
    case "$(dmi product_name) $(dmi board_name)" in
        *beyondmax* | *axb35*) return 0 ;;
        *) return 1 ;;
    esac
}

ec_ready() {   # the driver loaded and the interface we need is there
    [ -d "$EC" ] && [ -r "$EC/temp1/temp" ] || return 1
    for f in $FANS; do [ -w "$EC/$f/mode" ] || return 1; done
    return 0
}

curve() {
    ec_ready || die "no EC driver at $EC (nothing written)"
    known_board || die "not a known board (nothing written)"
    for f in $FANS; do
        printf '%s\n' "$UP" > "$EC/$f/rampup_curve" || die "cannot write $EC/$f/rampup_curve"
        printf '%s\n' "$DOWN" > "$EC/$f/rampdown_curve" || die "cannot write $EC/$f/rampdown_curve"
        printf '%s\n' curve > "$EC/$f/mode" || die "cannot write $EC/$f/mode"
    done
    say "fans: curve $UP / $DOWN on $FANS (the EC applies it on its own sensor)"
}

set_fans() {   # set_fans fixed|curve [level]
    for f in $FANS; do
        printf '%s\n' "$1" > "$EC/$f/mode" 2>/dev/null
        [ "${2:-}" ] && printf '%s\n' "$2" > "$EC/$f/level" 2>/dev/null
    done
}

boost() {
    ec_ready || die "no EC driver at $EC (boost not started)"
    known_board || die "not a known board (boost not started)"
    K=$(hwmon k10temp); G=$(hwmon amdgpu); B=$(ls /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -n 1)
    [ -n "$K" ] && [ -n "$G" ] || die "no k10temp/amdgpu hwmon (boost not started)"
    set_fans curve
    on=0; quiet=$(date +%s); last=$(( $(cat "$K/temp1_input" 2>/dev/null || echo 0) / 1000 )); lvl=$BOOST_LEVEL
    trap 'set_fans curve; exit 0' TERM INT
    say "fans: boost loop up (level $BOOST_LEVEL on load, $BOOST_HOLD s hold, ${BOOST_HOT5_C} C -> level 5)"
    while :; do
        tc=$(( $(cat "$K/temp1_input" 2>/dev/null || echo 0) / 1000 ))
        w=$(( $(cat "$G/power1_input" 2>/dev/null || echo 0) / 1000000 ))
        busy=$(cat "$B" 2>/dev/null || echo 0)
        now=$(date +%s); load=0
        if [ "$w" -ge "$BOOST_W" ] || [ "$busy" -ge "$BOOST_BUSY" ] || [ "$tc" -ge "$BOOST_HOT_C" ] \
           || [ $(( tc - last )) -ge "$BOOST_SLOPE" ]; then load=1; fi
        last=$tc
        want=$BOOST_LEVEL
        if [ "$tc" -ge "$BOOST_HOT5_C" ]; then want=5
        elif [ "$lvl" = 5 ] && [ "$tc" -gt $(( BOOST_HOT5_C - 6 )) ]; then want=5; fi
        if [ "$load" = 1 ]; then
            quiet=$now
            if [ "$on" = 0 ]; then lvl=$want; set_fans fixed "$lvl"; on=1
                say "fans: boost ${w}W busy=${busy}% Tctl=${tc}C level $lvl"
            elif [ "$want" != "$lvl" ]; then lvl=$want; set_fans fixed "$lvl"; say "fans: level $lvl at Tctl=${tc}C"
            fi
            # something else (the curve, a manual test) may have moved the mode: re-assert it
            [ "$(cat "$EC/fan1/mode" 2>/dev/null)" = fixed ] && \
            [ "$(cat "$EC/fan1/level" 2>/dev/null)" = "$lvl" ] || set_fans fixed "$lvl"
        elif [ "$on" = 1 ] && [ $(( now - quiet )) -ge "$BOOST_HOLD" ]; then
            set_fans curve; on=0; say "fans: back to the curve (${w}W Tctl=${tc}C)"
        fi
        nap
    done
}

status() {   # one line, for the banner, the health output and the release check
    if ! known_board; then
        say "fans: not controlled (this board is not one we know; the firmware curve stays)"
        return 0
    fi
    if ! ec_ready; then
        say "fans: not controlled (no EC driver: the firmware curve stays)"
        return 1
    fi
    line=""
    for f in $FANS; do
        rpm=$(cat "$EC/$f/rpm" 2>/dev/null || echo 0)
        line="$line $f $rpm rpm"
    done
    # informational: at low temperatures the EC legitimately leaves the fans stopped (level 0), so a
    # 0 rpm reading here is not a fault. The release check forces level 4 and then expects rpm > 0.
    say "fans: controlled ($(printf '%s' "$line" | sed 's/^ //'))"
    return 0
}

ryzenadj_bin() {
    [ -n "$RYZENADJ" ] && { [ -x "$RYZENADJ" ] && echo "$RYZENADJ"; return 0; }
    for p in /opt/1bit/ryzenadj /usr/local/bin/ryzenadj /usr/bin/ryzenadj; do
        [ -x "$p" ] && { echo "$p"; return 0; }
    done
    command -v ryzenadj 2>/dev/null && return 0
    return 1
}

power_conf() {   # read KEY=value overrides from $POWER_CONF without executing it
    [ -r "$POWER_CONF" ] || return 0
    while IFS= read -r line; do
        case "$line" in ''|'#'*) continue ;; esac
        key=${line%%=*}; val=${line#*=}
        key=$(printf '%s' "$key" | tr -d ' \t')
        val=$(printf '%s' "$val" | sed 's/[[:space:]]*#.*$//; s/^[[:space:]]*//; s/[[:space:]]*$//')
        case "$key" in
            POWER_MODE|STAPM_W|SLOW_W|FAST_W|TCTL_C|FCLK|SOCCLK|SKIP_POWER|RYZENADJ|FAN_UP|FAN_DOWN)
                eval "CONF_$key=\$val" ;;
        esac
    done < "$POWER_CONF"
    for k in POWER_MODE STAPM_W SLOW_W FAST_W TCTL_C FCLK SOCCLK SKIP_POWER RYZENADJ FALLBACK_MODE; do
        eval "v=\${CONF_$k:-}"; [ -n "$v" ] && eval "$k=\$v"
    done
    v=${CONF_FAN_HOT5_C:-}; [ -n "$v" ] && BOOST_HOT5_C=$v
    return 0
}

power() {
    power_conf
    [ "$SKIP_POWER" = 1 ] && { say "power: not applied (SKIP_POWER=1 in $POWER_CONF)"; return 0; }
    notes=""
    if [ -w "$EC/apu/power_mode" ]; then
        # the EC power mode resets the SMU limits, so it goes first - same order as the development box
        printf '%s\n' "$POWER_MODE" > "$EC/apu/power_mode" 2>/dev/null && notes="$notes mode=$POWER_MODE" \
            || notes="$notes mode=write-failed"
        sleep 2
    else
        notes="$notes mode=no-ec"
    fi
    cap_ok=0
    if [ -n "$RYZENADJ" ] && [ ! -x "$RYZENADJ" ]; then
        notes="$notes cap=failed(RYZENADJ=$RYZENADJ is not executable)"
    elif RJ=$(ryzenadj_bin); then
        if "$RJ" --stapm-limit=$(( STAPM_W * 1000 )) --slow-limit=$(( SLOW_W * 1000 )) \
                 --fast-limit=$(( FAST_W * 1000 )) --tctl-temp="$TCTL_C" >/dev/null 2>&1; then
            notes="$notes cap=${STAPM_W}/${SLOW_W}/${FAST_W}W tctl=$TCTL_C"; cap_ok=1
            # what power_status reports: the write is the fact, `ryzenadj --info` is not a test of it
            # (on the image's kernel, STRICT_DEVMEM lets ryzenadj write the SMU limits and refuses the
            # pm-table read that --info needs, so --info fails after a successful cap - measured on the
            # candidate 8 stick with Secure Boot on)
            printf '%s/%s/%s\n' "$STAPM_W" "$SLOW_W" "$FAST_W" > /run/1bit-smu-cap 2>/dev/null || true
        else
            notes="$notes cap=failed($RJ)"
        fi
    else
        notes="$notes cap=not-applied(no ryzenadj)"
    fi
    if [ "$cap_ok" = 0 ]; then
        # Secure Boot lockdown, no /dev/mem access or no tool: the SMU limits cannot be written, so fall back
        # to the EC's own balanced profile (85 W sustained, 120 W burst). Never leave the board in performance
        # mode (its ~120 W profile) with no cap at all.
        if printf '%s\n' "$FALLBACK_MODE" > "$EC/apu/power_mode" 2>/dev/null; then
            notes="$notes fallback=$FALLBACK_MODE (the EC profile alone is in force)"
        else
            notes="$notes fallback=unavailable"
        fi
    fi
    G=$(ls -d /sys/class/drm/card*/device 2>/dev/null | head -n 1)
    if [ -n "$G" ] && [ -w "$G/power_dpm_force_performance_level" ]; then
        if [ "$FCLK" = auto ]; then
            printf 'auto\n' > "$G/power_dpm_force_performance_level" 2>/dev/null && notes="$notes clocks=auto"
        else
            printf 'manual\n' > "$G/power_dpm_force_performance_level" 2>/dev/null
            printf '%s\n' "$FCLK" > "$G/pp_dpm_fclk" 2>/dev/null
            printf '%s\n' "$SOCCLK" > "$G/pp_dpm_socclk" 2>/dev/null
            notes="$notes fclk=$FCLK socclk=$SOCCLK"
        fi
    fi
    say "power:$notes"
}

power_status() {
    mode=$(cat "$EC/apu/power_mode" 2>/dev/null || echo none)
    if [ -s /run/1bit-smu-cap ]; then
        say "power: $mode mode, SMU cap $(cat /run/1bit-smu-cap) W (sustained/slow/fast) applied this boot"
    elif RJ=$(ryzenadj_bin) && info=$("$RJ" --info 2>/dev/null); then
        lim=$(printf '%s\n' "$info" | sed -n 's/^STAPM LIMIT *| *\([0-9.]*\)\..*/\1/p' | head -n 1)
        say "power: $mode mode, SMU cap ${lim:-?} W sustained (read back)"
    elif RJ=$(ryzenadj_bin); then
        say "power: $mode mode, EC profile only: $RJ is present but the SMU cap was not applied this boot (see the [power] apply line)"
    else
        say "power: $mode mode, EC profile only: no ryzenadj in this image"
    fi
}

case "${1:-}" in
    known)        known_board && exit 0 || exit 1 ;;
    curve)        curve ;;
    boost)        boost ;;
    status)       status ;;
    power)        power ;;
    power-status) power_status ;;
    *)            say "usage: 1bit-board known|curve|boost|status|power|power-status" >&2; exit 2 ;;
esac
