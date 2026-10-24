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
# tui.sh - the one renderer behind the install screen and the live ready screen.
#
# POSIX sh + ANSI escapes only: no curses, no dialog, no Python, no bash arrays (os/mini is BusyBox).
# Sourced by install.sh and init.sh. Full-screen console UI: our logo animated on a frame tick, a step
# checklist with live state, one progress bar that belongs to the running step (percent, MB/s, ETA), a
# fixed six-line log tail, and a red failure state naming the step, the command and the log.
#
# stdout not a tty (the vmtest harness, a serial console, a log file) selects the plain path: the same
# information as line-oriented text, so every assertion that reads the plain output keeps working.
#
# The frames are generated from our own mark by tools/gen-logo-frames.sh; regenerate rather than edit.
set -u

E=$(printf '\033')
TUI_PLAIN=0
TUI_W=80
TUI_H=24
TUI_FRAME=0
TUI_KEYS=""
TUI_CUR=
TUI_ERRCMD=
TUI_LOGPATH=${TUI_LOGPATH:-/data/install.log}

TUI_LOGO_1='        ▄▄▄▄▄▄▄        '
TUI_LOGO_2='      ▄█▀     ▀█▄      '
TUI_LOGO_3='     ██   ▄▄▄   ██     '
TUI_LOGO_4='     ██   ███   ██     '
TUI_LOGO_5='     ██   ▀▀▀   ██     '
TUI_LOGO_6='      ▀█▄     ▄█▀      '
TUI_LOGO_7='        ▀▀▀▀▀▀▀        '
TUI_LOGO_LINES=7

tui_steps() {   # tui_steps "key:label" ...
    for s in "$@"; do
        k=${s%%:*}
        l=${s#*:}
        TUI_KEYS="$TUI_KEYS $k"
        eval "TUI_L_$k=\$l"
        eval "TUI_S_$k=pending"
        eval "TUI_P_$k=0"
        eval "TUI_RATE_$k="
        eval "TUI_ETA_$k="
    done
}

tui_init() {   # tui_init <title> <subtitle> <target>
    TUI_TITLE=${1:-1bit OS install}
    TUI_SUB=${2:-1bit OS, Lemonade Server appliance}
    TUI_TARGET=${3:-}
    TUI_START=$(date +%s)
    TUI_LOG2=; TUI_LOG3=; TUI_LOG4=; TUI_LOG5=; TUI_LOG6=
    if [ ! -t 1 ]; then
        TUI_PLAIN=1
        return 0
    fi
    sz=$(stty size 2>/dev/null || true)
    if [ -n "${sz:-}" ]; then
        TUI_H=${sz% *}
        TUI_W=${sz#* }
    fi
    [ "${TUI_H:-0}" -lt 24 ] 2>/dev/null && TUI_H=24
    [ "${TUI_W:-0}" -lt 80 ] 2>/dev/null && TUI_W=80
    printf '\033[H\033[2J\033[?25l'
    trap 'tui_end; exit 130' INT TERM
}

tui_end() {
    [ "$TUI_PLAIN" = 1 ] && return 0
    printf '\033[?25h\n'
}

tui_elapsed() { echo $(( $(date +%s) - ${TUI_START:-$(date +%s)} )); }

tui_hm() { printf '%d:%02d' $(( $1 / 60 )) $(( $1 % 60 )); }

tui_log() {   # tui_log <line>: real command output, kept to the last six lines
    line=$(printf '%s' "$*" | tr -d '\r' | cut -c1-${TUI_W})
    if [ "$TUI_PLAIN" = 1 ]; then
        printf '%s\n' "$line"
        return 0
    fi
    TUI_LOG2=$TUI_LOG3; TUI_LOG3=$TUI_LOG4
    TUI_LOG4=$TUI_LOG5; TUI_LOG5=$TUI_LOG6; TUI_LOG6=$line
}

tui_begin() { eval "TUI_S_$1=running"; TUI_CUR=$1; eval "TUI_P_$1=0"; tui_frame; }
tui_done()  { eval "TUI_S_$1=done"; eval "TUI_P_$1=100"; TUI_CUR=; tui_frame; }
tui_prog()  { eval "TUI_P_$1=$2"; eval "TUI_RATE_$1=\${3:-}"; eval "TUI_ETA_$1=\${4:-}"; [ "$TUI_PLAIN" = 1 ] || tui_frame; }
tui_fail()  { eval "TUI_S_$1=failed"; TUI_CUR=$1; TUI_ERRCMD=${2:-}; TUI_STATE=failed; tui_frame; }

tui_bar() {   # tui_bar <percent> <width>
    p=$1
    w=$2
    [ "$p" -gt 100 ] 2>/dev/null && p=100
    [ "$p" -lt 0 ] 2>/dev/null && p=0
    fill=$(( p * w / 100 ))
    out=""
    i=0
    while [ "$i" -lt "$w" ]; do
        if [ "$i" -lt "$fill" ]; then out="$out#"; else out="$out."; fi
        i=$((i + 1))
    done
    printf '%s' "$out"
}

tui_frame() {
    [ "$TUI_PLAIN" = 1 ] && return 0
    case $(( TUI_FRAME % 4 )) in
        0) sp='-' ;;
        1) sp='\' ;;
        2) sp='|' ;;
        3) sp='/' ;;
    esac
    scr=$(printf '\033[H')
    n=0
    while [ "$n" -lt "$TUI_LOGO_LINES" ]; do
        n=$((n + 1))
        eval "ln=\$TUI_LOGO_$n"
        scr="$scr
${E}[36m$ln${E}[0m"
    done
    scr="$scr
 ${E}[1m$TUI_TITLE${E}[0m - $TUI_SUB
 target: ${TUI_TARGET:-?}    elapsed $(tui_hm "$(tui_elapsed)")
"
    for k in $TUI_KEYS; do
        eval "st=\$TUI_S_$k"
        eval "lb=\$TUI_L_$k"
        eval "pc=\$TUI_P_$k"
        case "$st" in
            pending) g="${E}[2m.${E}[0m" ;;
            running) g='${E}[33m$sp${E}[0m' ;;
            done) g="${E}[32m+${E}[0m" ;;
            failed) g="${E}[31mx${E}[0m" ;;
        esac
        if [ "$st" = running ]; then
            eval "rt=\${TUI_RATE_$k:-}"
            eval "et=\${TUI_ETA_$k:-}"
            scr="$scr
  $g $lb  [$(tui_bar "$pc" 28)] ${pc}% ${rt:+$rt }${et:+ETA $et}"
        else
            scr="$scr
  $g $lb"
        fi
    done
    scr="$scr

 log:"
    for l in "$TUI_LOG2" "$TUI_LOG3" "$TUI_LOG4" "$TUI_LOG5" "$TUI_LOG6"; do
        scr="$scr
   ${E}[2m$l${E}[0m"
    done
    if [ "${TUI_STATE:-run}" = failed ]; then
        scr="$scr

${E}[31m FAILED in step $TUI_CUR: $TUI_ERRCMD${E}[0m
 log : $TUI_LOGPATH
 The disk is left as it is; nothing else is written."
    else
        scr="$scr

 Ctrl-C is refused while writing."
    fi
    scr="$scr
${E}[J"
    printf '%s' "$scr"
    TUI_FRAME=$((TUI_FRAME + 1))
    if [ -n "${TUI_FRAMES_DUMP:-}" ]; then
        now=$(date +%s)
        if [ $(( now - ${TUI_LAST_DUMP:-0} )) -ge 2 ]; then
            TUI_LAST_DUMP=$now
            printf '%s\n---- frame %s at %s ----\n' \
                "$(printf '%s' "$scr" | sed 's/\033\[[0-9;?]*[a-zA-Z]//g')" "$TUI_FRAME" "$now" >> "$TUI_FRAMES_DUMP"
        fi
    fi
}

tui_run() {   # tui_run <step-key> <command...>: run it, its output into the log tail
    k=$1
    shift
    tui_begin "$k"
    "$@" 2>&1 | while IFS= read -r l; do tui_log "$l"; done
    rc=$?
    if [ "$rc" = 0 ]; then tui_done "$k"; else tui_fail "$k" "$*"; fi
    return $rc
}

tui_copy() {  # tui_copy <step-key> <src-dir> <dst-dir>: cp -a with a real byte counter, MB/s and ETA
    k=$1
    src=$2
    dst=$3
    total=$(du -sk "$src" 2>/dev/null | cut -f1)
    total=${total:-0}
    [ "$total" -gt 0 ] 2>/dev/null || total=1
    tui_begin "$k"
    cp -a "$src/." "$dst/" 2>>"$TUI_LOGPATH" &
    cp_pid=$!
    t0=$(date +%s)
    while kill -0 "$cp_pid" 2>/dev/null; do
        cur=$(du -sk "$dst" 2>/dev/null | cut -f1)
        cur=${cur:-0}
        p=$(( cur * 100 / total ))
        [ "$p" -gt 99 ] && p=99
        el=$(( $(date +%s) - t0 ))
        [ "$el" -lt 1 ] && el=1
        rate=$(( cur / el ))
        div=$rate
        [ "$div" -lt 1 ] && div=1
        left=$(( (total - cur) / div ))
        [ "$left" -lt 0 ] && left=0
        tui_prog "$k" "$p" "$(( rate / 1024 )) MB/s" "$(tui_hm "$left")"
        sleep 1
    done
    wait "$cp_pid"
    rc=$?
    if [ "$rc" = 0 ]; then
        tui_done "$k"
    else
        tui_fail "$k" "cp -a $src/. $dst/"
    fi
    return $rc
}

tui_countdown() {   # tui_countdown <seconds> <text>
    s=$1
    txt=$2
    i=$s
    while [ "$i" -gt 0 ]; do
        if [ "$TUI_PLAIN" = 1 ]; then
            printf '\r  %s %2d s ' "$txt" "$i"
        else
            tui_log "$txt - writing in ${i}s"
            tui_frame
        fi
        sleep 1
        i=$((i - 1))
    done
    [ "$TUI_PLAIN" = 1 ] && printf '\r  writing now      \n'
    return 0
}

tui_ready() {   # tui_ready <ip> <models> <gpu> <npu> <key-hint>
    ip=$1
    models=$2
    gpu=$3
    npu=$4
    hint=$5
    if [ ! -t 1 ]; then
        printf '1bit OS is ready\n  address : %s\n  models  : %s\n  GPU/NPU : %s / %s\n  API key : %s\n' \
            "$ip" "$models" "$gpu" "$npu" "$hint"
        return 0
    fi
    TUI_PLAIN=0
    TUI_TITLE="1bit OS is ready"
    TUI_SUB="1bit OS, Lemonade Server appliance"
    TUI_TARGET=${ip:-unknown}
    TUI_START=$(date +%s)
    TUI_CUR=
    TUI_KEYS=""
    TUI_STATE=run
    TUI_LOG2="address : ${ip:-none}"
    TUI_LOG3="models  : ${models:-none}"
    TUI_LOG4="GPU/NPU : ${gpu:-none} / ${npu:-none}"
    TUI_LOG5="API key : ${hint:-see /data/lemonade/lemond.conf}"
    TUI_LOG6=""
    printf '\033[H\033[2J\033[?25l'
    i=0
    while [ "$i" -lt 20 ]; do
        tui_frame
        if read -r -t 1 _ 2>/dev/null; then break; fi
        i=$((i + 1))
    done
    tui_end
    return 0
}
