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
# engine#3361: rc.shutdown's stray-unmount loop ran 'busybox fuser -k -m "$MOUNT_POINT"' (woof-CE rc.shutdown, the
# "Karl Godt: 2013-12-14 reworked the whole unmount block" section). fuser -k -m sends SIGKILL to every process with a
# cwd or an open file on the mount, rc.shutdown itself ($$) and its parent included. Puppy's /sbin/poweroff runs
# /etc/rc.d/rc.shutdown as a child and only afterwards calls /bin/busybox reboot, so a release script's
# 'cd /data && ... /sbin/reboot' left the whole chain holding /data: fuser SIGKILLed rc.shutdown and the shell that
# would reboot in the same second, and the box sat half shut down until Ctrl-Alt-Del (three rc1 board-run reboots,
# engine#3361). hook-rc-shutdown.sh replaces that one line with a kill of the fuser -m list one pid at a time,
# skipping rc.shutdown, its caller and the console shell, as woof-CE's own ONEBAD loop below it already does.
#
# This runs the loop on a real process tree, with a plain directory standing in for /data: no mount, no root, no
# reboot. A busybox stub answers 'fuser -m DIR' (the pids of the simulated tree whose cwd or an open file is under
# DIR, which is what fuser -m lists for a mounted filesystem) and 'fuser -k -m DIR', and records when 'reboot' is
# reached. Its kill order is numeric pid order, as fuser's /proc walk is: the caller (the parent, lower pid) dies
# before rc.shutdown does, which is the real sequence. A ps stub prints the appliance's row format: the rc1 image's
# busybox 1.37 right-aligns the pid (measured there: 'PID   USER     TIME  COMMAND' and '    1 root      0:00
# /bin/busybox init'), so a boot-time console shell's row carries leading spaces and the console-shell lookup has to
# cope with them. The same tree runs against the line as woof-CE ships it (control: the caller, rc.shutdown and the
# console shell are SIGKILLed and busybox reboot is never reached) and against the hooked file (fixed: rc.shutdown
# runs to its end, the caller survives and reboots, the console shell is spared, and a process that really is only
# using the mount is still killed). dash and busybox ash each run both. The appliance's busybox ash resolves 'busybox'
# and 'ps' to its own applets and never looks at PATH (measured: 'command -v busybox' is 'busybox' there), so the
# fixture reaches the stubs through functions, which shadow an applet in ash and a PATH lookup in dash alike.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
hook=$here/os/v3/puppy/hook-rc-shutdown.sh
# the real busybox, for the ash arm: the stubs are reached by function, not by PATH
bbash=$(command -v busybox 2>/dev/null) || bbash=""
t=$(mktemp -d) || exit 1
trap 'rm -rf "${t:?}"' EXIT
bad=0
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
# a child this shell has not reaped yet is a zombie after SIGKILL, and kill -0 still succeeds on a zombie
alive() {
    [ -d "/proc/$1" ] || return 1
    st=$(sed -e 's/^[^)]*) //' "/proc/$1/stat" 2>/dev/null | cut -d ' ' -f 1)
    [ "$st" != Z ]
}

# The stand-in for the appliance's busybox: the applets this test needs. 'fuser -m DIR' lists the pids the test put in
# the simulated tree that have DIR as cwd or an open file under it, space-separated as busybox prints them; -k sends
# SIGKILL, busybox fuser's default signal, in numeric pid order (/proc order). 'reboot' records that poweroff reached
# the line after rc.shutdown.
cat > "$t/busybox" <<'STUB'
#!/bin/sh
cmd=${1:-}
if [ $# -gt 0 ]; then shift; fi
case $cmd in
fuser)
    killit=0
    path=""
    for a in "$@" ; do
        case $a in
            -k) killit=1 ;;
            -m) ;;
            -*) ;;
            *) path=$a ;;
        esac
    done
    found=0
    for p in $(cat "$ONEBIT_3361_PIDS_DIR"/* 2>/dev/null | sort -n) ; do
        [ "$p" = "$$" ] && continue
        hit=0
        c=$(readlink "/proc/$p/cwd" 2>/dev/null) || c=""
        case "$c" in "$path"|"$path"/*) hit=1 ;; esac
        if [ "$hit" = 0 ]; then
            for f in "/proc/$p"/fd/* ; do
                l=$(readlink "$f" 2>/dev/null) || continue
                case "$l" in "$path"|"$path"/*) hit=1 ; break ;; esac
            done
        fi
        [ "$hit" = 1 ] || continue
        printf '%s\n' "$p" >> "$ONEBIT_3361_FUSER_LOG"
        found=1
        if [ "$killit" = 1 ]; then kill -9 "$p" 2>/dev/null; fi
        printf '%s ' "$p"
    done
    [ "$found" = 1 ] && echo
    exit 0
    ;;
reboot)
    printf '%s\n' "busybox reboot was reached" >> "$ONEBIT_3361_REBOOTED"
    exit 0
    ;;
esac
exit 0
STUB
chmod +x "$t/busybox"

# The appliance's busybox ps -A rows: header then one right-aligned pid per process, command line last. The padding is
# 7 columns here, the width the board's 7-digit pids gave every shorter pid (measured there with pid 1703272), so a
# boot-time console shell's row starts with spaces. The loop only greps ' -sh$', and the test lists only the pids of the
# simulated tree -- which is all the patched line looks for.
cat > "$t/ps" <<'STUB'
#!/bin/sh
printf 'PID   USER     TIME  COMMAND\n'
for p in $(cat "$ONEBIT_3361_PIDS_DIR"/* 2>/dev/null | sort -n) ; do
    [ -d "/proc/$p" ] || continue
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    [ -n "$cmd" ] || continue
    printf '%7s %-8s 0:00 %s\n' "$p" root "${cmd% }"
done
exit 0
STUB
chmod +x "$t/ps"

# woof-CE's rc.shutdown around the lines the two hooks anchor on, with the stray loop cut down to the one simulated
# mount: 'base' is what hook-rc-shutdown.sh sees (and what the control runs), 'fixed' is what it produces. The first
# two lines are the stubs; on the appliance these calls reach the busybox applets instead.
write_rc() {  # <path>
    cat > "$1" <<'RC'
#!/bin/ash
busybox() { "$ONEBIT_3361_BUSYBOX" "$@"; }
ps() { "$ONEBIT_3361_PS" "$@"; }
echo "Executing /etc/rc.d/rc.shutdown..."
cd /
printf '%s\n' "start" >> "$ONEBIT_3361_RCLOG"
echo $$ > "$ONEBIT_3361_RCPID"
printf '%s\n' $$ > "$ONEBIT_3361_PIDS_DIR/rc"
STRAY_MOUNTPOINTS="$ONEBIT_3361_DATA"
for MOUNT_POINT in $STRAY_MOUNTPOINTS ; do
 FLAGCIFS=`echo -n "${MOUNT_POINT}" | grep '^//'`
 if [ ! "$FLAGCIFS" ]; then
  xFUSER=`busybox fuser -m "$MOUNT_POINT" 2>/dev/null`
  [ "$xFUSER" ] && busybox fuser -k -m "$MOUNT_POINT"
 fi
done
printf '%s\n' "end" >> "$ONEBIT_3361_RCLOG"
RC
}

# The stand-in for woof-CE's /sbin/poweroff with the reboot started from /data: cwd and log fd on /data, run
# /etc/rc.d/rc.shutdown as a child, and only then call /bin/busybox reboot (sbin/poweroff lines 48 and 63).
cat > "$t/caller" <<'CALLER'
#!/bin/sh
cd "$ONEBIT_3361_DATA" || exit 1
exec >> "$ONEBIT_3361_DATA/caller.log" 2>&1
printf '%s\n' "caller: $$ in $PWD"
printf '%s\n' $$ > "$ONEBIT_3361_PIDS_DIR/caller"
# the shell under test is word-split on purpose: "busybox ash" is two words
$ONEBIT_3361_SHELL "$ONEBIT_3361_RC" &
rcpid=$!
wait "$rcpid"
st=$?
printf '%s\n' "caller: rc.shutdown returned $st"
"$ONEBIT_3361_BUSYBOX" reboot
CALLER

run_case() {  # <label> <rc.shutdown> <shell words> <fixed|control>
    label=$1
    rcfile=$2
    shell=$3
    expect=$4
    run="$t/run-$label"
    rm -rf "$run"
    mkdir -p "$run/data" "$run/pids"
    : > "$run/fuser.log"
    : > "$run/rebooted"
    ONEBIT_3361_DATA="$run/data"
    ONEBIT_3361_PIDS_DIR="$run/pids"
    ONEBIT_3361_FUSER_LOG="$run/fuser.log"
    ONEBIT_3361_REBOOTED="$run/rebooted"
    ONEBIT_3361_RCLOG="$run/rc.log"
    ONEBIT_3361_RCPID="$run/rcpid"
    ONEBIT_3361_RC="$rcfile"
    ONEBIT_3361_SHELL="$shell"
    ONEBIT_3361_BUSYBOX="$t/busybox"
    ONEBIT_3361_PS="$t/ps"
    export ONEBIT_3361_DATA ONEBIT_3361_PIDS_DIR ONEBIT_3361_FUSER_LOG ONEBIT_3361_REBOOTED
    export ONEBIT_3361_RCLOG ONEBIT_3361_RCPID ONEBIT_3361_RC ONEBIT_3361_SHELL ONEBIT_3361_BUSYBOX ONEBIT_3361_PS
    # a console login shell (argv ends in ' -sh', as the appliance's console shell's does) and a process that really
    # only uses the mount: both sit on the fake /data
    ( cd "$run/data" && exec sh -c 'while :; do sleep 1; done' -sh ) &
    console=$!
    ( cd "$run/data" && exec sleep 300 ) &
    stray=$!
    printf '%s\n' "$console" > "$run/pids/console"
    printf '%s\n' "$stray" > "$run/pids/stray"
    i=0
    while [ "$i" -lt 50 ]; do
        [ "$(readlink "/proc/$console/cwd" 2>/dev/null)" = "$run/data" ] &&
            [ "$(readlink "/proc/$stray/cwd" 2>/dev/null)" = "$run/data" ] && break
        i=$((i + 1))
        sleep 0.1
    done
    $shell "$t/caller" &   # word-split on purpose: "busybox ash" is two words
    caller=$!
    wait "$caller" 2>/dev/null
    caller_st=$?
    callerpid=$(cat "$run/pids/caller" 2>/dev/null)
    rcpid=$(cat "$run/rcpid" 2>/dev/null)
    # the control's caller dies while the fuser stub is still walking the list; wait for rc.shutdown to be gone too,
    # so the log and the survivors are read after the kill pass has finished
    i=0
    while [ "$i" -lt 100 ]; do
        alive "$rcpid" || break
        i=$((i + 1))
        sleep 0.05
    done
    console_alive=1
    alive "$console" || console_alive=0
    stray_alive=1
    alive "$stray" || stray_alive=0
    kill -9 "$console" "$stray" 2>/dev/null
    wait "$console" 2>/dev/null
    wait "$stray" 2>/dev/null
    consolepid=$(cat "$run/pids/console")
    grep -qx "$callerpid" "$run/fuser.log"
    check $? "$label: fuser -m lists the caller (its cwd and its log fd are on the mount)"
    grep -qx "$rcpid" "$run/fuser.log"
    check $? "$label: fuser -m lists rc.shutdown itself, the pid the old line SIGKILLed"
    grep -qx "$consolepid" "$run/fuser.log"
    check $? "$label: fuser -m lists the console shell"
    if [ "$expect" = fixed ]; then
        [ "$stray_alive" = 0 ]
        check $? "$label: the process that only used the mount was still SIGKILLed"
        [ -s "$run/rebooted" ] && [ "$caller_st" = 0 ]
        check $? "$label: the caller survived rc.shutdown and reached busybox reboot"
        [ "$(tail -n 1 "$run/rc.log" 2>/dev/null)" = end ]
        check $? "$label: rc.shutdown ran to its end"
        [ "$console_alive" = 1 ]
        check $? "$label: the console shell was spared"
    else
        [ ! -s "$run/rebooted" ] && [ "$caller_st" != 0 ]
        check $? "$label: control: the caller was SIGKILLed and busybox reboot was never reached"
        [ "$(tail -n 1 "$run/rc.log" 2>/dev/null)" != end ]
        check $? "$label: control: rc.shutdown itself was SIGKILLed before its end"
        [ "$console_alive" = 0 ]
        check $? "$label: control: the unpatched line killed the console shell too"
    fi
}

# 1. the hook replaces the one-line kill with the per-pid loop that spares rc.shutdown, its caller and the console shell
write_rc "$t/base"
cp "$t/base" "$t/hooked"
sh "$hook" "$t/hooked"
check $? "the hook patches woof-CE's stray-unmount kill line (exit 0)"
grep -qF '# 1bit engine#3361: never fuser-kill rc.shutdown or its caller' "$t/hooked"
check $? "the patch is marked in the file"
! grep -qF '  [ "$xFUSER" ] && busybox fuser -k -m "$MOUNT_POINT"' "$t/hooked"
check $? "the one-line fuser -k -m is gone"
grep -qF 'case "$ONESTRAY" in $$|$PPID|$STRAYSHID) continue ;; esac' "$t/hooked"
check $? "the per-pid kill skips rc.shutdown, its caller and the console shell"
grep -qF 'STRAYSHID=`ps -A 2>/dev/null | grep " -sh$" | head -n 1 | sed -e "s/^ *//"' "$t/hooked"
check $? "the console shell's pid is read from the padded ps row"
dash -n "$t/hooked"
check $? "the hooked rc.shutdown parses (dash)"
if [ -n "$bbash" ]; then
    "$bbash" ash -n "$t/hooked"
    check $? "the hooked rc.shutdown parses (busybox ash)"
fi
cp "$t/hooked" "$t/hooked.once"
sh "$hook" "$t/hooked"
cmp -s "$t/hooked" "$t/hooked.once"
check $? "a second run changes nothing"

# 2. a woof-CE change to the anchor line is refused, and the file is not changed first
sed -e 's/busybox fuser -k -m/busybox fuser -K -m/' "$t/base" > "$t/moved"
cp "$t/moved" "$t/moved.before"
sh "$hook" "$t/moved" 2>/dev/null
[ $? = 1 ]
check $? "an rc.shutdown whose stray-unmount line moved is refused (exit 1)"
cmp -s "$t/moved" "$t/moved.before"
check $? "and the refused file is left as it was"

# 3. the loop itself: the control (unpatched, as woof-CE ships it) and the hooked file, under dash and busybox ash
for shell in dash "$bbash ash" ; do
    case $shell in
        " ash") echo "skip busybox ash: no busybox on this box"; continue ;;
        dash) label=dash ;;
        *) label=busybox-ash ;;
    esac
    run_case "control-$label" "$t/base" "$shell" control
    cp "$t/base" "$t/fixed-$label"
    sh "$hook" "$t/fixed-$label" || { check 1 "$label: the hook applies to the fixture"; continue; }
    run_case "fixed-$label" "$t/fixed-$label" "$shell" fixed
done

echo "rc_shutdown_stray_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
