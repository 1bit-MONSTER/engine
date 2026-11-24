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
# hook-rc-shutdown.sh <rc.shutdown file>
#
# engine#3287: puts one line into woof-CE's /etc/rc.d/rc.shutdown, right after its `Executing /etc/rc.d/rc.shutdown...`
# banner, that starts /usr/local/bin/1bit-shutdown-watch on rc.shutdown's own pid in the background. The watcher says
# in the kernel log which step the shutdown is in, so a reboot that stalls (the c35 installer's freeze; the 22 Nov
# plain reboot that sat 3 min 24 s in rc.shutdown) shows where. container-build.sh runs this on the rootfs skeleton,
# cut.sh on the base SFS's copy. Idempotent; exit 1 when the banner line is not there (a woof-CE change to look at).
#
# engine#3361: replaces the stray-unmount loop's `busybox fuser -k -m "$MOUNT_POINT"` with a kill of the `fuser -m`
# list one pid at a time, skipping rc.shutdown ($$), its caller ($PPID) and the console shell - the exclusion woof-CE's
# ONEBAD loop below already makes. `/sbin/reboot` runs /etc/rc.d/rc.shutdown as a child and only afterwards calls
# `/bin/busybox reboot` (woof-CE sbin/poweroff lines 48 and 63); when the reboot was started from /data
# (`cd /data && ... /sbin/reboot`, the release scripts) rc.shutdown and the shell that would reboot both have a cwd
# or an open log fd on /data, so `fuser -k -m` SIGKILLed them and nothing was left to call `busybox reboot`: the box
# sat half shut down until Ctrl-Alt-Del (three rc1 board-run reboots, engine#3361). Idempotent; exit 1 when that loop
# line is not there either (a woof-CE change to look at).
set -eu
f=${1:?usage: hook-rc-shutdown.sh <rc.shutdown>}

# Refuse before changing anything when a line a patch below anchors on is gone (a woof-CE change to look at): the
# #3287 watcher goes after woof-CE's `Executing /etc/rc.d/rc.shutdown...` banner, and the #3361 patch replaces the
# stray-unmount loop's kill line, which is matched verbatim.
mark='# 1bit engine#3287: say each shutdown step in the kernel log'
smark='# 1bit engine#3361: never fuser-kill rc.shutdown or its caller'
skey='  [ "$xFUSER" ] && busybox fuser -k -m "$MOUNT_POINT"'
if ! grep -qF "$mark" "$f"; then
    grep -q '^echo "Executing /etc/rc.d/rc.shutdown\.\.\."' "$f" || {
        echo "hook-rc-shutdown.sh: REFUSED: $f has no 'Executing /etc/rc.d/rc.shutdown...' line to hook after" >&2; exit 1; }
fi
if ! grep -qF "$smark" "$f" && [ "$(grep -cF "$skey" "$f")" != 1 ]; then
    echo "hook-rc-shutdown.sh: REFUSED: $f has no single stray-unmount 'busybox fuser -k -m \"\$MOUNT_POINT\"' line to patch" >&2
    exit 1
fi

# engine#3287: the step watcher, right after the banner.
if ! grep -qF "$mark" "$f"; then
    sed -i "/^echo \"Executing \/etc\/rc.d\/rc.shutdown\.\.\.\"/a\\
$mark\\
[ -x /usr/local/bin/1bit-shutdown-watch ] \&\& /usr/local/bin/1bit-shutdown-watch \$\$ < /dev/null > /dev/null 2>\&1 \&" "$f"
    [ "$(grep -c '1bit-shutdown-watch \$\$' "$f")" = 1 ] || {
        echo "hook-rc-shutdown.sh: REFUSED: the hook line did not land exactly once in $f" >&2; exit 1; }
fi

# engine#3361: a second witness rc.shutdown cannot stop (1bit-shutdown-heartbeat), started DETACHED so it is init's
# child and not a step watcher v2 would follow; right after the watcher line.
hmark='# 1bit engine#3361: a heartbeat rc.shutdown cannot kill'
if ! grep -qF "$hmark" "$f"; then
    sed -i "/1bit-shutdown-watch \\\$\\\$ < \/dev\/null/a\\
$hmark\\
[ -x /usr/local/bin/1bit-shutdown-heartbeat ] \&\& ( /usr/local/bin/1bit-shutdown-heartbeat \$\$ < /dev/null > /dev/null 2>\&1 \& )" "$f"
    [ "$(grep -c '1bit-shutdown-heartbeat \$\$' "$f")" = 1 ] || {
        echo "hook-rc-shutdown.sh: REFUSED: the heartbeat line did not land exactly once in $f" >&2; exit 1; }
fi

# engine#3361: the stray-unmount loop's kill has to leave rc.shutdown and the process that runs `busybox reboot` alone.
# The line it replaces, verbatim from woof-CE's rc.shutdown, runs `busybox fuser -k -m` (default signal KILL) over
# every pid with a cwd or an open file on the mount, $$ and $PPID included. A multi-line replacement is done here in
# the shell rather than with sed: no quoting of the block for sed, and `cat` back over the file keeps its mode (a
# `sed -i`/mv would leave rc.shutdown non-executable, and poweroff execs it).
if ! grep -qF "$smark" "$f"; then
    tmp="$f.1bit-3361.$$"
    while IFS= read -r line || [ -n "$line" ]; do
        if [ "$line" = "$skey" ]; then
            printf '%s\n' \
                "$smark" \
                '  # `fuser -k -m` here SIGKILLs rc.shutdown ($$) and its parent /sbin/poweroff ($PPID) too when the reboot' \
                '  # was started from this mount (`cd /data && ... /sbin/reboot` leaves a cwd or an open log fd on /data),' \
                '  # and then nothing is left to run `/bin/busybox reboot`: the box sits half shut down until Ctrl-Alt-Del.' \
                '  # Kill the `fuser -m` list one pid at a time instead, skipping rc.shutdown, its caller and the console' \
                '  # shell, as the ONEBAD loop below already does.' \
                '  if [ "$xFUSER" ]; then' \
                '    # busybox ps right-aligns the pid (`    1 root      0:00 /bin/busybox init` on the rc1 image), so drop all' \
                '    # the padding before reading it; the ONEBAD loop below strips only one leading space.' \
                '    STRAYSHID=`ps -A 2>/dev/null | grep " -sh$" | head -n 1 | sed -e "s/^ *//" | cut -f 1 -d " "`' \
                '    [ "$STRAYSHID" ] || STRAYSHID=-1' \
                '    for ONESTRAY in $xFUSER ; do' \
                '      ONESTRAY=${ONESTRAY%m} # the ONEBAD loop below strips the same marker before it matches' \
                '      case "$ONESTRAY" in ""|*[!0-9]*) continue ;; esac' \
                '      case "$ONESTRAY" in $$|$PPID|$STRAYSHID) continue ;; esac' \
                '      kill -9 "$ONESTRAY" 2>/dev/null' \
                '    done' \
                '  fi'
        else
            printf '%s\n' "$line"
        fi
    done < "$f" > "$tmp"
    cat "$tmp" > "$f"
    rm -f "$tmp"
    [ "$(grep -cF "$smark" "$f")" = 1 ] && ! grep -qF "$skey" "$f" || {
        echo "hook-rc-shutdown.sh: REFUSED: the engine#3361 patch did not land exactly once in $f" >&2; exit 1; }
fi
exit 0
