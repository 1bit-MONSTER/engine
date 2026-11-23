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
# engine#3287: 1bit-shutdown-watch names the step a shutdown is in, in the kernel log, and hook-rc-shutdown.sh starts
# it from rc.shutdown. The watcher runs on a REAL process tree of this test (a script waiting on `sleep`), with the
# kernel log replaced by a file (ONEBIT_KMSG), so it needs no root and no reboot; the hook runs on a copy of the lines
# of woof-CE's rc.shutdown it anchors on.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
watch=$here/os/v3/puppy/embed/usr/local/bin/1bit-shutdown-watch
hook=$here/os/v3/puppy/hook-rc-shutdown.sh
hb=$here/os/v3/puppy/embed/usr/local/bin/1bit-shutdown-heartbeat
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }

# 1. the watcher on a stand-in rc.shutdown: a script that runs `sleep 4` (a step that stalls), then `sleep 1`
printf 'sleep 4\nsleep 1\n' > "$t/rc"
sh "$t/rc" & rc=$!
ONEBIT_KMSG=$t/kmsg ONEBIT_WATCH_FIRST=2 sh "$watch" "$rc"
check $? "the watcher exits 0 once rc.shutdown has ended"
wait "$rc" 2> /dev/null
cat "$t/kmsg"
grep -q "^1bit-shutdown: watching rc.shutdown $rc\$" "$t/kmsg"; check $? "it says what it watches"
grep -qE '^1bit-shutdown: step [0-9]+ sleep 4$' "$t/kmsg"; check $? "the stalling step is named with its command line"
grep -qE '^1bit-shutdown: still in [0-9]+ sleep 4 after 2 s, state [A-Z], wchan [^?]' "$t/kmsg"
check $? "a step past the threshold gets its state and wchan"
grep -qE '^1bit-shutdown: step [0-9]+ sleep 1$' "$t/kmsg"; check $? "the next step is named when it starts"
grep -q "^1bit-shutdown: rc.shutdown $rc finished\$" "$t/kmsg"; check $? "the end of rc.shutdown is said"
[ "$(grep -c 'step [0-9]* sleep 4$' "$t/kmsg")" = 1 ]; check $? "a step is named once, not every second"
! grep -qv '^1bit-shutdown: ' "$t/kmsg"; check $? "every record is one whole line (one write each)"

# 1b. v2: a heartbeat while rc.shutdown waits, and a SIGTERM (rc.shutdown's kill loop) is said before the watcher exits
printf 'sleep 5\n' > "$t/rc2"
sh "$t/rc2" & rc2=$!
ONEBIT_KMSG=$t/kmsg2 ONEBIT_WATCH_FIRST=60 ONEBIT_WATCH_HEARTBEAT=2 sh "$watch" "$rc2" & w2=$!
sleep 3
kill -TERM "$w2"; wait "$w2" 2> /dev/null; wait "$rc2" 2> /dev/null
cat "$t/kmsg2"
grep -qE '^1bit-shutdown: alive: rc.shutdown in [0-9]+ \[sleep\], state [A-Z], wchan ' "$t/kmsg2"
check $? "a heartbeat names where rc.shutdown is (comm, state, wchan) while nothing changes"
grep -qE '^1bit-shutdown: watcher got SIGTERM while rc.shutdown was in [0-9]+ \(sleep 5\)$' "$t/kmsg2"
check $? "a SIGTERM to the watcher is logged with the step it was watching"
if command -v busybox > /dev/null 2>&1; then
    printf 'sleep 2\n' > "$t/rc3"
    busybox ash "$t/rc3" & rc3=$!
    ONEBIT_KMSG=$t/kmsg3 busybox ash "$watch" "$rc3"; wait "$rc3" 2> /dev/null
    grep -q "^1bit-shutdown: rc.shutdown $rc3 finished\$" "$t/kmsg3" && grep -qE "step [0-9]+ sleep 2$" "$t/kmsg3"
    check $? "under busybox ash (the appliance's shell, builtin wait) the watcher still follows and ends"
fi

# 2. the hook, on the banner lines of woof-CE's rc.shutdown
printf '#!/bin/ash\necho "${DISTRO_NAME} is now shutting down..." > /dev/console\necho "Executing /etc/rc.d/rc.shutdown..."\nsync\n' \
    > "$t/rc.shutdown"
sh "$hook" "$t/rc.shutdown"; check $? "the hook goes in (exit 0)"
sed -n 4p "$t/rc.shutdown" | grep -q '^# 1bit engine#3287' &&
    sed -n 5p "$t/rc.shutdown" | grep -qF '[ -x /usr/local/bin/1bit-shutdown-watch ] && /usr/local/bin/1bit-shutdown-watch $$ < /dev/null > /dev/null 2>&1 &'
check $? "right after the Executing banner, backgrounded on rc.shutdown's own pid"
cp "$t/rc.shutdown" "$t/once"; sh "$hook" "$t/rc.shutdown"
cmp -s "$t/rc.shutdown" "$t/once"; check $? "a second run changes nothing"
sh -n "$t/rc.shutdown"; check $? "the hooked script still parses"
sed -n 6p "$t/rc.shutdown" | grep -q '^# 1bit engine#3361' &&
    sed -n 7p "$t/rc.shutdown" | grep -qF '[ -x /usr/local/bin/1bit-shutdown-heartbeat ] && ( /usr/local/bin/1bit-shutdown-heartbeat $$ < /dev/null > /dev/null 2>&1 & )'
check $? "the heartbeat line follows the watcher line, started detached (init's child, not a step)"
printf '#!/bin/ash\necho "Executing /etc/rc.d/rc.shutdown..."\n# 1bit engine#3287: say each shutdown step in the kernel log\n[ -x /usr/local/bin/1bit-shutdown-watch ] && /usr/local/bin/1bit-shutdown-watch $$ < /dev/null > /dev/null 2>&1 &\nsync\n' > "$t/old"
sh "$hook" "$t/old"; [ "$(grep -c '1bit-shutdown-heartbeat \$\$' "$t/old")" = 1 ] && [ "$(grep -c '1bit-shutdown-watch \$\$' "$t/old")" = 1 ]
check $? "an rc.shutdown hooked before #3361 gets the heartbeat too, and keeps one watcher line"

# 3. engine#3361: the heartbeat on a stand-in rc.shutdown that loses its step watcher, then stalls in `sleep 9`
printf '#!/bin/sh\nsleep 30\n' > "$t/1bit-shutdown-watch"; chmod +x "$t/1bit-shutdown-watch"
printf '#!/bin/sh\n"%s" & w=$!\nsleep 3\nkill $w\nsleep 9\n' "$t/1bit-shutdown-watch" > "$t/rc4"
sh "$t/rc4" & rc4=$!
sleep 1
( ONEBIT_KMSG="$t/kmsg4" ONEBIT_SYSRQ="$t/sysrq" ONEBIT_HB_DIR="$t/hbdir" ONEBIT_HB_PERIOD=2 ONEBIT_HB_W=3 ONEBIT_HB_STACKS=5 \
    sh "$hb" $rc4 < /dev/null > /dev/null 2>&1 & )
sleep 2
hbpid=$(sed -n 's/^1bit-shutdown-hb: watching rc.shutdown [0-9]* as \([0-9]*\) .*/\1/p' "$t/kmsg4")
[ -n "$hbpid" ] && kill -TERM "$hbpid" 2> /dev/null
wait $rc4
for i in 1 2 3 4 5 6; do grep -q "finished" "$t/kmsg4" 2> /dev/null && break; sleep 1; done
grep -q "^1bit-shutdown-hb: step watcher [0-9]* is GONE (rc.shutdown in [0-9]* " "$t/kmsg4"
check $? "it says when the step watcher disappears, and where rc.shutdown was then"
grep -qE "^1bit-shutdown-hb: alive: rc.shutdown in [0-9]+ \[sleep\] state S" "$t/kmsg4" && ! grep -q "alive: .*\[1bit-shutdown-w\]" "$t/kmsg4"
check $? "its heartbeat names rc.shutdown's step (never the step watcher)"
[ "$(cat "$t/sysrq" 2> /dev/null)" = w ] && grep -q "for [0-9]*s: sysrq w" "$t/kmsg4" && grep -q "stacks of its chain and of every D/R task" "$t/kmsg4"
check $? "a step that sits still gets sysrq w, then the stack dump"
grep -q "got SIGTERM, ignored" "$t/kmsg4" && grep -q "^1bit-shutdown-hb: rc.shutdown $rc4 finished" "$t/kmsg4"
check $? "SIGTERM (rc.shutdown's kill passes) is ignored: it runs on to rc.shutdown's end"
[ "$(grep -c "^1bit-shutdown-hb: alive:" "$t/kmsg4")" -le 8 ]
check $? "it ticks once a second, it does not spin (a 13 s shutdown, a 2 s period: at most 8 heartbeats)"
# a tick that returns at once (here: a fake `sleep` that does not sleep) is caught and replaced
mkdir -p "$t/fastbin"; printf '#!/bin/sh\nexit 0\n' > "$t/fastbin/sleep"; chmod +x "$t/fastbin/sleep"
printf '#!/bin/sh\n/bin/sleep 3\n' > "$t/rc5"; sh "$t/rc5" & rc5=$!
( PATH="$t/fastbin:$PATH" ONEBIT_KMSG="$t/kmsg5" ONEBIT_SYSRQ="$t/sysrq5" ONEBIT_HB_DIR="$t/hbdir5" ONEBIT_HB_PERIOD=100 \
    sh "$hb" $rc5 < /dev/null > /dev/null 2>&1 & )
wait $rc5; sleep 2
[ "$(grep -c "tick returned at once; falling back to /bin/sleep" "$t/kmsg5")" = 1 ] && grep -q "rc.shutdown $rc5 finished" "$t/kmsg5"
check $? "a tick that returns at once is caught ONCE (more than 3 in a second -> /bin/sleep) and it still ends with rc.shutdown"
[ -f "$t/hbdir/hb" ] && grep -q "as $hbpid (parent 1" "$t/kmsg4"
check $? "it runs from its tmpfs copy, as init's child (detached)"

printf '#!/bin/ash\nsync\n' > "$t/other"
sh "$hook" "$t/other" 2> /dev/null; [ $? = 1 ]; check $? "an rc.shutdown without the banner is refused (exit 1)"

echo "shutdown_watch_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
