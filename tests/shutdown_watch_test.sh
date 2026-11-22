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
printf '#!/bin/ash\nsync\n' > "$t/other"
sh "$hook" "$t/other" 2> /dev/null; [ $? = 1 ]; check $? "an rc.shutdown without the banner is refused (exit 1)"

echo "shutdown_watch_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
