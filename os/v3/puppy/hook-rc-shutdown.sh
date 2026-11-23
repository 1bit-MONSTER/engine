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
set -eu
f=${1:?usage: hook-rc-shutdown.sh <rc.shutdown>}
mark='# 1bit engine#3287: say each shutdown step in the kernel log'
if ! grep -qF "$mark" "$f"; then
    grep -q '^echo "Executing /etc/rc.d/rc.shutdown\.\.\."' "$f" || {
        echo "hook-rc-shutdown.sh: REFUSED: $f has no 'Executing /etc/rc.d/rc.shutdown...' line to hook after" >&2; exit 1; }
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
exit 0
