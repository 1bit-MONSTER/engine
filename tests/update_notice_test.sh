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
# OS updater U6: 1bit-update-notice writes "<tag> <channel> <date>" to the notice file when `1bit-update --check` says
# an update is available, removes it otherwise, asks at most once a day, never installs, and is silenced by
# notice=off; the console's header line carries the notice. A fake 1bit-update records its calls: no network, no root.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
n=$here/os/v3/puppy/embed/usr/local/bin/1bit-update-notice
m=$here/os/v3/puppy/embed/usr/local/bin/1bit-console-menu
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
cat > "$t/up" << 'S'
#!/bin/sh
echo "$*" >> "$CALLS"; cat "$ANSWER"
S
chmod +x "$t/up"
printf 'running:  v2026.48\nstable: v2026.50 (2026-12-07)\nan update is available: 1bit-update --latest installs it\n' > "$t/avail"
printf 'running:  v2026.50\nstable: v2026.50 (2026-12-07)\nup to date\n' > "$t/none"
printf 'running:  v2026.48\nstable: v2026.50;rm -rf / (2026-12-07)\nan update is available: x\n' > "$t/hostile"
go() {  # go <answer file> [day seconds]
    CALLS=$t/calls ANSWER=$1 ONEBIT_UPDATE=$t/up ONEBIT_UPDATE_CONF=$t/update.conf ONEBIT_UPDATE_NOTICE_STAMP=$t/stamp \
    ONEBIT_UPDATE_NOTICE_OUT=$t/notice ONEBIT_UPDATE_NOTICE_DAY=${2:-86400} sh "$n" --once
}
go "$t/avail"; [ "$(cat "$t/notice")" = "v2026.50 stable 2026-12-07" ]; check $? "an available update writes '<tag> <channel> <date>'"
grep -q -- '--latest' "$t/calls"; [ $? = 1 ] && [ "$(cat "$t/calls")" = "--check" ]; check $? "it only ever asks --check (never installs)"
go "$t/none"; [ "$(wc -l < "$t/calls")" = 1 ] && [ -f "$t/notice" ]; check $? "within a day it does not ask again (one call; the notice stays)"
go "$t/none" 0; [ ! -e "$t/notice" ]; check $? "a day later, up to date: the notice is removed"
go "$t/hostile" 0; [ ! -e "$t/notice" ]; check $? "a tag outside the safe charset never reaches the notice file"
go "$t/avail" 0; echo 'notice=off' > "$t/update.conf"; go "$t/avail" 0; [ ! -e "$t/notice" ]; check $? "notice=off removes the notice and writes nothing"
rm -f "$t/update.conf"; go "$t/avail" 0
h=$(ONEBIT_MENU_LIB=1 ONEBIT_UPDATE_NOTICE_OUT=$t/notice sh -c '. "$1"; release_tag() { echo v2026.48; }; hostname_now() { echo box; }; first_ip() { echo 10.0.0.2; }; header_line' _ "$m")
[ "$h" = "1bit OS v2026.48   box   10.0.0.2   update v2026.50: item 5" ]; check $? "the console header carries the notice ($h)"
rm -f "$t/notice"
h=$(ONEBIT_MENU_LIB=1 ONEBIT_UPDATE_NOTICE_OUT=$t/notice sh -c '. "$1"; release_tag() { echo v2026.48; }; hostname_now() { echo box; }; first_ip() { echo 10.0.0.2; }; header_line' _ "$m")
[ "$h" = "1bit OS v2026.48   box   10.0.0.2" ]; check $? "no notice file: the header is unchanged"

echo "update_notice_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
