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
# OS updater U4: on an INSTALLED machine the console's item 5 is the updater (check -> confirm -> --latest -> reboot
# offer; roll back; channel toggle). Sourced with ONEBIT_MENU_LIB=1 on the text-terminal path (no dialog), with a fake
# 1bit-update that records what it was asked, so no root, no disk, no network. An install or a rollback must never
# happen without a person's "y", and never non-interactively.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
m=$here/os/v3/puppy/embed/usr/local/bin/1bit-console-menu
t=$(mktemp) && rm -f "$t" && mkdir -p "$t" || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
cat > "$t/up" << 'S'
#!/bin/sh
echo "$*" >> "$UP_CALLS"
case $1 in
  --check) cat "$UP_CHECK" ;;
  --status) echo "1bit-update: active slot = 1bit" ;;
  --latest|--rollback) exit 0 ;;
esac
S
chmod +x "$t/up"
printf 'running:  v2026.48\nstable: v2026.49 (2026-11-30)\nimage:    1bit-os-v2026.49.img.zst, 1400 MiB\nan update is available: 1bit-update --latest installs it into the other slot\n' > "$t/avail"
printf 'running:  v2026.48\nstable: v2026.49 (2026-11-30)\nan update is available: x\nnote: v2026.49 carries no SHA256SUMS.sig, so --latest will refuse it\n' > "$t/unsigned"
run() {  # run <stdin> <check file> <shell code>: the menu's functions on the text path, installed, reboot recorded
    rm -f "$t/calls"
    printf '%s' "$1" | ONEBIT_MENU_LIB=1 ONEBIT_UPDATE=$t/up ONEBIT_UPDATE_CONF=$t/update.conf UP_CALLS=$t/calls UP_CHECK=$2 \
        sh -c '. "$1"; dialog_ok() { return 1; }; installed_mode() { return 0; }; reboot() { echo REBOOT >> "$UP_CALLS"; }
               show_text() { cat "$2"; }; '"$3" _ "$m" > "$t/out" 2>&1
}
run '' "$t/avail" 'NONINT=1; confirm "x"'; [ $? = 1 ]; check $? "non-interactive confirm is NO"
run 'y
' "$t/avail" 'NONINT=0; confirm "x"'; check $? "a typed y is yes"
run '
' "$t/avail" 'NONINT=0; confirm "x"'; [ $? = 1 ]; check $? "Enter alone is no (the default is no)"
run '' "$t/avail" 'NONINT=1; item_update'; grep -q '^--check$' "$t/calls" && grep -q 'update-checked (channel stable)' "$t/out" && ! grep -q -- '--latest' "$t/calls"
check $? "--run 5 on an installed machine checks only, never installs"
run '1
y
n
' "$t/avail" 'NONINT=0; item_update'; grep -q '^--latest --channel stable$' "$t/calls" && ! grep -q REBOOT "$t/calls"
check $? "check -> y installs with --latest on the current channel; n to the reboot offer does not reboot"
run '1
y
y
' "$t/avail" 'NONINT=0; item_update'; grep -q REBOOT "$t/calls"; check $? "y to the reboot offer reboots"
run '1
n
' "$t/avail" 'NONINT=0; item_update'; ! grep -q -- '--latest' "$t/calls"; check $? "n to the install question installs nothing"
run '1
y
' "$t/unsigned" 'NONINT=0; item_update'; ! grep -q -- '--latest' "$t/calls"; check $? "an unsigned release is not even offered"
run '2
y
n
' "$t/avail" 'NONINT=0; item_update'; grep -q '^--rollback$' "$t/calls"; check $? "roll back: --status shown, y runs --rollback"
run '3
y
' "$t/avail" 'NONINT=0; item_update'; [ "$(cat "$t/update.conf")" = channel=test ]; check $? "the channel toggle switches to test after a y (update.conf)"
run '3
' "$t/avail" 'NONINT=0; item_update'; [ "$(cat "$t/update.conf")" = channel=stable ]; check $? "and back to stable without a question"
run '' "$t/avail" 'item5_label'; grep -qx "Update / roll back 1bit OS" "$t/out"; check $? "item 5 reads 'Update / roll back 1bit OS' on an installed machine"

echo "console_update_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
