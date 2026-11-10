#!/usr/bin/env bash
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
# console_chain_test.sh - `embed/usr/local/bin/1bit-console` is the script that owns tty1, and its whole job is
# that there is no path from this screen to a bare prompt. Nothing tested it. Every branch is reachable here with
# the paths rewritten into a temp tree (the script uses absolute paths by design; the copy is what the test runs),
# and each stub records that it ran, so the assertion is "which program took over", not "the script exited 0".
#
# The four branches, in the order the script takes them:
#
#   1. `1bit.installer=1` on the cmdline        -> the installer owns tty1 (a getty and the installer on one
#                                                  terminal eat each other's keystrokes, hence the exec here)
#   2. no `pfix=...nox` (the desktop entry)      -> plogin, which lays out the first X session
#   3. `nox` + a TUI binary                      -> the TUI draws; when it leaves FOR ANY REASON the dialog menu
#                                                  is exec'd, and the exit status is announced on the screen and
#                                                  appended to /data/logs/console-tui.log (silent fallbacks are the
#                                                  defect nobody can name afterwards)
#   4. `nox` + no TUI binary                     -> the menu, with nothing to fall back from
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/os/v3/puppy/embed/usr/local/bin/1bit-console
[ -f "$src" ] || { echo "console_chain_test: no $src"; exit 1; }

work=$(mktemp -d /tmp/cc.XXXXXX) || exit 1
trap 'rm -rf "$work"' EXIT
fails=0
check() {   # <what> <got> <want>
    if [ "$2" = "$3" ]; then echo "ok   $1 = $3"; else echo "FAIL $1 = $2 (wanted $3)"; fails=$((fails + 1)); fi
}

bin=$work/bin
mkdir -p "$bin" "$work/data/logs"
# Each stub records its own name; `menu` and `install` exit 0, the TUI exits with whatever $TUI_RC says.
for s in install plogin menu; do
    printf '#!/bin/sh\necho %s >> %s/ran\n' "$s" "$work" > "$bin/$s"
done
printf '#!/bin/sh\necho tui >> %s/ran\nexit "${TUI_RC:-0}"\n' "$work" > "$bin/tui"
chmod +x "$bin"/*

# The script under test, with its absolute paths pointed at the temp tree. Only path strings change.
sed -e "s|/proc/cmdline|$work/cmdline|g" \
    -e "s|/usr/local/bin/1bit-console-install|$bin/install|g" \
    -e "s|/usr/local/bin/1bit-console-tui|$bin/tui|g" \
    -e "s|/usr/local/bin/1bit-console-menu|$bin/menu|g" \
    -e "s|/bin/plogin|$bin/plogin|g" \
    -e "s|/dev/tty1|$work/tty|g" \
    -e "s|/data/logs/console-tui.log|$work/data/logs/console-tui.log|g" \
    "$src" > "$work/console"
chmod +x "$work/console"

ran() { cat "$work/ran" 2> /dev/null | tr '\n' ' ' | sed 's/ $//'; }
run() {   # <cmdline> [TUI_RC]
    : > "$work/ran"
    : > "$work/tty"
    : > "$work/data/logs/console-tui.log"
    printf '%s\n' "$1" > "$work/cmdline"
    TUI_RC=${2:-0} "$work/console" > "$work/out" 2>&1
    echo $?
}

# ---- 1. the installer entry -----------------------------------------------------------------------------------------
rc=$(run "BOOT_IMAGE=/vmlinuz pfix=copy,nox 1bit.installer=1")
check "installer cmdline: exit" "$rc" "0"
check "installer cmdline: who took over" "$(ran)" "install"

# ---- 2. the desktop entry (no `nox`) --------------------------------------------------------------------------------
rc=$(run "BOOT_IMAGE=/vmlinuz pfix=copy")
check "desktop entry: exit" "$rc" "0"
check "desktop entry: who took over" "$(ran)" "plogin"

# ---- 3. the TUI, leaving non-zero: the menu must still take over, and both announcements must exist ---------------
cp "$bin/tui" "$work/tui.bak"
rc=$(run "BOOT_IMAGE=/vmlinuz pfix=copy,nox" 7)
check "tui exits 7: exit" "$rc" "0"
check "tui exits 7: what ran, in order" "$(ran)" "tui menu"
grep -q '1bit-console-tui exited (7) - starting the dialog menu' "$work/tty" &&
    echo "ok   tui exits 7: the screen was told, with the status" ||
    { echo "FAIL tui exits 7: tty said '$(cat "$work/tty")'"; fails=$((fails + 1)); }
grep -qE "1bit-console-tui exited \(7\) at 20[0-9]{2}-" "$work/data/logs/console-tui.log" &&
    echo "ok   tui exits 7: the log line carries the status and a timestamp" ||
    { echo "FAIL tui exits 7: log said '$(cat "$work/data/logs/console-tui.log")'"; fails=$((fails + 1)); }

# ---- 3b. the TUI leaving ZERO means the same thing: the menu is still the fallback --------------------------------
rc=$(run "BOOT_IMAGE=/vmlinuz pfix=copy,nox" 0)
check "tui exits 0: what ran, in order" "$(ran)" "tui menu"

# ---- 4. no TUI binary at all: the menu, and no announcement about a program that never ran -------------------------
rm -f "$bin/tui"
rc=$(run "BOOT_IMAGE=/vmlinuz pfix=copy,nox")
check "no TUI: exit" "$rc" "0"
check "no TUI: who took over" "$(ran)" "menu"
[ -s "$work/tty" ] &&
    { echo "FAIL no TUI: something was written to the screen: '$(cat "$work/tty")'"; fails=$((fails + 1)); } ||
    echo "ok   no TUI: nothing announced about a binary that does not exist"
cp "$work/tui.bak" "$bin/tui"

echo "console_chain: $fails failure(s)"
[ "$fails" -eq 0 ] || exit 1
exit 0
