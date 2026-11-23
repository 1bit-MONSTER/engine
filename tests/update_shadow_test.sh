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
# engine#3342: a save layer's copy of a RELEASE file keeps the old code over every later release (v2026.48 kept c35's
# hand-edited 05onboard-data). 1bit-update's shadow_* functions, sourced with ONEBIT_UPDATE_LIB=1, run on a fake upper
# dir and a fake release tree laid out like the c35 -> v2026.48 machine: they must find the release code, leave the
# machine's state alone, and move the code OUT of the overlay. Also: every code file the image ships from
# os/v3/puppy/embed is on etc/1bit/release-owned, so a new script cannot be added without the list knowing it.
# No root, no overlay, no device. (A whiteout is a 0/0 character device, which only root can make; it is not built here.)
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
script=$here/os/v3/puppy/embed/usr/local/bin/1bit-update
owned=$here/os/v3/puppy/embed/etc/1bit/release-owned
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
ONEBIT_UPDATE_LIB=1 . "$script"
check $? "1bit-update sources as a library (ONEBIT_UPDATE_LIB=1)"

pats=$(owned_pats "$owned")
printf '%s\n' "$pats" | grep -qx 'etc/init.d/\[0-9\]\[0-9\]onboard-\*' && ! printf '%s\n' "$pats" | grep -q '#'
check $? "owned_pats reads the shipped list: patterns kept, comments dropped"

# the machine: an upper dir with release code and machine state, and the new release's tree
U=$t/save/upper B=$t/release
mk() { mkdir -p "$(dirname "$1")"; printf '%s\n' "$2" > "$1"; }
mk "$B/etc/init.d/05onboard-data" "new 05onboard-data (makes /data/npm-global)"
mk "$U/etc/init.d/05onboard-data" "c35 05onboard-data, hand-edited"
mk "$B/etc/rc.d/rc.shutdown" "rc.shutdown with the watcher hook"
mk "$U/etc/rc.d/rc.shutdown" "rc.shutdown with the watcher hook"
mk "$U/usr/local/bin/1bit-retired" "a tool the new release dropped"
mkdir -p "$B/usr/local/lib/nodejs/bin" "$U/usr/local/lib/nodejs/bin"
ln -s ../lib/node_modules/npm/bin/npm-cli.js "$B/usr/local/lib/nodejs/bin/npm"
ln -s ../lib/node_modules/npm/bin/npm-cli.js "$U/usr/local/lib/nodejs/bin/npm"
for f in etc/passwd etc/shadow etc/sudoers etc/connman/main.conf etc/hosts.allow etc/keymap etc/1bit/lemond.conf \
         etc/rc.d/rc.local etc/init.d/30herdr root/.ssh/authorized_keys etc/hostname; do
    mk "$U/$f" "machine state: $f"; mk "$B/$f" "release default: $f"
done
( cd "$U" && find . -type f -o -type l ) | sort > "$t/before"

shadow_scan "$U" "$pats" > "$t/scan"
cat "$t/scan"
[ "$(cut -f2 "$t/scan" | tr '\n' ' ')" = \
  "etc/init.d/05onboard-data etc/rc.d/rc.shutdown usr/local/bin/1bit-retired usr/local/lib/nodejs/bin/npm " ]
check $? "shadow_scan names exactly the four release-owned entries, none of the machine's state"
grep -qx "link	usr/local/lib/nodejs/bin/npm" "$t/scan" && grep -qx "file	etc/init.d/05onboard-data" "$t/scan"
check $? "a symlink is reported as a link, a regular file as a file"

[ "$(shadow_kind "$B" etc/init.d/05onboard-data "$U")" = differs ] &&
    [ "$(shadow_kind "$B" etc/rc.d/rc.shutdown "$U")" = identical ] &&
    [ "$(shadow_kind "$B" usr/local/bin/1bit-retired "$U")" = "not in the release" ] &&
    [ "$(shadow_kind "$B" usr/local/lib/nodejs/bin/npm "$U")" = identical ]
check $? "shadow_kind: the hand edit differs, the hook copy is identical, the dropped tool is not in the release"

D=$t/save/1bit-shadowed/v2026.49-test
shadow_move "$U" "$pats" "$D" > "$t/moved"; rc=$?
cat "$t/moved"
check $rc "shadow_move returns 0 when every entry moved"
[ -z "$(shadow_scan "$U" "$pats")" ]; check $? "after the move no release-owned path is left in the upper dir"
[ "$(cat "$D/etc/init.d/05onboard-data")" = "c35 05onboard-data, hand-edited" ] && [ -L "$D/usr/local/lib/nodejs/bin/npm" ]
check $? "the moved copies keep their bytes (the hand edit can still be read) and a link stays a link"
case $D in "$U"/*) inside=0 ;; *) inside=1 ;; esac
[ $inside = 1 ] && [ ! -e "$U/etc/init.d/05onboard-data" ]
check $? "they land OUTSIDE upper/, where nothing runs them (an init script left in /etc/init.d would run at boot)"
( cd "$U" && find . -type f -o -type l ) | sort > "$t/after"
grep -vxF -f "$t/after" "$t/before" | sort > "$t/gone"
[ "$(tr '\n' ' ' < "$t/gone")" = "./etc/init.d/05onboard-data ./etc/rc.d/rc.shutdown ./usr/local/bin/1bit-retired ./usr/local/lib/nodejs/bin/npm " ]
check $? "every machine-state file is still in upper/, byte for byte untouched"
[ "$(cat "$U/etc/passwd")" = "machine state: etc/passwd" ]; check $? "accounts are not touched"
shadow_move "$U" "$pats" "$D" > /dev/null; check $? "a second move finds nothing and succeeds"

# the shipped list covers every code file the image takes from embed/ (machine defaults the machine may change are not code)
miss=
for f in $(cd "$here/os/v3/puppy/embed" && find . -type f | sed 's|^\./||' | sort); do
    case $f in etc/hostname|etc/keymap|etc/localtime|etc/timezone|etc/personal_settings_popup_disabled|root/*) continue ;; esac
    hit=
    set -f
    for pat in $pats; do case $f in $pat) hit=1; break ;; esac; done
    set +f
    [ -n "$hit" ] || miss="$miss $f"
done
[ -z "$miss" ]; check $? "every embed code file is release-owned (missing:${miss:- none})"
for f in etc/passwd etc/shadow etc/group etc/sudoers etc/connman/main.conf etc/hosts.allow etc/resolv.conf etc/keymap \
         etc/clock etc/localtime etc/1bit/lemond.conf etc/rc.d/rc.local etc/ssh/ssh_host_ed25519_key root/.ssh/authorized_keys; do
    set -f; for pat in $pats; do case $f in $pat) miss="$miss $f" ;; esac; done; set +f
done
[ -z "$miss" ]; check $? "no machine-state path is on the list (owned:${miss:- none})"

echo "update_shadow_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
