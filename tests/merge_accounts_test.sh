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
# tools/merge_accounts.py: the save layer's account files merged with the image's - system accounts from the image,
# local users kept with their memberships, a base default password replaced and a person's kept, never printed;
# plus the guards: the release-owned list never names an account file (moving one out would delete local users), and
# release-check's fingerprint set is the tool's. Synthetic files in a temp dir; no root, no image.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
tool=$here/tools/merge_accounts.py
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
DEF='$1$TESTdflt$notarealhashatall0000'                  # stands in for the base default (the test's own fingerprint)
FP=$(printf '%s' "$DEF" | sha256sum | cut -c1-64)
PERSON='$6$person$aaaaaaaaaaaaaaaaaaaaaaaaaaaa'
mk() {
    rm -rf "$t/img" "$t/live"; mkdir -p "$t/img" "$t/live"
    # the image: root locked, spot, messagebus (new in this image); disk without spot (#3452); an 8-field shadow line
    printf 'root:x:0:0:root:/root:/bin/sh\nspot:x:502:502:spot:/home/spot:/bin/sh\nmessagebus:x:81:81::/:/bin/false\n' > "$t/img/passwd"
    printf 'root:*:20400:0:99999:7:::\nspot:!:20400:0:99999:7:::\nmessagebus:*:20400:0:99999:7::\n' > "$t/img/shadow"
    printf 'root::0:\ndisk::6:root\nvideo::28:spot\nwheel::10:root\n' > "$t/img/group"
    printf 'root:::\ndisk:::root\nvideo:::spot\n' > "$t/img/gshadow"
    # the save layer: root with the default (and a changed shell), spot with a person's password, the old spot-in-disk,
    # a local user bcloud in video and wheel, a local group dev; no messagebus
    printf 'root:x:0:0:root:/root:/bin/bash\nspot:x:502:502:spot:/home/spot:/bin/sh\nbcloud:x:1000:1000::/home/bcloud:/bin/bash\n' > "$t/live/passwd"
    printf 'root:%s:20000:0:99999:7:::\nspot:%s:20410:0:99999:7:::\nbcloud:$y$j9T$b$c:20411:0:99999:7:::\n' "$DEF" "$PERSON" > "$t/live/shadow"
    printf 'root::0:\ndisk::6:root,spot\nvideo::28:spot,bcloud\nwheel::10:root,bcloud\ndev::1000:bcloud\n' > "$t/live/group"
    printf 'root:::\ndisk:::root,spot\nvideo:::spot,bcloud\ndev:::bcloud\n' > "$t/live/gshadow"
    chmod 640 "$t/live/shadow"; chmod 600 "$t/live/gshadow"
}
M() { python3 -I "$tool" "$t/img" "$t/live" --known-default-sha256 "$FP" "$@"; }
f() { awk -F: -v n="$2" -v k="$3" '$1 == n { print $k }' "$t/live/$1"; }

mk; cp -a "$t/live" "$t/before"
out=$(M 2>&1); check $? "dry run: exit 0"
diff -r "$t/before" "$t/live" > /dev/null; check $? "dry run: nothing written"
printf '%s' "$out" | grep -q "root held the base image's DEFAULT password"; check $? "dry run: names root's default, by name"

out=$(M --apply --backup "$t/bk" 2>&1); check $? "apply: exit 0"
[ "$(f shadow root 2)" = '*' ]; check $? "root's default password is replaced by the image's '*'"
[ "$(f shadow spot 2)" = "$PERSON" ] && [ "$(f shadow spot 3)" = 20410 ]; check $? "a password a person set on a system account is kept, with its date"
[ "$(f passwd root 7)" = /bin/sh ]; check $? "a system account's passwd line follows the image"
[ -n "$(f passwd messagebus 1)" ] && [ "$(f shadow messagebus 2)" = '*' ]; check $? "a system account new in the image appears (passwd and shadow)"
[ "$(f passwd bcloud 6)" = /home/bcloud ] && [ "$(f shadow bcloud 3)" = 20411 ]; check $? "the local user is kept (passwd and shadow)"
[ "$(f group disk 4)" = root ] && [ "$(f gshadow disk 4)" = root ]; check $? "the image's spot leaves disk (image members follow the image)"
[ "$(f group video 4)" = spot,bcloud ] && [ "$(f group wheel 4)" = root,bcloud ]; check $? "the local user keeps its image-group memberships"
[ "$(f group dev 4)" = bcloud ] && [ "$(f gshadow dev 4)" = bcloud ]; check $? "the local group is kept"
[ "$(tail -1 "$t/live/passwd" | cut -d: -f1)" = bcloud ]; check $? "local accounts come after the image's"
[ "$(awk -F: '$1 == "messagebus" { print NF }' "$t/live/shadow")" = 8 ]; check $? "an 8-field shadow line keeps its 8 fields"
[ "$(stat -c %a "$t/live/shadow")" = 640 ] && [ "$(stat -c %a "$t/live/gshadow")" = 600 ]; check $? "each file keeps its mode"
[ "$(stat -c %a "$t/bk")" = 700 ] && cmp -s "$t/bk/shadow" "$t/before/shadow" && [ "$(stat -c %a "$t/bk/shadow")" = 600 ]
check $? "the previous copies are in the backup (dir 0700, files 0600)"
! printf '%s' "$out" | grep -qF -e "$DEF" -e "$PERSON" -e 'TESTdflt' -e 'person'; check $? "no password field is ever printed"
[ -z "$(ls "$t/live" | grep '\.merge\.')" ]; check $? "no temporary file is left"
out=$(M --apply --backup "$t/bk2" 2>&1); printf '%s' "$out" | grep -q "already agree" && [ ! -e "$t/bk2" ]; check $? "a second run changes nothing"

mk; python3 -I "$tool" "$t/img" "$t/live" --apply --backup "$t/bk3" > /dev/null 2>&1
[ "$(f shadow root 2)" = "$DEF" ]; check $? "without the fingerprint a root field is a person's and is kept (fingerprint matching only)"
mk; sed -i 's/^spot:[^:]*:/spot::/' "$t/live/shadow"; M --apply --backup "$t/bk4" > /dev/null 2>&1
[ "$(f shadow spot 2)" = '!' ]; check $? "an EMPTY saved password on a system account takes the image's field"
mk; printf 'broken:x:1\n' >> "$t/live/passwd"; cp -a "$t/live" "$t/before2"
out=$(M --apply --backup "$t/bk5" 2>&1); rc=$?
[ $rc = 1 ] && printf '%s' "$out" | grep -q REFUSED && diff -r "$t/before2" "$t/live" > /dev/null && [ ! -e "$t/bk5" ]
check $? "a file that does not parse is refused (exit 1) and nothing is written"
mk; python3 -I "$tool" "$t/img" "$t/live" --apply > /dev/null 2>&1; [ $? = 2 ]; check $? "--apply without --backup is a usage error"

! grep -Eq '^etc/(passwd|group|shadow|gshadow)([[:space:]]|$)|^etc/\*' "$here/os/v3/puppy/embed/etc/1bit/release-owned"
check $? "release-owned names no account file (1bit-update's move-out would delete local users)"
tfp=$(python3 -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import merge_accounts as m; print(" ".join(sorted(m.KNOWN_DEFAULT_SHA256)))' "$here/tools")
rfp=$(sed -n 's/^DEFAULT_PW_SHA256="\(.*\)"$/\1/p' "$here/os/mini/release-check.sh" | tr ' ' '\n' | sort | tr '\n' ' ' | sed 's/ $//')
[ -n "$tfp" ] && [ "$tfp" = "$rfp" ]; check $? "release-check's default-password fingerprints are the tool's"

echo "merge_accounts_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
