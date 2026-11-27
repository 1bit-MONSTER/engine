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
# tools/merge_accounts.py: the save layer's account files merged with the image's LAYERS - image accounts appear and
# keep a person's shell/password, local users kept with their memberships, a base default password replaced, id
# clashes refused, never a field printed;
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
    printf 'root::0:\ndisk::6:root\nvideo::28:spot\nwheel::10:root\naudio::29:\n' > "$t/img/group"
    printf 'root:::\ndisk:::root\nvideo:::spot\n' > "$t/img/gshadow"
    # the save layer: root with the default (and a changed shell), spot with a person's password, the old spot-in-disk,
    # a local user bcloud in video and wheel, a local group dev; no messagebus
    printf 'root:x:0:0:root:/root:/bin/bash\nspot:x:502:502:spot:/home/spot:/bin/sh\nbcloud:x:1000:1000::/home/bcloud:/bin/bash\n' > "$t/live/passwd"
    printf 'root:%s:20000:0:99999:7:::\nspot:%s:20410:0:99999:7:::\nbcloud:$y$j9T$b$c:20411:0:99999:7:::\n' "$DEF" "$PERSON" > "$t/live/shadow"
    printf 'root::0:\ndisk::6:root,spot\nvideo::28:spot,bcloud\nwheel::10:root,bcloud\naudio::29:spot\ndev::1000:bcloud\n' > "$t/live/group"
    printf 'root:::\ndisk:::root,spot\nvideo:::spot,bcloud\ndev:::bcloud\n' > "$t/live/gshadow"
    chmod 640 "$t/live/shadow"; chmod 600 "$t/live/gshadow"
}
M() { python3 -I "$tool" "$t/img" "$t/live" --known-default-sha256 "$FP" "$@"; }
f() { awk -F: -v n="$2" -v k="$3" '$1 == n { print $k }' "$t/live/$1"; }

mk; cp -a "$t/live" "$t/before"
out=$(M 2>&1); check $? "dry run: exit 0"
diff -r "$t/before" "$t/live" > /dev/null; check $? "dry run: nothing written"
printf '%s' "$out" | grep -q "root held the base image's DEFAULT password"; check $? "dry run: names root's default, by name"

out=$(M --apply --backup "$t/bk/1" --status "$t/status" 2>&1); check $? "apply: exit 0"
[ "$(f shadow root 2)" = '*' ]; check $? "root's default password is replaced by the image's '*'"
[ "$(f shadow spot 2)" = "$PERSON" ] && [ "$(f shadow spot 3)" = 20410 ]; check $? "a password a person set on a system account is kept, with its date"
[ "$(f passwd root 7)" = /bin/bash ]; check $? "a shell a person set on a system account is kept (chsh survives)"
[ -n "$(f passwd messagebus 1)" ] && [ "$(f shadow messagebus 2)" = '*' ]; check $? "a system account new in the image appears (passwd and shadow)"
[ "$(f passwd bcloud 6)" = /home/bcloud ] && [ "$(f shadow bcloud 3)" = 20411 ]; check $? "the local user is kept (passwd and shadow)"
[ "$(f group disk 4)" = root ] && [ "$(f gshadow disk 4)" = root ]; check $? "the image's spot leaves disk (root-equivalent groups follow the image)"
[ "$(f group video 4)" = spot,bcloud ] && [ "$(f group wheel 4)" = root,bcloud ]; check $? "the local user keeps its image-group memberships (wheel included)"
[ "$(f group audio 4)" = spot ]; check $? "a membership a person gave an image account in an ordinary group is kept (usermod -aG audio spot)"
[ "$(f group dev 4)" = bcloud ] && [ "$(f gshadow dev 4)" = bcloud ]; check $? "the local group is kept"
[ "$(tail -1 "$t/live/passwd" | cut -d: -f1)" = bcloud ]; check $? "local accounts come after the image's"
[ "$(awk -F: '$1 == "messagebus" { print NF }' "$t/live/shadow")" = 8 ]; check $? "an 8-field shadow line keeps its 8 fields"
[ "$(stat -c %a "$t/live/shadow")" = 640 ] && [ "$(stat -c %a "$t/live/gshadow")" = 600 ]; check $? "each file keeps its mode"
[ "$(stat -c %a "$t/bk/1")" = 700 ] && cmp -s "$t/bk/1/shadow" "$t/before/shadow" && [ "$(stat -c %a "$t/bk/1/shadow")" = 600 ]
check $? "the previous copies are in the backup (dir 0700, files 0600)"
grep -q ' ok: rewrote' "$t/status"; check $? "--status records ok"
! printf '%s' "$out" | grep -qF -e "$DEF" -e "$PERSON" -e TESTdflt -e "\$6\$person"; check $? "no password field is ever printed"
[ -z "$(ls "$t/live" | grep '\.merge\.')" ]; check $? "no temporary file is left"
out=$(M --apply --backup "$t/bk/2" 2>&1); printf '%s' "$out" | grep -q "already agree" && [ ! -e "$t/bk/2" ]; check $? "a second run changes nothing"
for i in 3 4 5 6 7 8; do mk; M --apply --backup "$t/bk/$i" > /dev/null 2>&1; done
[ "$(ls "$t/bk" | wc -l)" = 5 ] && [ -d "$t/bk/8" ] && [ ! -d "$t/bk/1" ]; check $? "only the newest 5 backups are kept"

# two image layers, as 1bit OS mounts them (bdrv above the main SFS): each file is the first layer's that has it
mk; mkdir -p "$t/bdrv"
printf 'root:x:0:0:root:/root:/bin/sh\nspot:x:502:502:spot:/home/spot:/bin/sh\nmessagebus:x:81:81::/:/bin/false\npipewire:x:994:994::/:/bin/false\n' > "$t/bdrv/passwd"
printf 'root:*:20500:0:99999:7:::\nspot:!:20400:0:99999:7:::\nmessagebus:*:20400:0:99999:7::\npipewire:!:20400:0:99999:7:::\n' > "$t/bdrv/shadow"
printf 'root:!:20400:0:99999:7:::\nspot:!:20400:0:99999:7:::\nmessagebus:*:20400:0:99999:7::\n' > "$t/img/shadow"
python3 -I "$tool" --image "$t/bdrv" --image "$t/img" "$t/live" --known-default-sha256 "$FP" --apply --backup "$t/bkL" > /dev/null 2>&1
[ "$(f passwd pipewire 3)" = 994 ] && [ "$(f shadow pipewire 2)" = '!' ]; check $? "layers: the upper layer's passwd/shadow are the image's (its pipewire appears)"
[ "$(f shadow root 2)" = '*' ] && [ "$(f shadow root 3)" = 20500 ]; check $? "layers: root takes the UPPER layer's field, not the lower one's"
[ "$(f gshadow disk 4)" = root ]; check $? "layers: a file only the lower layer has (gshadow) is still merged"
mk; mkdir -p "$t/bdrv2"; cp "$t/bdrv/passwd" "$t/bdrv2/passwd"
printf 'pipewire:x:994:994::/:/bin/false\n' >> "$t/live/passwd"; printf 'pipewire:!:20400:0:99999:7:::\n' >> "$t/live/shadow"
python3 -I "$tool" --image "$t/bdrv2" --image "$t/img" "$t/live" --apply --backup "$t/bkL2" > /dev/null 2>&1
[ "$(grep -n '^pipewire:' "$t/live/passwd" | cut -d: -f1)" = 4 ]; check $? "layers: bdrv's pipewire is an IMAGE account (in the image's order), not a local one"

# refusals: nothing written, the status says why (names and ids only)
mk; sed -i 's/^spot:x:502:502:/spot:x:1501:1501:/' "$t/live/passwd"; cp -a "$t/live" "$t/b2"
out=$(M --apply --backup "$t/bk9" --status "$t/st2" 2>&1); rc=$?
[ $rc = 1 ] && printf '%s' "$out" | grep -q "account spot is 1501/1501 on this box but 502/502 in the image" && diff -r "$t/b2" "$t/live" > /dev/null && [ ! -e "$t/bk9" ] && grep -q ' refused: account spot' "$t/st2"
check $? "a saved account with the image's name but another uid is refused, never re-owned (status says why)"
mk; printf 'clash:x:81:81::/home/clash:/bin/sh\n' >> "$t/live/passwd"; printf 'clash:%s:20412:0:99999:7:::\n' "$PERSON" >> "$t/live/shadow"; cp -a "$t/live" "$t/b3"
out=$(M --apply --backup "$t/bk10" 2>&1); rc=$?
[ $rc = 1 ] && printf '%s' "$out" | grep -q "the image adds account messagebus with id 81, which the local clash already has" && diff -r "$t/b3" "$t/live" > /dev/null && ! printf '%s' "$out" | grep -qF -e "$PERSON" -e "\$6\$person"
check $? "an account the image adds on a local account's uid is refused (names and ids only)"
mk; printf 'root::0:\ndisk::6:root\nvideo::28:spot\nwheel::10:root\naudio::29:\nnewgrp::1000:\n' > "$t/img/group"; cp -a "$t/live" "$t/b4"
out=$(M --apply --backup "$t/bk11" 2>&1); rc=$?
[ $rc = 1 ] && printf '%s' "$out" | grep -q "the image adds group newgrp with id 1000, which the local dev already has" && diff -r "$t/b4" "$t/live" > /dev/null
check $? "a group the image adds on a local group's gid is refused"

mk; printf 'odd:x:1500:1500::/:/bin/sh\n' >> "$t/live/passwd"; printf 'odd:\377\376:1:0:99999:7:::\n' >> "$t/live/shadow"
out=$(M --apply --backup "$t/bk12" 2>&1); rc=$?
[ $rc = 0 ] && ! printf '%s' "$out" | grep -q Traceback; check $? "a non-UTF-8 byte in a password field gives an answer, not a traceback (rc $rc)"
mk; python3 -I "$tool" "$t/img" "$t/live" --apply --backup "$t/bk3x" > /dev/null 2>&1
[ "$(f shadow root 2)" = "$DEF" ]; check $? "without the fingerprint a root field is a person's and is kept (fingerprint matching only)"
mk; sed -i 's/^spot:[^:]*:/spot::/' "$t/live/shadow"; M --apply --backup "$t/bk4x" > /dev/null 2>&1
[ "$(f shadow spot 2)" = '!' ]; check $? "an EMPTY saved password on a system account takes the image's field"
mk; printf 'broken:x:1\n' >> "$t/live/passwd"; cp -a "$t/live" "$t/before2"
out=$(M --apply --backup "$t/bk5x" 2>&1); rc=$?
[ $rc = 1 ] && printf '%s' "$out" | grep -q REFUSED && diff -r "$t/before2" "$t/live" > /dev/null && [ ! -e "$t/bk5x" ]
check $? "a file that does not parse is refused (exit 1) and nothing is written"
mk; python3 -I "$tool" "$t/img" "$t/live" --apply > /dev/null 2>&1; [ $? = 2 ]; check $? "--apply without --backup is a usage error"
mk; printf '%s:x:1:\n%s:x:1:\n' "$DEF" "$DEF" >> "$t/live/shadow"
out=$(M 2>&1); rc=$?
[ $rc = 1 ] && printf '%s' "$out" | grep -q "not printable" && ! printf '%s' "$out" | grep -qF -e "$DEF" -e TESTdflt
check $? "a shadow name column that looks like a hash is never echoed, even in a refusal"

! grep -Eq '^etc/(passwd|group|shadow|gshadow)([[:space:]]|$)|^etc/\*' "$here/os/v3/puppy/embed/etc/1bit/release-owned"
check $? "release-owned names no account file (1bit-update's move-out would delete local users)"
tfp=$(python3 -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import merge_accounts as m; print(" ".join(sorted(m.KNOWN_DEFAULT_SHA256)))' "$here/tools")
rfp=$(sed -n 's/^DEFAULT_PW_SHA256="\(.*\)"$/\1/p' "$here/os/mini/release-check.sh" | tr ' ' '\n' | sort | tr '\n' ' ' | sed 's/ $//')
[ -n "$tfp" ] && [ "$tfp" = "$rfp" ]; check $? "release-check's default-password fingerprints are the tool's"

echo "merge_accounts_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
