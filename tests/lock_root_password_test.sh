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
# lock-root-password.sh: root's password field becomes '*' and nothing else in /etc/shadow changes; the old field is
# never printed. A fake shadow file in a temp dir: no root, no image.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
s=$here/os/v3/puppy/lock-root-password.sh
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
old='$1$OLDsalt$abcdefghijklmnopqrstuv'
printf 'root:%s:20400:0:99999:7:::\ndaemon:*:20400:0:99999:7:::\nspot:!:20400:0:99999:7:::\nonebit:$y$j9T$x$y:20401:0:99999:7:::\n' "$old" > "$t/shadow"
cp "$t/shadow" "$t/orig"
out=$(sh "$s" "$t/shadow" 2>&1); check $? "exit 0"
[ "$(awk -F: '$1 == "root" { print $2 }' "$t/shadow")" = '*' ]; check $? "root's password field is '*'"
[ "$(grep -v '^root:' "$t/shadow")" = "$(grep -v '^root:' "$t/orig")" ]; check $? "every other user's line is byte-identical"
[ "$(grep '^root:' "$t/shadow" | cut -d: -f1,3-)" = "$(grep '^root:' "$t/orig" | cut -d: -f1,3-)" ]; check $? "root's other fields are kept"
! printf '%s' "$out" | grep -qF 'OLDsalt'; check $? "the old field is never printed"
cp "$t/shadow" "$t/once"; sh "$s" "$t/shadow"; cmp -s "$t/once" "$t/shadow"; check $? "a second run changes nothing"
printf 'daemon:*:20400:0:99999:7:::\n' > "$t/noroot"; cp "$t/noroot" "$t/noroot.orig"
sh "$s" "$t/noroot" 2> "$t/err"; [ $? = 1 ] && grep -q "no root: line" "$t/err" && cmp -s "$t/noroot" "$t/noroot.orig"
check $? "a file with no root: line is refused (exit 1) and left untouched"
[ ! -e "$t/shadow.lock.$$" ] && [ -z "$(ls "$t" | grep '\.lock\.')" ]; check $? "no temporary copy is left beside the file"

echo "lock_root_password_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
