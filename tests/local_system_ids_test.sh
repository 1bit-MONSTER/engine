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
# local-system-ids.sh: login.defs gets SYS_UID_MIN/MAX + SYS_GID_MIN/MAX 800/899 (set keys replaced, missing ones
# appended, every other line byte-identical, comments included), refused when an id in 800-899 is taken in the dir's
# own or another layer's passwd/group, a dir with no login.defs left alone. Synthetic files; no root, no image.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
s=$here/os/v3/puppy/local-system-ids.sh
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
k() { awk -v k="$1" '$1 == k { print $2; exit }' "$t/main/login.defs"; }
mk() {
    rm -rf "$t/main" "$t/bdrv"; mkdir -p "$t/main" "$t/bdrv"
    printf '# login.defs\nUID_MIN\t\t\t 1000\nUID_MAX\t\t\t60000\n#SYS_UID_MIN\t\t  101\n#SYS_UID_MAX\t\t  999\nGID_MIN\t\t\t 1000\n#SYS_GID_MIN\t\t  101\n#SYS_GID_MAX\t\t  999\n' > "$t/main/login.defs"
    printf 'root:x:0:0::/root:/bin/sh\nspot:x:502:502::/home/spot:/bin/sh\n' > "$t/main/passwd"
    printf 'root::0:\nspot::502:\n' > "$t/main/group"
    printf 'systemd-network:x:998:998::/:/bin/false\n' > "$t/bdrv/passwd"
    printf 'systemd-journal::999:\n' > "$t/bdrv/group"
}

mk; cp "$t/main/login.defs" "$t/orig"
sh "$s" "$t/main" "$t/bdrv"; check $? "exit 0"
[ "$(k SYS_UID_MIN)-$(k SYS_UID_MAX)/$(k SYS_GID_MIN)-$(k SYS_GID_MAX)" = 800-899/800-899 ]; check $? "the four keys are 800 / 899"
keys='^(SYS_UID_MIN|SYS_UID_MAX|SYS_GID_MIN|SYS_GID_MAX)[[:space:]]'
[ "$(grep -Ev "$keys" "$t/main/login.defs")" = "$(cat "$t/orig")" ]; check $? "every other line (the commented defaults included) is byte-identical"
cp "$t/main/login.defs" "$t/once"; sh "$s" "$t/main" "$t/bdrv"; cmp -s "$t/once" "$t/main/login.defs"; check $? "a second run changes nothing"
mk; printf 'SYS_UID_MIN\t101\nSYS_UID_MAX\t999\n' >> "$t/main/login.defs"; sh "$s" "$t/main" "$t/bdrv"
[ "$(k SYS_UID_MIN)-$(k SYS_UID_MAX)" = 800-899 ] && [ "$(grep -c '^SYS_UID_MIN' "$t/main/login.defs")" = 1 ]; check $? "a key already set is replaced, not doubled"
mk; printf 'x:x:850:850::/:/bin/false\n' >> "$t/main/passwd"; cp "$t/main/login.defs" "$t/b"
sh "$s" "$t/main" "$t/bdrv" 2> "$t/err"; [ $? = 1 ] && cmp -s "$t/b" "$t/main/login.defs" && grep -q "800-899" "$t/err"
check $? "an id in 800-899 in the layer's own passwd is refused, nothing written"
mk; printf 'y::812:\n' >> "$t/bdrv/group"; cp "$t/main/login.defs" "$t/b"
sh "$s" "$t/main" "$t/bdrv" 2> /dev/null; [ $? = 1 ] && cmp -s "$t/b" "$t/main/login.defs"; check $? "an id in 800-899 in ANOTHER layer's group is refused too"
mk; rm "$t/main/login.defs"; sh "$s" "$t/main" "$t/bdrv"; [ $? = 0 ] && [ ! -e "$t/main/login.defs" ]; check $? "a dir with no login.defs is left alone"
mk; sh "$s" "$t/main" "$t/bdrv"; printf 'runner:x:899:899::/var/lib/runner:/bin/false\n' >> "$t/main/passwd"; cp "$t/main/login.defs" "$t/b"
sh "$s" "$t/main" "$t/bdrv"; [ $? = 0 ] && cmp -s "$t/b" "$t/main/login.defs"
check $? "on a live /etc already set to 800-899, a box account at 899 is fine: a no-op, not a refusal"
[ -z "$(ls "$t/main" | grep '\.ids\.')" ]; check $? "no temporary copy is left"

echo "local_system_ids_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
