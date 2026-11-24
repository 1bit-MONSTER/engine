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
# harden_groups_test.sh - os/v3/puppy/harden-groups.sh on the shape of Puppy's base /etc/group (spot in disk and kmem,
# measured in the v2026.47/48/49 base SFS): the root-equivalent groups keep root only, every desktop group is untouched,
# a second run changes nothing, and a file that is not a group file is refused. awk, sed, grep only.
set -u
here=$(cd "$(dirname "$0")" && pwd)
h=$here/../os/v3/puppy/harden-groups.sh
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
bad=0
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
cat > "$t/group" <<'G'
root:x:0:
users:x:100:
spot:x:502:spot
disk:x:6:root,spot
tty:x:5:root,spot
kmem:x:15:root,spot
sudo:x:27:
audio:x:29:root,spot
shadow:x:42:spot
video:x:44:root,spot
plugdev:x:46:root,spot
wheel:x:10:spot,root
render:x:106:root,spot
G
cp "$t/group" "$t/orig"
sh "$h" "$t/group"; check $? "exit 0 on Puppy's group file"
grep -qx 'disk:x:6:root' "$t/group" && grep -qx 'kmem:x:15:root' "$t/group" && grep -qx 'shadow:x:42:' "$t/group" &&
    grep -qx 'wheel:x:10:root' "$t/group"; check $? "disk, kmem, shadow, wheel keep root only (spot removed)"
for g in tty audio video plugdev render spot users root sudo; do
    [ "$(grep "^$g:" "$t/group")" = "$(grep "^$g:" "$t/orig")" ] || { echo "  changed: $g"; false; } || break
done; check $? "every other group line is byte-for-byte unchanged (audio, video, plugdev, render, tty, ...)"
cp "$t/group" "$t/once"; sh "$h" "$t/group"; cmp -s "$t/group" "$t/once"; check $? "a second run changes nothing"
printf 'not a group file\n' > "$t/junk"; sh "$h" "$t/junk" 2> /dev/null; [ $? = 1 ]; check $? "a file with no disk: line is refused (exit 1)"
echo "harden_groups_test: $([ $bad = 0 ] && echo ok || echo "$bad FAIL")"
[ $bad = 0 ]
