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
# shell-login.sh <rootfs>
#
# U1 (owner, cycle 12): root and every real user log in to /bin/bash, not to busybox ash.
#
# Measured in the woof-CE tree: rootfs-skeleton/etc/passwd ships `root:x:0:0:root:/root:/bin/sh`, and
# bin/plogin ends with `exec getty -n -l /bin/autologin ... tty1`, so /bin/autologin ends in the shell named
# in that field - one field, and the board's `u1-login: no account has bash as its shell` is exactly it.
#
# Runs on the build host (and therefore inside the build container), on the skeleton BEFORE 3builddistro
# packs it. It is its own script because build.sh's container block is single-quoted for docker: an awk
# program inline there ends the block early, the quoting trap that has now cost four builds.
#
# Exit 0 rewritten and verified, 1 refused (nothing usable), 2 the tree has no passwd.
set -u
root=${1:-}
[ -n "$root" ] && [ -d "$root" ] || { echo "shell-login: usage: shell-login.sh <rootfs>" >&2; exit 1; }
pw=$root/etc/passwd
[ -f "$pw" ] || { echo "shell-login: no $pw" >&2; exit 2; }
# bash is NOT in rootfs-skeleton at this stage - it arrives with the debootstrap base (measured: the packed
# image carries usr/bin/bash in bdrv) - so its presence is asserted on the packed image by build.sh, not
# here. This script only moves the field; refusing on the skeleton would refuse every build.
if [ -x "$root/bin/bash" ] || [ -x "$root/usr/bin/bash" ]; then
    echo "shell-login: bash is in the skeleton"
else
    echo "shell-login: bash is not in the skeleton yet (it comes from the base); build.sh checks the image"
fi

# root and every human account (uid >= 1000); system accounts keep their nologin/false shells
tmp=$pw.new
# root, every uid >= 1000, and Puppy's own `spot` (uid 502, the account the desktop apps run as and the only
# non-root account a person logs in as). NOT every /bin/sh account: woof-CE's passwd gives `daemon` and
# `nobody` /bin/sh too, and a service account with a login shell is not what "any user lands in bash" means
# (measured: the first version of this script converted both, so uid 65534 - nobody - is excluded as well).
awk -F: 'BEGIN { OFS = ":" }
         $1 == "root" || $1 == "spot" || (($3 + 0) >= 1000 && ($3 + 0) < 65534) { $7 = "/bin/bash" }
         { print }' "$pw" > "$tmp" || {
    echo "shell-login: REFUSED: could not rewrite $pw" >&2; exit 1; }
cat "$tmp" > "$pw" || { echo "shell-login: REFUSED: could not write $pw" >&2; exit 1; }
rm -f "$tmp"

bad=$(awk -F: '($1 == "root" || $1 == "spot" || (($3 + 0) >= 1000 && ($3 + 0) < 65534)) && $7 != "/bin/bash" { printf "%s(%s) ", $1, $7 }' "$pw")
[ -z "$bad" ] || { echo "shell-login: REFUSED: still not bash: $bad" >&2; exit 1; }
users=$(awk -F: '($1 == "root" || $1 == "spot" || (($3 + 0) >= 1000 && ($3 + 0) < 65534)) { printf "%s ", $1 }' "$pw")
echo "login shells -> /bin/bash: $users"
