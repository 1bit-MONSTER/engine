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
# fix-ownership.sh <tree> - normalise a tree to root:root WITHOUT losing its setuid/setgid bits.
#
# 3builddistro unpacks the PET packages into rootfs-complete and mksquashfs keeps whatever ownership it finds, so the
# tree is chowned immediately before it is packed (cycle 16: the frugalpup docs arrived owned by 6975/2030 and the
# packed-SFS check refused them). What that fix also did, and nothing noticed until the c20 board reported it, is
# strip every setuid bit in the image: POSIX says chown() clears S_ISUID/S_ISGID on every file it touches, and
# `chown -R` touches all of them. Measured on the c21 build tree: rootfs-complete carried 12 setuid helpers
# (dbus-daemon-launch-helper, passwd, chsh, chfn, sudo.ws, ssh-keysign, umount-FULL, mount-FULL, su-FULL, pkexec's
# siblings, fusermount3, ntfs-3g, gpasswd) and the packed puppy sfs carried none - which is exactly the board's
# "Activated service 'org.freedesktop.login1' failed: The permission of the setuid helper is not correct" and why
# a password change or `su` cannot work on the image.
#
# So: remember every setuid/setgid entry, chown, put the bits back, and refuse to return if the count changed.
set -u
tree=${1:-}
[ -n "$tree" ] && [ -d "$tree" ] || { echo "fix-ownership: usage: fix-ownership.sh <tree>" >&2; exit 2; }

list=$(mktemp) || exit 2
trap 'rm -f "$list"' EXIT
find "$tree" -perm /6000 \( -type f -o -type d \) -printf '%m %p\n' > "$list" || exit 2
before=$(wc -l < "$list")

chown -R 0:0 "$tree" || { echo "fix-ownership: chown -R 0:0 $tree failed" >&2; exit 2; }

while IFS=' ' read -r mode path; do
    [ -n "$mode" ] && [ -n "$path" ] || continue
    chmod "$mode" "$path" || { echo "fix-ownership: chmod $mode $path failed" >&2; exit 2; }
done < "$list"

after=$(find "$tree" -perm /6000 \( -type f -o -type d \) | wc -l)
if [ "$after" != "$before" ]; then
    echo "fix-ownership: $tree had $before setuid/setgid entries, now $after - refusing to pack it" >&2
    exit 1
fi
# ── the BASE layer's modes (engine#1474) ─────────────────────────────────────────────────────────────────────────────
# This hook runs immediately before mksquashfs, which is the last moment the tree exists and the right place to take the
# group/other-write bits off: woof-CE's build leaves ~2,176 such paths (the builder's umask, carried by `cp -a`), nothing
# needs them because the appliance runs as root, and the keep-list keeps the eight classes that do. The normaliser only
# NARROWS (go-w), so it cannot expose a secret the way "other files 0644" would, and the reporter is checked here rather
# than trusted: a tree that is still writable outside the keep-list is refused, exactly like a moved setuid count.
here=$(cd "$(dirname "$0")" && pwd)
keep=${BASE_KEEP:-$here/base-modes-keep}
# the normaliser is in the repository's scripts/: /repo in the build container (the repo is mounted there), and three
# levels up from this hook on a host. Both are tried, because this file runs in both places.
normaliser=''
for _c in /repo/scripts/base-modes.sh "$here/../../../scripts/base-modes.sh"; do
    [ -x "$_c" ] && { normaliser=$_c; break; }
done
if [ ! -f "$keep" ]; then
    echo "fix-ownership: no keep-list at $keep: the base layer's modes were NOT normalised (engine#1474)" >&2
elif [ -z "$normaliser" ]; then
    echo "fix-ownership: no scripts/base-modes.sh found (looked in /repo and beside this hook): the base layer's modes were NOT normalised (engine#1474)" >&2
else
    bash "$normaliser" "$tree" --keep "$keep" || {
        echo "fix-ownership: the base layer's modes could not be normalised (engine#1474)" >&2; exit 1; }
fi

echo "fix-ownership: $tree is root:root with $after setuid/setgid entries restored"
