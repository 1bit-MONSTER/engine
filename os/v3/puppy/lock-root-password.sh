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
# lock-root-password.sh <etc/shadow file>
#
# The base image the 1bit OS cuts are patched from ships root with the base distribution's default password, which is
# public, so any non-root process could become root with `su` (setuid) and any member of the base's %users could sudo
# with it - undercutting the root-only model the signed in-system updater relies on. This replaces root's password
# field with `*`: no password matches it, so `su`/sudo-with-the-root-password cannot be used, while root's console
# autologin (tty1, getty -l) and root's KEY-only ssh keep working. `*` rather than `!`: OpenSSH treats a `!`-prefixed
# field as a LOCKED account and, without PAM, refuses even key logins; `*` is not locked, just password-less.
# Nothing else in the file changes (checked: same lines, same users, every other field and line byte-identical), and
# the old field is never printed. A root password a person set on an installed machine lives in its save layer and is
# kept; only the image's default changes. container-build.sh runs it on the rootfs skeleton, cut.sh on the base SFS's
# copy. Idempotent; exit 1 when the file has no root: line (not the shadow file we expect). The release check's
# root-password row checks the result on the running system.
set -eu
f=${1:?usage: lock-root-password.sh <etc/shadow file>}
grep -q '^root:' "$f" || { echo "lock-root-password.sh: REFUSED: $f has no root: line (is it an /etc/shadow?)" >&2; exit 1; }
tmp=$f.lock.$$
( umask 077; awk -F: -v OFS=: '$1 == "root" && NF >= 2 { $2 = "*" } { print }' "$f" > "$tmp" )
# the same users in the same order, and every line that is not root's byte-identical
same=1
[ "$(wc -l < "$tmp")" = "$(wc -l < "$f")" ] && [ "$(cut -d: -f1 "$tmp")" = "$(cut -d: -f1 "$f")" ] || same=0
[ "$(grep -v '^root:' "$tmp")" = "$(grep -v '^root:' "$f")" ] || same=0
[ "$(grep '^root:' "$tmp" | cut -d: -f1,3-)" = "$(grep '^root:' "$f" | cut -d: -f1,3-)" ] || same=0
[ "$same" = 1 ] || { rm -f "$tmp"; echo "lock-root-password.sh: REFUSED: the rewrite changed more than root's password field" >&2; exit 1; }
cat "$tmp" > "$f"; rm -f "$tmp"
