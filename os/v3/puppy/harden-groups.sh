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
# harden-groups.sh <etc/group file>
#
# Puppy's base adds its desktop user `spot` (uid 502) to `disk` (6) and `kmem` (15): raw block devices and kernel memory,
# so a process running as spot can read or rewrite any disk (the A/B slots, the data partition) or the running kernel -
# root in all but name, which undercuts the signed in-system updater (found by cl-npuconc on the installed
# v2026.49-rc1). This removes every member other than root from the groups whose members are root-equivalent:
#     disk kmem shadow root sudo wheel
# and leaves every other group (audio, video, plugdev, input, render, tty, dialout, cdrom, ...) exactly as it is.
# container-build.sh runs it on the rootfs skeleton, cut.sh on the base SFS's copy. Idempotent; exit 1 when the file has
# no `disk:` line (not the group file we expect: a woof-CE change to look at). The release check's privileged-groups
# row checks the result on the running system.
set -eu
f=${1:?usage: harden-groups.sh <etc/group file>}
grep -q '^disk:' "$f" || { echo "harden-groups.sh: REFUSED: $f has no disk: line (is it an /etc/group?)" >&2; exit 1; }
tmp=$f.harden.$$
awk -F: -v OFS=: '
    BEGIN { split("disk kmem shadow root sudo wheel", g, " "); for (i in g) priv[g[i]] = 1 }
    NF >= 4 && ($1 in priv) {
        n = split($4, m, ","); keep = ""
        for (i = 1; i <= n; i++) if (m[i] == "root") keep = "root"
        $4 = keep
    }
    { print }' "$f" > "$tmp"
# same line count, same group names, same gids: only member lists changed
[ "$(wc -l < "$tmp")" = "$(wc -l < "$f")" ] && [ "$(cut -d: -f1,3 "$tmp")" = "$(cut -d: -f1,3 "$f")" ] || {
    rm -f "$tmp"; echo "harden-groups.sh: REFUSED: the rewrite changed more than member lists" >&2; exit 1; }
cat "$tmp" > "$f"; rm -f "$tmp"
