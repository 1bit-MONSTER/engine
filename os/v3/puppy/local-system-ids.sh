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
# local-system-ids.sh <etc dir> [<other layer's etc dir>...]
#
# A box's own system accounts (`useradd -r`: a CI runner, a sandbox user, a package's daemon) must not take ids the
# image's own system accounts use. login.defs names no SYS_UID_MAX in the base, so it defaults to UID_MIN-1 = 999 and
# useradd -r allocates from 999 DOWN - exactly where the image's system accounts sit (systemd-network 998, pipewire
# 994, polkitd 993; groups 993-999). On strixhalo a local ghrunner took 999 and tpsandbox 997. An image that later
# adds a system account the same way picks a free id from the same range, the account merge (06onboard-accounts) then
# finds two names on one id and refuses, and the box stays on its unmerged files (engine#3555 review, cl-flm).
# This gives the box's own system accounts a range the image never uses: SYS_UID_MIN/MAX and SYS_GID_MIN/MAX = 800/899
# in the layer's etc/login.defs (a key that is set is replaced, a missing one appended; every other line is kept
# byte-identical, comments included). Refuses (exit 1, nothing written) when an id in 800-899 is already taken in the
# passwd or group of that dir OR of any other layer's dir given (the box sees the layers' union: bdrv's accounts sit
# above the main SFS's). A dir with no login.defs is left alone (exit 0). Idempotent. Only the main SFS carries
# login.defs in the base (measured: bdrv/fdrv/zdrv have none), so cut.sh runs it on the main SFS's copy with bdrv's
# etc as the other layer, container-build.sh on the skeleton; release-check's local-system-ids row reads the
# effective /etc/login.defs.
set -eu
d=${1:?usage: local-system-ids.sh <etc dir> [<etc dir of another layer>...]}
f=$d/login.defs
[ -f "$f" ] || exit 0
# already 800-899 (review cl-lemsync): a no-op, before the range check - so a re-run on a LIVE /etc, after the box's
# first useradd -r took 899, is the idempotent no-op it should be and not a "range is not free" refusal. The check
# below is for image layers at cut/build time, where no id in 800-899 may be in use.
if awk 'BEGIN { want["SYS_UID_MIN"] = 800; want["SYS_UID_MAX"] = 899; want["SYS_GID_MIN"] = 800; want["SYS_GID_MAX"] = 899 }
        $1 in want && !($1 in seen) { seen[$1] = 1; if ($2 == want[$1]) ok++ }
        END { exit ok != 4 }' "$f"; then
    exit 0
fi
for e in "$@"; do
    for a in passwd group; do
        [ -f "$e/$a" ] || continue
        if awk -F: '$3 >= 800 && $3 <= 899 { found = 1 } END { exit !found }' "$e/$a"; then
            echo "local-system-ids.sh: REFUSED: $e/$a already has an id in 800-899 (the range is not free in the image)" >&2; exit 1
        fi
    done
done
tmp=$f.ids.$$
awk 'BEGIN { want["SYS_UID_MIN"] = 800; want["SYS_UID_MAX"] = 899; want["SYS_GID_MIN"] = 800; want["SYS_GID_MAX"] = 899 }
     $1 in want && $1 !~ /^#/ { if (!done[$1]++) print $1 "\t\t" want[$1]; next }
     { print }
     END { n = split("SYS_UID_MIN SYS_UID_MAX SYS_GID_MIN SYS_GID_MAX", k, " ")
           for (i = 1; i <= n; i++) if (!done[k[i]]) print k[i] "\t\t" want[k[i]] }' "$f" > "$tmp"
# every line that is not one of the four keys is unchanged, in order
keys='^(SYS_UID_MIN|SYS_UID_MAX|SYS_GID_MIN|SYS_GID_MAX)[[:space:]]'
[ "$(grep -Ev "$keys" "$tmp")" = "$(grep -Ev "$keys" "$f")" ] || {
    rm -f "$tmp"; echo "local-system-ids.sh: REFUSED: the rewrite changed more than the four SYS_ id keys" >&2; exit 1; }
cat "$tmp" > "$f"; rm -f "$tmp"
