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
# Security review (24 Nov): root-run 1bit-update used a FIXED /tmp/1bit-update (releases.json, check.err, the ESP and
# image mounts, sums) - a planted dir or symlink there decides where root writes. Now every run makes its own private
# work dir (mktemp -d, 0700: a fresh random name, nothing pre-existing is followed), and /data/updates must be a real
# root-only directory. Runs unprivileged: the planted-symlink case and the refusals; no root, no network.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
u=$here/os/v3/puppy/embed/usr/local/bin/1bit-update
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
printf '[]\n' > "$t/rel.json"; printf 'payload=lemonade-onebit-v2026.48\n' > "$t/id"
mkdir -p "$t/tmp" "$t/victim"
ln -s "$t/victim" "$t/tmp/1bit-update"                       # the planted link at the OLD fixed path
TMPDIR=$t/tmp ONEBIT_UPDATE_IDENTITY=$t/id ONEBIT_UPDATE_API=https://127.0.0.1:9/none sh "$u" --check > "$t/out" 2>&1
[ -z "$(ls -A "$t/victim")" ]; check $? "a planted symlink at the old /tmp/1bit-update path receives nothing"
[ "$(ls "$t/tmp" | grep -v '^1bit-update$' | wc -l)" = 0 ]; check $? "the run's own work dir is gone after it (nothing left in TMPDIR)"
grep -q "could not reach GitHub" "$t/out"; check $? "and --check still answers (no network: said, exit 0)"
w=$( (ONEBIT_UPDATE_LIB=1 TMPDIR=$t/tmp LOG=/dev/null; . "$u"; work_dir; echo "$WORK") )
base=$t/tmp; [ "$(id -u)" = 0 ] && [ -d /run ] && base=/run        # root (an install) works under /run
[ -d "$w" ] && [ ! -L "$w" ] && [ "$(stat -c %a "$w")" = 700 ] && case $w in "$base/1bit-update."*) true ;; *) false ;; esac
check $? "work_dir: a new 0700 directory with a random name under $base ($(basename "$w"))"
[ "$base" = /run ] && rm -rf "${w:?}"
w2=$( (ONEBIT_UPDATE_LIB=1 TMPDIR=$t/tmp LOG=/dev/null; . "$u"; work_dir; echo "$WORK") ); [ "$w" != "$w2" ]
check $? "two runs never share a work dir"
[ "$base" = /run ] && rm -rf "${w2:?}"
sd() { ( ONEBIT_UPDATE_LIB=1 LOG=/dev/null; . "$u"; safe_dir "$1" ) > "$t/sd" 2>&1; }
ln -s "$t/victim" "$t/updates-link"; sd "$t/updates-link"; [ $? = 1 ] && grep -q "is a symbolic link" "$t/sd" && [ -z "$(ls -A "$t/victim")" ]
check $? "safe_dir refuses a symlinked /data/updates, and its target is untouched"
mkdir -m 755 "$t/updates-mine"; [ "$(id -u)" = 0 ] && chown 1 "$t/updates-mine"   # as root (CI): hand it to uid 1
sd "$t/updates-mine"; rc=$?
[ $rc = 1 ] && grep -q "owned by uid" "$t/sd"; check $? "safe_dir refuses a directory not owned by root"
: > "$t/updates-file"; sd "$t/updates-file"; [ $? = 1 ] && grep -q "not a directory" "$t/sd"; check $? "safe_dir refuses a file in place of the directory"

echo "update_workdir_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
