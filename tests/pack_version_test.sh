#!/usr/bin/env bash
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
# pack_version_test.sh: what scripts/pack-version.sh maps a tag to, and the ordering a candidate needs.
#
# The ordering is the part that is easy to get wrong in a way no install proves: a candidate package
# has to install OVER the release before it and sort BELOW the release it precedes. A transform that
# reads "rc-2026-11-07-c32" as 2026.11.07.c32 gives a version below every release since week 11 of the
# year, so apt refuses the candidate as a downgrade and rpm -U does too, which is the one move a
# candidate exists for (the review of #1368 found exactly that).
#
# dpkg --compare-versions is used for the deb/rpm form (dpkg is on every runner these tests run on).
# vercmp (pacman) and rpm.labelCompare are used when they happen to be on the box, and skipped with a
# note otherwise - their answers are recorded in docs/install-any-distro.md's measured rows.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=scripts/pack-version.sh
. "$here/../scripts/pack-version.sh"

fail=0
check() { [ "$2" = "$3" ] || { echo "FAIL $1: got '$2', want '$3'"; fail=$((fail + 1)); }; }
note() { echo "  $*"; }

echo "mapping"
check "release tag"     "$(pack_version v2026.45)"               2026.45
check "release version" "$(pack_version 2026.45)"                2026.45
check "candidate, deb"  "$(pack_version rc-2026-11-07-c32 deb)"  2026.45~c32
check "candidate, rpm"  "$(pack_version rc-2026-11-07-c32 rpm)"  2026.45~c32
check "candidate, tar"  "$(pack_version rc-2026-11-07-c32 tarball)" 2026.45~c32
check "candidate, arch" "$(pack_version rc-2026-11-07-c32 arch)" 2026.45c32
check "dry run"         "$(pack_version rehearsal-private)"      0.rehearsal.private
check "unusable"        "$(pack_version --- 2>/dev/null; echo "rc=$?")" "rc=2"

# Each ordering row uses the comparator of the family it is about and is skipped, with a note, when that
# comparator is not on the box: the box that builds the packages has dpkg only, so rpm's and pacman's rows
# are also run in their own containers (the output is recorded in docs/install-any-distro.md).
echo "ordering (deb): previous release < candidate < its release"
if command -v dpkg >/dev/null; then
    order() {  # order <low> <high>
        if dpkg --compare-versions "$1" lt "$2"; then note "$1 < $2"; else echo "FAIL order: $1 !< $2"; fail=$((fail + 1)); fi
    }
    order 2026.44 2026.45~c32
    order 2026.45~c32 2026.45
    order 2026.45~c1 2026.45~c32
    # the mapping this replaced, kept as the reason the test exists
    if dpkg --compare-versions 2026.11.07.c32 lt 2026.44; then
        note "2026.11.07.c32 < 2026.44 (why the old mapping could not be installed over a release)"
    else
        echo "FAIL the old mapping no longer sorts below 2026.44 (dpkg changed? update this test)"; fail=$((fail + 1))
    fi
else
    note "dpkg not on this box: skipped"
fi

echo "ordering (pacman, when vercmp is here)"
if command -v vercmp >/dev/null; then
    archc=$(pack_version rc-2026-11-07-c32 arch)
    for pair in "2026.44 $archc" "$archc 2026.45" "2026.45c1 2026.45c32"; do
        set -- $pair
        if [ "$(vercmp "$1" "$2")" = -1 ]; then note "$1 < $2"; else echo "FAIL vercmp: $1 !< $2"; fail=$((fail + 1)); fi
    done
else
    note "vercmp not on this box: skipped (pacman's row is measured in a container, see the PR)"
fi

echo "ordering (rpm, when python3-rpm is here)"
if python3 -c 'import rpm' 2>/dev/null; then
    rpmc=$(pack_version rc-2026-11-07-c32 rpm)
    for pair in "2026.44 $rpmc" "$rpmc 2026.45"; do
        set -- $pair
        if [ "$(python3 -c 'import rpm,sys; print(rpm.labelCompare(("0",sys.argv[1],"1"),("0",sys.argv[2],"1")))' "$1" "$2")" = -1 ]; then
            note "$1 < $2"
        else
            echo "FAIL rpm.labelCompare: $1 !< $2"; fail=$((fail + 1))
        fi
    done
else
    note "python3-rpm not on this box: skipped (rpm's row is measured in a container, see the PR)"
fi

if [ "$fail" = 0 ]; then echo "pack_version: all rows PASS"; else echo "pack_version: $fail row(s) FAILED"; fi
[ "$fail" = 0 ]
