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
# pack-version.sh: the version a package is built with, from the tag a cut carries. Sourced by
# scripts/pack-distro.sh (which builds the names) and scripts/pack-proof.sh (which looks the packages
# up), so the two cannot disagree about a name - which is how #1279's candidate tag went unbuilt: the
# packer refused `rc-2026-11-07-c32` and the proof looked for a name the packer would never produce.
#
#   . "$(dirname "$0")/pack-version.sh"
#   ver=$(pack_version "$tag" deb) || exit 2
#
# A release tag is already a version and comes back as it stands (v2026.45 -> 2026.45; a leading "v"
# goes and anything outside [A-Za-z0-9._] becomes ".", since rpm and pacman refuse "-").
#
# A CANDIDATE tag is not a version, and the transform is not cosmetic: a candidate has to install over
# the release before it and sort below the release it precedes, or installing it is a downgrade that
# apt and rpm refuse by default. So a candidate is mapped into the ISO week of its own date - the
# release it is a candidate for - with a suffix that sorts below that release:
#
#   rc-2026-11-07-c32 -> 2026.45~c32   (deb, rpm, tarball: "~" sorts below everything)
#   rc-2026-11-07-c32 -> 2026.45c32    (arch: pacman gives "~" no special meaning, so it sorts the
#                                       suffix below 2026.45 the way vercmp sorts alpha before
#                                       numeric - measured, not assumed, in tests/pack_version_test.sh)
#
# giving 2026.44 < 2026.45~c32 < 2026.45 for both. The week comes from `date -d 2026-11-07 +%G.%V`.
#
# A tag with no digit at all (docs/releases.md's `rehearsal-private` dry run) becomes 0.<tag>: below
# every release, which is what a rehearsal should be. A tag that looks like a candidate but cannot be
# read as one is mapped that way too, with a warning - it must not quietly sort as a real version.

pack_version() {
    local raw=${1#v} fam=${2:-deb} sanitized base suffix y m d
    sanitized=$(printf '%s' "$raw" | sed -e 's/[^A-Za-z0-9._]\+/./g' -e 's/^\.*//' -e 's/\.*$//')
    [ -n "$sanitized" ] || {
        echo "pack-version: '$1' has nothing usable in it (letters, digits, '.', '_')" >&2
        return 2; }
    case $sanitized in
        [0-9]*) printf '%s\n' "$sanitized"; return 0 ;;   # a release tag, or a version handed in
    esac
    if [[ $sanitized =~ ^rc\.([0-9]{4})\.([0-9]{2})\.([0-9]{2})\.(c[0-9]+)$ ]]; then
        y=${BASH_REMATCH[1]} m=${BASH_REMATCH[2]} d=${BASH_REMATCH[3]} suffix=${BASH_REMATCH[4]}
        if base=$(date -d "$y-$m-$d" +%G.%V 2>/dev/null); then
            case $fam in
                arch) printf '%s%s\n' "$base" "$suffix" ;;
                *) printf '%s~%s\n' "$base" "$suffix" ;;
            esac
            return 0
        fi
        echo "pack-version: '$1' looks like a candidate but $y-$m-$d is not a date; packing 0.$sanitized, which sorts below every release" >&2
    else
        case $sanitized in
            rc.*) echo "pack-version: '$1' is not of the form rc-<YYYY-MM-DD>-c<N>; packing 0.$sanitized, which sorts below every release" >&2 ;;
        esac
    fi
    printf '0.%s\n' "$sanitized"
}
