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
#   ver=$(pack_version "$tag") || exit 2
#
# Every family needs a version that starts with a digit: a leading "v" goes, anything outside
# [A-Za-z0-9._] becomes "." (rpm and pacman refuse "-" in a version), and what precedes the first digit
# goes too, so the weekly's candidate tag rc-2026-11-07-c32 packs as 2026.11.07.c32. A tag with no
# digit at all - docs/releases.md's `rehearsal-private` dry run - packs as 0.rehearsal.private, a number
# no release can have.

pack_version() {
    local raw=${1#v} sanitized ver
    sanitized=$(printf '%s' "$raw" | sed -e 's/[^A-Za-z0-9._]\+/./g' -e 's/^\.*//' -e 's/\.*$//')
    [ -n "$sanitized" ] || {
        echo "pack-version: '$1' has nothing usable in it (letters, digits, '.', '_')" >&2
        return 2; }
    ver=${sanitized#"${sanitized%%[0-9]*}"}
    [ -n "$ver" ] || ver=0.$sanitized
    printf '%s\n' "$ver"
}
