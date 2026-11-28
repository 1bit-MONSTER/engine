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
# stage-debs.sh <pin file> <out dir> [--deb-dir <dir of local .debs>]
#
# Puts the `take` files of a pin file's `deb` packages (os/v3/puppy/python-venv.pin's format) under <out dir> at their
# image paths. Each .deb comes from --deb-dir, else $DEBS_CACHE (default ~/.cache/1bit-cut/debs) when it is already
# there, else its URL (https only); it must have the pin's size and sha256, or nothing is staged (exit 1). Extracted with
# `dpkg-deb -x` (no maintainer scripts run) into a temporary dir; only the take paths are copied, mode 0644.
set -eu
pin=${1:?usage: stage-debs.sh <pin file> <out dir> [--deb-dir <dir>]}
out=${2:?usage: stage-debs.sh <pin file> <out dir> [--deb-dir <dir>]}
shift 2
dd=
[ "${1:-}" = --deb-dir ] && dd=${2:?--deb-dir needs a dir}
cache=${DEBS_CACHE:-$HOME/.cache/1bit-cut/debs}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
awk '$1 == "deb" { print $2, $4, $5, $6 }' "$pin" > "$tmp/debs"
[ -s "$tmp/debs" ] || { echo "stage-debs.sh: $pin names no deb" >&2; exit 1; }
while read -r name url sha size; do
    case $url in https://*) ;; *) echo "stage-debs.sh: $name: not an https url: $url" >&2; exit 1 ;; esac
    f=$(basename "$url")
    if [ -n "$dd" ]; then deb=$dd/$f
    else
        mkdir -p "$cache"; deb=$cache/$f
        if [ ! -f "$deb" ] || [ "$(sha256sum "$deb" | cut -c1-64)" != "$sha" ]; then
            curl -fsSL --proto =https --proto-redir =https --tlsv1.2 --retry 3 -o "$deb.part" "$url" ||
                { rm -f "$deb.part"; echo "stage-debs.sh: could not download $url" >&2; exit 1; }
            mv -f "$deb.part" "$deb"
        fi
    fi
    [ -f "$deb" ] && [ "$(stat -c %s "$deb")" = "$size" ] && [ "$(sha256sum "$deb" | cut -c1-64)" = "$sha" ] ||
        { echo "stage-debs.sh: REFUSED: $deb is not the pinned $name (size $size, sha256 $sha)" >&2; exit 1; }
    mkdir -p "$tmp/x/$name"; dpkg-deb -x "$deb" "$tmp/x/$name"
done < "$tmp/debs"
mkdir -p "$out"
awk '$1 == "take" { print $2, $3 }' "$pin" | while read -r name path; do
    [ -f "$tmp/x/$name/$path" ] || { echo "stage-debs.sh: $name has no $path" >&2; exit 1; }
    install -D -m 644 "$tmp/x/$name/$path" "$out/$path"
done
echo "stage-debs.sh: $(awk '$1 == "take"' "$pin" | wc -l) file(s) from $(wc -l < "$tmp/debs") pinned package(s) staged in $out"
