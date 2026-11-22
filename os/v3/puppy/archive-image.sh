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
# archive-image.sh <image.img> <tag> [<archive root>]
#
# Keeps the exact image that is tested and shipped (engine#3287): the v2026.47 image lived in a build directory
# outside the backup set and was lost when the build host was reinstalled, so the release image had to be
# reassembled from checksum-verified parts. This copies <image.img> to <archive root>/<tag>/1bit-os-puppy-<tag>.img
# (default root: $ONEBIT_IMAGE_ARCHIVE, else ~/release-archive), writes its sha256 beside it, re-reads the COPY to
# verify it, and makes both read-only. Extra files named in ARCHIVE_EXTRA (space-separated paths: the probe log,
# BUILT-FROM.txt, the payload listing) are copied beside it.
#
# It refuses to overwrite: an archived tag is a fact, and a re-cut is a new tag. The stick is then written FROM
# the archived path, so the bytes tested are the bytes kept. Prints the archived image's path on stdout.
set -euo pipefail
prog=$(basename "$0")
img=${1:?usage: archive-image.sh <image.img> <tag> [<archive root>]}
tag=${2:?usage: archive-image.sh <image.img> <tag> [<archive root>]}
root=${3:-${ONEBIT_IMAGE_ARCHIVE:-$HOME/release-archive}}
[[ "$tag" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "$prog: tag '$tag' is not a plain name" >&2; exit 2; }
[ -f "$img" ] || { echo "$prog: no image at $img" >&2; exit 2; }
d=$root/$tag; a=$d/1bit-os-puppy-$tag.img
[ ! -e "$a" ] || { echo "$prog: REFUSED: $a already exists (an archived tag is not overwritten; cut a new tag)" >&2; exit 1; }
mkdir -p "$d"
src=$(sha256sum "$img" | cut -c1-64)
need=$(stat -c %s "$img"); free=$(( $(df -P --block-size=1 "$d" | awk 'NR==2 {print $4}') ))
[ "$free" -gt "$need" ] || { echo "$prog: REFUSED: $d has $free B free, the image is $need B" >&2; exit 1; }
cp --reflink=auto "$img" "$a.part"
got=$(sha256sum "$a.part" | cut -c1-64)
[ "$got" = "$src" ] || { rm -f "$a.part"; echo "$prog: REFUSED: the copy hashes to $got, the image to $src" >&2; exit 1; }
mv "$a.part" "$a"
(cd "$d" && echo "$src  $(basename "$a")" > "$(basename "$a").sha256")
for f in ${ARCHIVE_EXTRA:-}; do [ -f "$f" ] && cp "$f" "$d/"; done
chmod a-w "$a" "$a.sha256"
echo "$prog: archived $tag: $a ($need B, sha256 $src)" >&2
echo "$a"
