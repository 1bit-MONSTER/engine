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
# stage-radv.sh <out dir> [--deb <local mesa-vulkan-drivers .deb>]
#
# Puts the files os/v3/puppy/radv.pin names (Mesa's RADV driver, its ICD file, its copyright) under <out dir> at their
# image paths, from the pinned .deb: a local copy (--deb), else $RADV_CACHE (default ~/.cache/1bit-cut/radv) when it
# already holds it, else the pin's URL. The .deb must have the pin's size and sha256 or nothing is staged (exit 1).
# Extracted with `dpkg-deb -x` (no maintainer scripts run) into a temporary dir; only the `take` paths are copied, with
# mode 0644. cut.sh maps them into the main SFS; container-build.sh stages them into the rootfs skeleton.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: stage-radv.sh <out dir> [--deb <file>]}
shift
deb=
[ "${1:-}" = --deb ] && deb=${2:?--deb needs a file}
pin=$here/radv.pin
val() { awk -v k="$1" '$1 == k { print $2; exit }' "$pin"; }
url=$(val url) sha=$(val sha256) size=$(val size)
[ -n "$url" ] && [ -n "$sha" ] && [ -n "$size" ] || { echo "stage-radv.sh: $pin names no url/sha256/size" >&2; exit 1; }
if [ -z "$deb" ]; then
    cache=${RADV_CACHE:-$HOME/.cache/1bit-cut/radv}; mkdir -p "$cache"; deb=$cache/$(basename "$url")
    if [ ! -f "$deb" ] || [ "$(sha256sum "$deb" | cut -c1-64)" != "$sha" ]; then
        curl -fsSL --proto =https --proto-redir =https --tlsv1.2 --retry 3 -o "$deb.part" "$url" || { rm -f "$deb.part"; echo "stage-radv.sh: could not download $url" >&2; exit 1; }
        mv -f "$deb.part" "$deb"
    fi
fi
[ "$(stat -c %s "$deb")" = "$size" ] && [ "$(sha256sum "$deb" | cut -c1-64)" = "$sha" ] ||
    { echo "stage-radv.sh: REFUSED: $deb is not the pinned mesa-vulkan-drivers (size $size, sha256 $sha)" >&2; exit 1; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
dpkg-deb -x "$deb" "$tmp"
for f in $(awk '$1 == "take" { print $2 }' "$pin"); do
    [ -f "$tmp/$f" ] || { echo "stage-radv.sh: the package has no $f" >&2; exit 1; }
    install -D -m 644 "$tmp/$f" "$out/$f"
done
# the ICD names the library by its absolute path (see radv.pin): no dependence on an ld.so.cache nobody rebuilds
icd=$out/usr/share/vulkan/icd.d/radeon_icd.json
grep -q '"library_path": "libvulkan_radeon.so"' "$icd" || { echo "stage-radv.sh: $icd is not the ICD file this expects" >&2; exit 1; }
sed -i 's|"library_path": "libvulkan_radeon.so"|"library_path": "/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so"|' "$icd"
echo "stage-radv.sh: RADV $(val version) (sha256 $(printf '%.16s' "$sha")...) staged in $out"
