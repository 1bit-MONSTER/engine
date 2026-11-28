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
# stage-radv.sh + radv.pin: the pin is well formed (url, 64-hex sha256, size, take paths: RADV + ICD + copyright, never
# lavapipe), other bytes are refused with nothing staged, and - when the pinned .deb is in the cache (RADV_CACHE or
# ~/.cache/1bit-cut/radv; the test never downloads) - exactly the three files are staged, mode 0644.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
s=$here/os/v3/puppy/stage-radv.sh pin=$here/os/v3/puppy/radv.pin
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
v() { awk -v k="$1" '$1 == k { print $2; exit }' "$pin"; }

printf '%s' "$(v sha256)" | grep -qxE '[0-9a-f]{64}' && [ "$(v size)" -gt 1000000 ] && case "$(v url)" in https://archive.ubuntu.com/*) true ;; *) false ;; esac
check $? "the pin names an https url on the Ubuntu archive, a 64-hex sha256 and a size"
[ "$(v base_mesa)" = "libgallium-$(v version).so" ]; check $? "base_mesa names the libgallium of the same Mesa build as the package"
[ "$(basename "$(v url)")" = "$(v package)_$(v version)_amd64.deb" ]; check $? "the url's file is <package>_<version>_amd64.deb"
takes=$(awk '$1 == "take" { print $2 }' "$pin" | sort | tr '\n' ' ')
[ "$takes" = "usr/lib/x86_64-linux-gnu/libvulkan_radeon.so usr/share/doc/mesa-vulkan-drivers/copyright usr/share/vulkan/icd.d/radeon_icd.json " ]
check $? "it takes exactly RADV, its ICD file and the copyright"
! grep -qE '^take .*(lvp|lavapipe|llvmpipe)' "$pin"; check $? "never lavapipe (a CPU Vulkan driver would bring the CPU speed back)"
printf 'not a deb' > "$t/bad.deb"
sh "$s" "$t/out" --deb "$t/bad.deb" 2> "$t/err"; [ $? = 1 ] && grep -q REFUSED "$t/err" && [ ! -e "$t/out" ]
check $? "other bytes are REFUSED and nothing is staged"
c=${RADV_CACHE:-$HOME/.cache/1bit-cut/radv}/$(basename "$(v url)")
if [ -f "$c" ] && command -v dpkg-deb > /dev/null; then
    sh "$s" "$t/real" --deb "$c" > /dev/null 2>&1; check $? "the pinned .deb from the cache stages"
    [ "$(cd "$t/real" && find . -type f | sort | tr '\n' ' ')" = "./usr/lib/x86_64-linux-gnu/libvulkan_radeon.so ./usr/share/doc/mesa-vulkan-drivers/copyright ./usr/share/vulkan/icd.d/radeon_icd.json " ] &&
        [ -z "$(find "$t/real" -type f ! -perm 644)" ]
    check $? "  ... exactly the three files, mode 0644"
    grep -q '"library_path": "/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so"' "$t/real/usr/share/vulkan/icd.d/radeon_icd.json"
    check $? "  ... the ICD file names the library by its ABSOLUTE path (no ld.so.cache dependence)"
else
    echo "info the pinned .deb is not cached here: the staging arm did not run (the test never downloads)"
fi

echo "stage_radv_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
