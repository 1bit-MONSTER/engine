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
# stage-debs.sh + python-venv.pin: the pin is well formed (https urls on the Ubuntu archive, 64-hex sha256, sizes, every
# take names a pinned package, base paths), other bytes are refused with nothing staged, and - when the pinned .debs are
# in the cache (DEBS_CACHE or ~/.cache/1bit-cut/debs; the test never downloads) - exactly the take files are staged,
# mode 0644, and the wheels are where ensurepip looks (/usr/share/python-wheels).
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
s=$here/os/v3/puppy/stage-debs.sh pin=$here/os/v3/puppy/python-venv.pin
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }

awk '$1 == "deb" { if ($4 !~ /^https:\/\/archive\.ubuntu\.com\// || $5 !~ /^[0-9a-f]{64}$/ || $6 !~ /^[0-9]+$/) bad = 1; n++ }
     END { exit bad || n == 0 }' "$pin"
check $? "every deb names an https url on the Ubuntu archive, a 64-hex sha256 and a size"
awk '$1 == "deb" { d[$2] = 1 } $1 == "take" { if (!($2 in d)) bad = 1 } END { exit bad }' "$pin"; check $? "every take names a pinned package"
[ "$(awk '$1 == "take"' "$pin" | wc -l)" = 7 ] && grep -q '^take python3.14-venv  *usr/lib/python3.14/ensurepip/__init__.py$' "$pin" &&
    [ "$(grep -c '^take .*usr/share/python-wheels/.*\.whl$' "$pin")" = 2 ]
check $? "it takes ensurepip, the pip + setuptools wheels (in /usr/share/python-wheels) and the copyrights - 7 files"
grep -q '^base usr/bin/python3.14$' "$pin"; check $? "it names the base Python it was pinned for"
mkdir -p "$t/fake"
awk '$1 == "deb" { print $4 }' "$pin" | while read -r u; do printf 'not a deb' > "$t/fake/$(basename "$u")"; done
sh "$s" "$pin" "$t/out" --deb-dir "$t/fake" 2> "$t/err"; [ $? = 1 ] && grep -q REFUSED "$t/err" && [ ! -e "$t/out" ]
check $? "other bytes are REFUSED and nothing is staged"
c=${DEBS_CACHE:-$HOME/.cache/1bit-cut/debs}; have=1
for u in $(awk '$1 == "deb" { print $4 }' "$pin"); do [ -f "$c/$(basename "$u")" ] || have=0; done
if [ "$have" = 1 ] && command -v dpkg-deb > /dev/null; then
    sh "$s" "$pin" "$t/real" --deb-dir "$c" > /dev/null 2>&1; check $? "the pinned .debs from the cache stage"
    [ "$(cd "$t/real" && find . -type f | wc -l)" = 7 ] && [ -z "$(find "$t/real" -type f ! -perm 644)" ]; check $? "  ... exactly the 7 take files, mode 0644"
    ls "$t/real/usr/share/python-wheels/" | grep -q '^pip-.*\.whl$'; check $? "  ... a pip wheel where ensurepip looks"
else
    echo "info the pinned .debs are not all cached here: the staging arms did not run (the test never downloads)"
fi

echo "stage_debs_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
