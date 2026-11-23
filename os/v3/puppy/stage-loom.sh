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
# stage-loom.sh <payload dir> [--wheel FILE] [--cards-from DIR]
#
# engine#3350 rc1 (orchestrator 23 Nov): the 1bit OS payload carries the Loom route's server and its cards, so a model
# whose route is in config/loom-routes.json runs on loom-serve once its weights are pulled. Writes, into the payload
# (cut.sh step 3 calls it before the payload is packed):
#
#   <payload>/loom/loom-serve, libiree.so, libloomc.so   from the hrx-loom wheel config/loom.config pins (wheel=,
#                                         wheel_sha256=, loom_serve_sha256=); loom-serve's RUNPATH is $ORIGIN, so
#                                         the two libraries beside it are all it loads. The rest of the wheel (the
#                                         compiler, the benchmark) is not shipped.
#   <payload>/loom/LICENSE                the wheel's own licence (Apache-2.0 WITH LLVM-exception)
#   <payload>/loom-cards/<card>/          each card os/v3/puppy/loom-cards.pin lists, every file checked by sha256 and
#                                         size; NO parameter/ (the weights): those are links into Lemonade's Hugging
#                                         Face cache, and links out of the payload are refused (pack_outside_links),
#                                         so 1bit-loom-cards makes them on the data partition at boot.
#
# --wheel FILE      use FILE instead of downloading the pinned wheel from PyPI (it must still match wheel_sha256)
# --cards-from DIR  read <DIR>/<card>/<path> instead of downloading the card's files (each must still match its pin)
# ONEBIT_LOOM_PIN=<file> and ONEBIT_LOOM_CARDS_PIN=<file> read other pin files (the test).
# Exit: 0 staged, 1 usage, 2 refused (download failed, sha256 or size mismatch, a pin incomplete).
set -eu
umask 022
prog=$(basename "$0")
here=$(cd "$(dirname "$0")" && pwd)
# the repository's config/ is three levels up from os/v3/puppy (it read os/config/loom.config, which does not exist, so every
# cut.sh from main failed here with a misleading "has no wheel": the test always set ONEBIT_LOOM_PIN and never ran this path)
pin=${ONEBIT_LOOM_PIN:-$here/../../../config/loom.config}
[ -f "$pin" ] || { echo "$prog: REFUSED: no Loom pin file at $pin (ONEBIT_LOOM_PIN names another)" >&2; exit 2; }
cards=${ONEBIT_LOOM_CARDS_PIN:-$here/loom-cards.pin}

payload= wheel= cards_from=
while [ $# -gt 0 ]; do
    case $1 in
        --wheel) wheel=${2:-}; shift 2 || { echo "$prog: --wheel needs a file" >&2; exit 1; } ;;
        --cards-from) cards_from=${2:-}; shift 2 || { echo "$prog: --cards-from needs a directory" >&2; exit 1; } ;;
        -h|--help) sed -n '/^# stage-loom.sh/,/^# Exit:/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "$prog: unknown option $1" >&2; exit 1 ;;
        *) [ -z "$payload" ] || { echo "$prog: one payload dir only" >&2; exit 1; }; payload=$1; shift ;;
    esac
done
[ -n "$payload" ] && [ -d "$payload" ] || { echo "usage: $prog <payload dir> [--wheel FILE] [--cards-from DIR]" >&2; exit 1; }

val() { sed -n "s/^$1=//p" "$pin" | head -1; }
wname=$(val wheel) wsha=$(val wheel_sha256) ssha=$(val loom_serve_sha256)
[ -n "$wname" ] && [ -n "$wsha" ] && [ -n "$ssha" ] || { echo "$prog: REFUSED: $pin has no wheel / wheel_sha256 / loom_serve_sha256" >&2; exit 2; }
command -v python3 > /dev/null || { echo "$prog: REFUSED: needs python3 to read the wheel" >&2; exit 2; }

work=$(mktemp -d)
trap 'rm -rf "${work:?}"' EXIT
if [ -z "$wheel" ]; then
    wheel=$work/$wname
    curl -fsSL --retry 3 --connect-timeout 20 -o "$wheel" "https://files.pythonhosted.org/packages/py3/h/hrx_loom/$wname" ||
        { echo "$prog: REFUSED: could not download $wname" >&2; exit 2; }
fi
[ "$(sha256sum "$wheel" | cut -c1-64)" = "$wsha" ] || { echo "$prog: REFUSED: ${wheel##*/} is not the pinned wheel (sha256 $wsha)" >&2; exit 2; }

dest=$payload/loom
rm -rf "$dest"; mkdir -p "$dest"
python3 -I - "$wheel" "$dest" << 'PY' || { echo "$prog: REFUSED: the wheel does not hold loom-serve, its two libraries and a LICENSE" >&2; exit 2; }
import sys, zipfile
wheel, dest = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(wheel)
want = {"loom-serve", "libiree.so", "libloomc.so"}
got = set()
for n in z.namelist():
    base = n.rsplit("/", 1)[-1]
    if "/scripts/" in n and base in want:
        open(dest + "/" + base, "wb").write(z.read(n)); got.add(base)
    elif ".dist-info/licenses/" in n and base == "LICENSE":
        open(dest + "/LICENSE", "wb").write(z.read(n)); got.add("LICENSE")
sys.exit(0 if got == want | {"LICENSE"} else 1)
PY
chmod 755 "$dest/loom-serve" "$dest/libiree.so" "$dest/libloomc.so"; chmod 644 "$dest/LICENSE"
[ "$(sha256sum "$dest/loom-serve" | cut -c1-64)" = "$ssha" ] ||
    { echo "$prog: REFUSED: the wheel's loom-serve is not loom_serve_sha256 $ssha" >&2; exit 2; }

n=0
cdir=$payload/loom-cards
rm -rf "$cdir"
while read -r kind a b c; do
    case $kind in
        card) name=$a repo=$b rev=$c; mkdir -p "$cdir/$name"; n=$((n + 1)) ;;
        [0-9a-f]*)
            [ -n "${name:-}" ] || { echo "$prog: REFUSED: $cards lists a file before any card line" >&2; exit 2; }
            sha=$kind size=$a path=$b f=$cdir/$name/$path
            mkdir -p "$(dirname "$f")"
            if [ -n "$cards_from" ]; then cp "$cards_from/$name/$path" "$f" 2> /dev/null || { echo "$prog: REFUSED: no $cards_from/$name/$path" >&2; exit 2; }
            else curl -fsSL --retry 3 --connect-timeout 20 -o "$f" "https://huggingface.co/$repo/resolve/$rev/$path" ||
                { echo "$prog: REFUSED: could not download $repo@$rev/$path" >&2; exit 2; }; fi
            [ "$(stat -c %s "$f")" = "$size" ] && [ "$(sha256sum "$f" | cut -c1-64)" = "$sha" ] ||
                { echo "$prog: REFUSED: $name/$path is not its pinned sha256/size" >&2; exit 2; }
            chmod 644 "$f" ;;
    esac
done << EOF
$(grep -vE '^[[:space:]]*(#|$)' "$cards")
EOF
[ "$n" -gt 0 ] || { echo "$prog: REFUSED: $cards names no card" >&2; exit 2; }
chmod -R go-w "$dest" "$cdir"
echo "$prog: loom-serve ($(val version), ${wsha%"${wsha#????????????}"}) + $n card(s) staged into $payload" >&2
exit 0
