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
# stage-node.sh <root> [--tarball FILE] [--manifest]
#
# engine#3310: stage the pinned Node.js (os/v3/puppy/pin.txt: node_version, node_url, node_sha256) into an image
# root, so `lemonade launch <agent>` can follow its own install hint (`npm i -g <agent>`) on the appliance. Writes:
#
#   <root>/usr/local/lib/nodejs/          the official linux-x64 release tree, unmodified, minus include/ (67 MB of
#                                         headers for native addons: the appliance has no compiler to build one),
#                                         CHANGELOG.md, README.md and share/doc
#   <root>/usr/local/lib/nodejs/etc/npmrc npm's GLOBAL config (npm reads <its prefix>/etc/npmrc): prefix=/data/npm-global,
#                                         so `npm i -g` lands on the data partition on a live stick (whose root is
#                                         RAM) and on an installed disk alike; no update notifier, no funding lines
#   <root>/usr/local/bin/{node,npm,npx,corepack}   small exec wrappers into the tree. Not symlinks: patch-sfs.sh
#                                         copies a single-file target with `install`, which follows a link and would
#                                         turn npm's own relative symlink into a copy that cannot find its modules
#
# The release's LICENSE (Node.js MIT plus every bundled dependency's notice) stays in the tree, where the release
# puts it; os/licenses/nodejs/ carries the same file and the SOURCE record.
#
# --tarball FILE  use FILE instead of downloading node_url (it must still match node_sha256)
# --manifest      after staging, print the patch-sfs.sh manifest lines (`<path-in-sfs>\t<source>[\t<mode>]`) for the
#                 staged paths, for a cut that patches an existing SFS instead of building one
# ONEBIT_NODE_PIN=<file> reads another pin file (the test uses this).
#
# Exit: 0 staged, 1 usage, 2 refused (download failed, sha256 mismatch, unexpected tarball layout).
set -eu
umask 022                 # an image path may not be group/other-writable (base-modes, engine#1474)
prog=$(basename "$0")
here=$(cd "$(dirname "$0")" && pwd)
pin=${ONEBIT_NODE_PIN:-$here/pin.txt}

root= tarball= manifest=
while [ $# -gt 0 ]; do
    case $1 in
        --tarball) tarball=${2:-}; shift 2 || { echo "$prog: --tarball needs a file" >&2; exit 1; } ;;
        --manifest) manifest=1; shift ;;
        -h|--help) sed -n '/^# stage-node.sh/,/^# Exit:/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "$prog: unknown option $1" >&2; exit 1 ;;
        *) [ -z "$root" ] || { echo "$prog: one root only" >&2; exit 1; }; root=$1; shift ;;
    esac
done
[ -n "$root" ] || { echo "usage: $prog <root> [--tarball FILE] [--manifest]" >&2; exit 1; }

ver=$(sed -n 's/^node_version=//p' "$pin" | head -1)
url=$(sed -n 's/^node_url=//p' "$pin" | head -1)
sha=$(sed -n 's/^node_sha256=//p' "$pin" | head -1)
[ -n "$ver" ] && [ -n "$url" ] && [ -n "$sha" ] || { echo "$prog: REFUSED: $pin has no node_version/node_url/node_sha256" >&2; exit 2; }

work=$(mktemp -d)
trap 'rm -rf "${work:?}"' EXIT
if [ -z "$tarball" ]; then
    tarball=$work/${url##*/}
    curl -fsSL "$url" -o "$tarball" || { echo "$prog: REFUSED: could not download $url" >&2; exit 2; }
fi
[ -f "$tarball" ] || { echo "$prog: REFUSED: no tarball at $tarball" >&2; exit 2; }
got=$(sha256sum "$tarball" | cut -d' ' -f1)
[ "$got" = "$sha" ] || { echo "$prog: REFUSED: ${tarball##*/} is sha256 $got, the pin says $sha" >&2; exit 2; }

mkdir -p "$work/x"
tar -xf "$tarball" -C "$work/x" --no-same-owner      # GNU tar reads the compression from the file (the release is .xz)
top=$work/x/node-$ver-linux-x64
[ -x "$top/bin/node" ] && [ -f "$top/lib/node_modules/npm/bin/npm-cli.js" ] && [ -f "$top/LICENSE" ] || {
    echo "$prog: REFUSED: the tarball has no node-$ver-linux-x64/{bin/node,lib/node_modules/npm,LICENSE}" >&2; exit 2; }
rm -rf "$top/include" "$top/CHANGELOG.md" "$top/README.md" "$top/share/doc"

dest=$root/usr/local/lib/nodejs
rm -rf "$dest"
mkdir -p "$root/usr/local/lib" "$root/usr/local/bin"
mv "$top" "$dest"
mkdir -p "$dest/etc"
cat > "$dest/etc/npmrc" <<'NPMRC'
; 1bit OS (engine#3310): global installs go to the data partition, so an agent installed with `npm i -g` survives a
; reboot on a live stick (RAM root) and on an installed disk. /data/npm-global/bin is on PATH (/etc/profile.d).
prefix=/data/npm-global
update-notifier=false
fund=false
NPMRC
chmod 644 "$dest/etc/npmrc"

# npm finds its GLOBAL config at <global prefix>/etc/npmrc, and takes the global prefix from $PREFIX when that is set.
# Puppy's /etc/profile exports PREFIX=/usr, so in a login shell npm read /usr/etc/npmrc (absent), never saw
# prefix=/data/npm-global, and `npm i -g` wrote to /usr - RAM on a live stick (measured in the v2026.48 QEMU boot:
# `npm prefix -g` = /usr, release-check `node` FAIL). The npm and npx wrappers therefore name the global config
# file itself; PREFIX stays as Puppy set it for everything else, and a caller's own npm_config_globalconfig wins.
gcfg='export npm_config_globalconfig="${npm_config_globalconfig:-/usr/local/lib/nodejs/etc/npmrc}"'
for b in node npm npx corepack; do
    t=/usr/local/lib/nodejs/bin/$b
    [ -e "$dest/bin/$b" ] || { echo "$prog: REFUSED: the release has no bin/$b" >&2; exit 2; }
    case $b in npm|npx) env_line="$gcfg\n" ;; *) env_line= ;; esac
    printf "#!/bin/sh\n# engine#3310: Node.js %s, staged by os/v3/puppy/stage-node.sh\n${env_line}exec %s \"\$@\"\n" "$ver" "$t" \
        > "$root/usr/local/bin/$b"
    chmod 755 "$root/usr/local/bin/$b"
done
# nothing in an image may be group/other-writable (base-modes, engine#1474)
chmod -R go-w "$dest"

echo "$prog: Node.js $ver staged ($(du -sh "$dest" | cut -f1) in /usr/local/lib/nodejs, sha256 $(printf '%s' "$sha" | cut -c1-16))" >&2
if [ -n "$manifest" ]; then
    printf 'usr/local/lib/nodejs\t%s\n' "$dest"
    for b in node npm npx corepack; do printf 'usr/local/bin/%s\t%s\t755\n' "$b" "$root/usr/local/bin/$b"; done
fi
exit 0
