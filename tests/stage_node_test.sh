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
# engine#3310: os/v3/puppy/stage-node.sh on a FAKE release tarball: the node-vX-linux-x64 layout, gzip-compressed (the CI
# container has gzip and no xz; stage-node.sh lets tar read the compression, so the real .xz goes the same path), with a
# shell script standing in for bin/node. No network, no 32 MB download: the sha256 refusal, the tree and what is left
# out, npm's global prefix, the exec wrappers, the modes, and the patch-sfs manifest lines. Also checks that pin.txt's
# real Node pin is complete.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
stage=$here/os/v3/puppy/stage-node.sh
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
command -v gzip > /dev/null || { echo "stage_node_test: no gzip: skipped"; exit 77; }

ver=v0.0.1
r=$t/node-$ver-linux-x64
mkdir -p "$r/bin" "$r/lib/node_modules/npm/bin" "$r/lib/node_modules/corepack/dist" "$r/include/node" "$r/share/doc/node" \
    "$r/share/man/man1"
printf '#!/bin/sh\necho %s\n' "$ver" > "$r/bin/node"; chmod 755 "$r/bin/node"
printf '// npm\n' > "$r/lib/node_modules/npm/bin/npm-cli.js"; printf '// npx\n' > "$r/lib/node_modules/npm/bin/npx-cli.js"
printf '// corepack\n' > "$r/lib/node_modules/corepack/dist/corepack.js"
ln -s ../lib/node_modules/npm/bin/npm-cli.js "$r/bin/npm"; ln -s ../lib/node_modules/npm/bin/npx-cli.js "$r/bin/npx"
ln -s ../lib/node_modules/corepack/dist/corepack.js "$r/bin/corepack"
printf 'MIT\n' > "$r/LICENSE"; printf 'x\n' > "$r/CHANGELOG.md"; printf 'x\n' > "$r/README.md"; printf 'h\n' > "$r/include/node/v8.h"
chmod 775 "$r/lib"                                         # a group-writable directory the stage must normalise
tar -czf "$t/node.tar.gz" -C "$t" "node-$ver-linux-x64"
sha=$(sha256sum "$t/node.tar.gz" | cut -d' ' -f1)
printf 'node_version=%s\nnode_url=https://invalid.example/node.tar.xz\nnode_sha256=%s\n' "$ver" "$sha" > "$t/pin.txt"
printf 'node_version=%s\nnode_url=https://invalid.example/node.tar.xz\nnode_sha256=%s\n' "$ver" \
    0000000000000000000000000000000000000000000000000000000000000000 > "$t/badpin.txt"

ONEBIT_NODE_PIN=$t/badpin.txt sh "$stage" "$t/root0" --tarball "$t/node.tar.gz" > "$t/out0" 2>&1
[ $? = 2 ]; check $? "a tarball whose sha256 is not the pin's is refused (exit 2)"
[ ! -e "$t/root0/usr/local/lib/nodejs" ]; check $? "nothing is staged from a refused tarball"

ONEBIT_NODE_PIN=$t/pin.txt sh "$stage" "$t/root" --tarball "$t/node.tar.gz" --manifest > "$t/man" 2> "$t/err"
check $? "the pinned tarball stages (exit 0)"
n=$t/root/usr/local/lib/nodejs
[ -x "$n/bin/node" ] && [ -L "$n/bin/npm" ] && [ -f "$n/lib/node_modules/npm/bin/npm-cli.js" ] && [ -f "$n/LICENSE" ]
check $? "the release tree is in /usr/local/lib/nodejs with npm's own relative symlink and the LICENSE"
[ ! -e "$n/include" ] && [ ! -e "$n/CHANGELOG.md" ] && [ ! -e "$n/README.md" ] && [ ! -e "$n/share/doc" ] && [ -d "$n/share/man" ]
check $? "include/, CHANGELOG.md, README.md and share/doc are left out, the man pages kept"
grep -qx 'prefix=/data/npm-global' "$n/etc/npmrc"; check $? "npm's global config puts the prefix on /data"
ok=0
for b in node npm npx corepack; do
    w=$t/root/usr/local/bin/$b
    [ -f "$w" ] && [ ! -L "$w" ] && [ "$(stat -c %a "$w")" = 755 ] && grep -qx "exec /usr/local/lib/nodejs/bin/$b \"\$@\"" "$w" || ok=1
done
check $ok "node, npm, npx, corepack are 0755 exec wrappers (regular files) into the tree"
# engine#3310 follow-up: Puppy's /etc/profile exports PREFIX=/usr, which moves npm's global config to /usr/etc/npmrc and
# its global prefix to /usr (RAM on a stick). npm and npx name the global config themselves, a caller's own wins.
ok=0
for b in npm npx; do
    g=$(PREFIX=/usr; unset npm_config_globalconfig; eval "$(grep '^export npm_config_globalconfig=' "$t/root/usr/local/bin/$b")"; echo "${npm_config_globalconfig:-}")
    [ "$g" = /usr/local/lib/nodejs/etc/npmrc ] || ok=1
    g=$(npm_config_globalconfig=/elsewhere/npmrc; eval "$(grep '^export npm_config_globalconfig=' "$t/root/usr/local/bin/$b")"; echo "$npm_config_globalconfig")
    [ "$g" = /elsewhere/npmrc ] || ok=1
done
check $ok "npm and npx name /usr/local/lib/nodejs/etc/npmrc as the global config (PREFIX=/usr cannot move it), a caller's own wins"
! grep -q npm_config_globalconfig "$t/root/usr/local/bin/node" "$t/root/usr/local/bin/corepack"
check $? "node and corepack wrappers stay plain exec wrappers"
sh -n "$t/root/usr/local/bin/npm" && sh -n "$t/root/usr/local/bin/npx"; check $? "the npm and npx wrappers parse as sh"
[ -z "$(find "$t/root" \( -perm -0020 -o -perm -0002 \) ! -type l)" ]; check $? "nothing staged is group- or other-writable"
[ "$(wc -l < "$t/man")" = 5 ] && grep -q "^usr/local/lib/nodejs	$n\$" "$t/man" && grep -q "^usr/local/bin/npm	.*	755\$" "$t/man"
check $? "--manifest prints the five patch-sfs lines (the tree, four wrappers with mode 755)"

pin=$here/os/v3/puppy/pin.txt
v=$(sed -n 's/^node_version=//p' "$pin"); u=$(sed -n 's/^node_url=//p' "$pin"); s=$(sed -n 's/^node_sha256=//p' "$pin")
case "$u" in "https://nodejs.org/dist/$v/node-$v-linux-x64.tar.xz") ok=0 ;; *) ok=1 ;; esac
printf '%s' "$s" | grep -qE '^[0-9a-f]{64}$' || ok=1
check $ok "pin.txt pins an official nodejs.org linux-x64 release with a 64-hex sha256 ($v)"

echo "stage_node_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
