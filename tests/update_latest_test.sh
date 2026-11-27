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
# OS updater U2: `1bit-update --latest` installs nothing it cannot verify. update_verify runs the release verifier
# (tools/release_sign.py, which the image ships as /usr/lib/1bit/release_sign.py) on a THROWAWAY release signed here
# with a throwaway key: only the release key's signature over SHA256SUMS, naming this exact image, passes. update_prune
# keeps the download just fetched and one previous. Sourced with ONEBIT_UPDATE_LIB=1: no root, no network, no disk.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
u=$here/os/v3/puppy/embed/usr/local/bin/1bit-update
rs=${ONEBIT_TEST_RELEASE_SIGN:-$here/tools/release_sign.py}
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
[ -f "$rs" ] || { echo "update_latest_test: FAIL no release verifier at $rs"; exit 1; }
seed=$(python3 -I -c 'import os; print(os.urandom(32).hex())'); echo "$seed" > "$t/seed"
other=$(python3 -I -c 'import os; print(os.urandom(32).hex())'); echo "$other" > "$t/other"
python3 -I "$rs" pubkey "$t/seed" > "$t/pub" || { echo "update_latest_test: FAIL pubkey"; exit 1; }
mk() {  # mk <dir>: a release dir with a fake image, its SHA256SUMS, signed by the throwaway key
    mkdir -p "$1"; head -c 4096 /dev/urandom > "$1/1bit-os-v2026.50.img.zst"
    (cd "$1" && sha256sum 1bit-os-v2026.50.img.zst > SHA256SUMS)
    python3 -I "$rs" sign "$1/SHA256SUMS" --key "${2:-$t/seed}" --key-id test-key > /dev/null
}
v() {  # v <dir>: update_verify with the lib sourced, verifier and public key pointed at the test's
    ( ONEBIT_UPDATE_LIB=1 ONEBIT_UPDATE_VERIFIER=$rs ONEBIT_UPDATE_PUB=${PUB:-$t/pub} LOG=/dev/null; . "$u"
      update_verify "$1/1bit-os-v2026.50.img.zst" "$1/SHA256SUMS" "$1/SHA256SUMS.sig" ) > "$t/v.out" 2>&1
}
mk "$t/good"; v "$t/good"; check $? "a release signed by the release key, naming this image: verified (exit 0)"
mk "$t/img"; printf x >> "$t/img/1bit-os-v2026.50.img.zst"; v "$t/img"
[ $? = 2 ] && grep -q "not the released image" "$t/v.out"; check $? "an image changed after signing is refused"
mk "$t/sums"; sed -i 's/^\(.\)/0\1/; s/^0\(.\{63\}\)./\1/' "$t/sums/SHA256SUMS"; v "$t/sums"
[ $? = 2 ] && grep -q "does not match" "$t/v.out"; check $? "a SHA256SUMS changed after signing is refused"
mk "$t/nosig"; rm -f "$t/nosig/SHA256SUMS.sig"; v "$t/nosig"
[ $? = 2 ] && grep -q "no SHA256SUMS.sig: only signed releases are installed" "$t/v.out"; check $? "an unsigned release is refused"
mk "$t/wrong" "$t/other"; v "$t/wrong"
[ $? = 2 ] && grep -q "REFUSED" "$t/v.out"; check $? "a release signed by another key is refused"
PUB=$t/none v "$t/good"; [ $? = 2 ] && grep -q "no release public key" "$t/v.out"; check $? "no public key in the image: refused, not trusted"
( ONEBIT_UPDATE_LIB=1 ONEBIT_UPDATE_VERIFIER=$t/missing.py ONEBIT_UPDATE_PUB=$t/pub LOG=/dev/null; . "$u"
  update_verify "$t/good/1bit-os-v2026.50.img.zst" "$t/good/SHA256SUMS" "$t/good/SHA256SUMS.sig" ) > "$t/v.out" 2>&1
[ $? = 2 ] && grep -q "no release verifier" "$t/v.out"; check $? "no verifier in the image: refused"

# update_prune: the fetched release and ONE previous stay, older downloads go
mkdir -p "$t/upd"; for d in v2026.46 v2026.47 v2026.48 v2026.49; do mkdir -p "$t/upd/$d"; touch "$t/upd/$d/x"; sleep 1; done
( ONEBIT_UPDATE_LIB=1 ONEBIT_UPDATE_DIR=$t/upd LOG=/dev/null; . "$u"; update_prune v2026.49 ) > /dev/null 2>&1
[ "$(ls "$t/upd" | tr '\n' ' ')" = "v2026.48 v2026.49 " ]; check $? "update_prune keeps the fetched release and one previous"
# ... and leaves the CALLER's variables alone: update_latest decompresses "$_d/$_img" right after it, and a prune that
# looped with `_d` sent that to whichever old download it saw last (tools/update_vm_test.sh, 27 Nov)
mkdir -p "$t/upd2"; for d in v2026.47 v2026.48 v2026.49; do mkdir -p "$t/upd2/$d"; sleep 1; done
got=$( ONEBIT_UPDATE_LIB=1 ONEBIT_UPDATE_DIR=$t/upd2 LOG=/dev/null; . "$u"; _d=$t/upd2/v2026.49 _n=7 _keep=k
       update_prune v2026.49 > /dev/null 2>&1; printf '%s|%s|%s' "$_d" "$_n" "$_keep" )
[ "$got" = "$t/upd2/v2026.49|7|k" ]; check $? "update_prune does not change the caller's _d/_n/_keep (got ${got##*/})"

echo "update_latest_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
