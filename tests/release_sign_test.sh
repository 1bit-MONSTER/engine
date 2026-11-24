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
# release_sign_test.sh - tools/release_sign.py, the signed SHA256SUMS of a 1bit OS release (OS-UPDATER item 3): its
# self-test, then the CLI exit codes the in-system updater relies on (0 = install it, anything else = refuse), with a
# throwaway key (the RFC 8032 test seed), raw and hex key files. Python 3 standard library only, no network.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
tool=$root/tools/release_sign.py
command -v python3 > /dev/null || { echo "release_sign_test: skip (no python3)"; exit 77; }
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
bad=0
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
py() { python3 -I "$tool" "$@"; }

py --self-test > "$t/self" 2>&1; check $? "self-test ($(tail -1 "$t/self"))"
python3 -I "$root/tests/release_sign_toctou_check.py" "$root/tools" > "$t/toctou" 2>&1
check $? "SHA256SUMS read once: a file swapped between reads cannot pass; y >= p refused ($(grep -c '^ok' "$t/toctou") ok)"
printf '9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60\n' > "$t/seed.hex"
python3 -c "import sys; open(sys.argv[1],'wb').write(bytes.fromhex(open(sys.argv[2]).read().strip()))" "$t/seed.raw" "$t/seed.hex"
py pubkey "$t/seed.raw" > "$t/pub.hex"; python3 -c "import sys; open(sys.argv[1],'wb').write(bytes.fromhex(open(sys.argv[2]).read().strip()))" "$t/pub.raw" "$t/pub.hex"
head -c 2097152 /dev/urandom > "$t/1bit-os-puppy-vT.img"
( cd "$t" && sha256sum 1bit-os-puppy-vT.img > SHA256SUMS )
py sign "$t/SHA256SUMS" --key "$t/seed.raw" --key-id test-v1 > /dev/null 2>&1; check $? "sign with a raw 32-byte seed (kmanifest's npu.sec form)"
[ -f "$t/SHA256SUMS.sig" ] && ! grep -q 9d61b19d "$t/SHA256SUMS.sig"; check $? "the .sig carries no part of the seed"
py verify "$t/SHA256SUMS" "$t/SHA256SUMS.sig" --pub "$t/pub.raw" > /dev/null 2>&1; check $? "verify: exit 0 with the release key (raw)"
py verify-image "$t/1bit-os-puppy-vT.img" "$t/SHA256SUMS" "$t/SHA256SUMS.sig" --pub "$t/pub.hex" > /dev/null 2>&1
check $? "verify-image: exit 0 for the released image (hex key)"
# flip one byte (XOR 0xff): writing a fixed 'X' left the random image unchanged whenever byte 4096 already was 'X'
cp "$t/1bit-os-puppy-vT.img" "$t/x.img"
python3 -I -c 'import sys; f = open(sys.argv[1], "r+b"); f.seek(4096); b = f.read(1); f.seek(4096); f.write(bytes([b[0] ^ 0xff]))' "$t/1bit-os-puppy-vT.img"
cmp -s "$t/x.img" "$t/1bit-os-puppy-vT.img" && { echo "FAIL the tampered image is not changed"; bad=$((bad + 1)); }
py verify-image "$t/1bit-os-puppy-vT.img" "$t/SHA256SUMS" "$t/SHA256SUMS.sig" --pub "$t/pub.raw" > /dev/null 2> "$t/err"
[ $? = 1 ] && grep -q "not the released image" "$t/err"; check $? "verify-image: exit 1 for a tampered image, and says so"
mv "$t/x.img" "$t/1bit-os-puppy-vT.img"
# change the digest so it ALWAYS differs: replacing the first character with a fixed "0" left the file identical (and
# its signature valid) whenever the random image's sha256 already began with 0 - 1 run in 16 (CI run 45317846195)
sed 's/^0/1/;t;s/^./0/' "$t/SHA256SUMS" > "$t/SUMS2"; cp "$t/SHA256SUMS.sig" "$t/SUMS2.sig"
cmp -s "$t/SHA256SUMS" "$t/SUMS2" && { echo "FAIL the changed SHA256SUMS is not changed"; bad=$((bad + 1)); }
py verify "$t/SUMS2" "$t/SUMS2.sig" --pub "$t/pub.raw" > /dev/null 2>&1; [ $? = 1 ]; check $? "verify: exit 1 for a SHA256SUMS changed after signing"
printf '%064x\n' 1 > "$t/other.hex"
py verify "$t/SHA256SUMS" "$t/SHA256SUMS.sig" --pub "$t/other.hex" > /dev/null 2>&1; [ $? = 1 ]; check $? "verify: exit 1 against another public key"
py verify "$t/SHA256SUMS" "$t/missing.sig" --pub "$t/pub.raw" > /dev/null 2>&1; [ $? = 1 ]; check $? "verify: exit 1 when the .sig is missing"
echo "release_sign_test: $([ $bad = 0 ] && echo ok || echo "$bad FAIL")"
[ $bad = 0 ]
