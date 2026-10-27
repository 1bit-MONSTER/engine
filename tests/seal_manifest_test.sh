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
# The signed manifest of a sealed NPU kernel set (engine #574): tools/seal_manifest.py writes
# <set>/manifest.json (every file the set ships, with the sha256 of its ciphertext) and
# <set>/manifest.sig (Ed25519 over the manifest.json bytes), and checks a set against it.
#
# No key, no device, no build: a throwaway set in a temp dir whose kernels are the 1BPKCRYP magic
# plus filler bytes, a throwaway key pair in the same temp dir. The four cases #574 names are the
# four the packaging and the stick have to refuse: a set with no manifest, a file the manifest does
# not list, a ciphertext byte flipped after signing, and a manifest signed by another key.
#
# usage: tests/seal_manifest_test.sh
set -uo pipefail

tool=$(cd "$(dirname "$0")/.." && pwd)/tools/seal_manifest.py
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

# a sealed set as scripts/package-linux.sh stages one: layout.json + layer1_*/layer1.elf
setdir=$scratch/qwen3-0.6b-npu
mkdir -p "$setdir/layer1_1" "$setdir/layer1_2"
printf '{"layers": 2, "tiles": "q4"}\n' > "$setdir/layout.json"
{ printf 1BPKCRYP; head -c 4096 /dev/urandom; } > "$setdir/layer1_1/layer1.elf"
{ printf 1BPKCRYP; head -c 4096 /dev/urandom; } > "$setdir/layer1_2/layer1.elf"

key=$scratch/release.key pub=$scratch/release.pub other=$scratch/other.key otherpub=$scratch/other.pub
python3 "$tool" keygen "$key" "$pub" > "$scratch/keygen.log" 2>&1 ||
    { echo "FAIL keygen: $(cat "$scratch/keygen.log")"; exit 1; }
python3 "$tool" keygen "$other" "$otherpub" > /dev/null 2>&1
echo "ok   keygen: a throwaway key pair in $scratch (nothing here ships)"

# ── pass: intact, signed, checked with the public key only ────────────────────────────────────
python3 "$tool" write "$setdir" --key "$key" || fail=1
check "the set carries a manifest and a signature" \
      '[ -f "$setdir/manifest.json" ] && [ -f "$setdir/manifest.sig" ]'
check "the manifest lists every shipped file with the sha256 of its ciphertext" \
      '[ "$(grep -c "\"sha256\": \"$(sha256sum "$setdir/layer1_2/layer1.elf" | cut -c1-64)\"" "$setdir/manifest.json")" = 1 ]'
check "check (no key) passes on the intact set" \
      'python3 "$tool" check "$setdir"'
check "verify with the release public key passes on the intact set" \
      'python3 "$tool" verify "$setdir" --pub "$pub"'

# The loader is C++ and holds a key, not this tool: what it signs has to be a plain Ed25519
# signature, so openssl must verify it (and the public key has to be the same point).
openssl_crosscheck() {
    local d=$scratch/ossl
    mkdir -p "$d"
    # PKCS#8 for Ed25519 is the 16-byte prefix plus the 32-byte seed (RFC 8410)
    { printf '302e020100300506032b657004220420'; tr -d '\n' < "$key"; } | xxd -r -p > "$d/k.der"
    openssl pkey -inform DER -in "$d/k.der" -out "$d/k.pem" 2>/dev/null || return 1
    openssl pkey -in "$d/k.pem" -pubout -out "$d/pub.pem" 2>/dev/null || return 1
    tr -d '\n' < "$setdir/manifest.sig" | xxd -r -p > "$d/sig.bin"
    openssl pkeyutl -verify -rawin -pubin -inkey "$d/pub.pem" -in "$setdir/manifest.json" \
        -sigfile "$d/sig.bin" > "$d/verify.log" 2>&1
}
if command -v openssl >/dev/null 2>&1 && command -v xxd >/dev/null 2>&1; then
    check "openssl verifies the signature over the manifest bytes" 'openssl_crosscheck' && \
        echo "     $(grep -o 'Signature Verified Successfully' "$scratch/ossl/verify.log" || true)"
else
    echo "skip openssl cross-check (no openssl or xxd on this machine)"
fi

# ── the four refusals ────────────────────────────────────────────────────────────────────────
# 1. no manifest: a set that was never signed
cp -a "$setdir" "$scratch/no-manifest"
rm -f "$scratch/no-manifest/manifest.json" "$scratch/no-manifest/manifest.sig"
check "FAIL on a set with no manifest.json" \
      '! python3 "$tool" check "$scratch/no-manifest" > /dev/null'
check "the failure says the set was never signed" \
      'python3 "$tool" check "$scratch/no-manifest" > "$scratch/no-manifest.log" 2>&1; grep "no manifest.json (the set was never signed" "$scratch/no-manifest.log" > /dev/null'
cp -a "$setdir" "$scratch/no-sig"
rm -f "$scratch/no-sig/manifest.sig"
check "FAIL on a manifest.json with no signature (verify)" \
      '! python3 "$tool" verify "$scratch/no-sig" --pub "$pub" > /dev/null'

# 2. an extra file the manifest does not list: something shipped that nothing vouches for
cp -a "$setdir" "$scratch/unlisted"
{ printf 1BPKCRYP; head -c 512 /dev/urandom; } > "$scratch/unlisted/layer1_3.elf"
mkdir -p "$scratch/unlisted/layer1_3"
{ printf 1BPKCRYP; head -c 512 /dev/urandom; } > "$scratch/unlisted/layer1_3/layer1.elf"
check "FAIL on a file the manifest does not list" \
      '! python3 "$tool" check "$scratch/unlisted" > /dev/null'
check "the failure names the unlisted file" \
      'python3 "$tool" check "$scratch/unlisted" > "$scratch/unlisted.log" 2>&1; grep "layer1_3/layer1.elf ships in the set but is not in the manifest" "$scratch/unlisted.log" > /dev/null'

# 3. a ciphertext byte flipped after signing: re-sealed, or corrupted in transit
cp -a "$setdir" "$scratch/tampered"
python3 - "$scratch/tampered/layer1_1/layer1.elf" <<'PY'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[64] ^= 0x01
open(p, "wb").write(bytes(b))
PY
check "FAIL on one flipped ciphertext byte" \
      '! python3 "$tool" check "$scratch/tampered" > /dev/null'
check "the failure names the sha256 mismatch" \
      'python3 "$tool" check "$scratch/tampered" > "$scratch/tampered.log" 2>&1; grep "its bytes are not the ones that were signed" "$scratch/tampered.log" > /dev/null'

# 4. the same set, signed by another key: a set re-sealed/re-signed by someone who is not us
cp -a "$setdir" "$scratch/other-key"
rm -f "$scratch/other-key/manifest.json" "$scratch/other-key/manifest.sig"
python3 "$tool" write "$scratch/other-key" --key "$other" > /dev/null || fail=1
check "a manifest signed by a different key still passes the keyless check" \
      'python3 "$tool" check "$scratch/other-key"'
check "FAIL on a manifest signed by a different key (verify)" \
      '! python3 "$tool" verify "$scratch/other-key" --pub "$pub" > /dev/null'
check "the failure says the signature does not verify" \
      'python3 "$tool" verify "$scratch/other-key" --pub "$pub" > "$scratch/other-key.log" 2>&1; grep "does not verify with the release public key" "$scratch/other-key.log" > /dev/null'

# 5. a set moved onto another set's name: the manifest names the set it was made for
cp -a "$setdir" "$scratch/qwen3-1.7b-npu"
check "FAIL on a set copied onto another set's name" \
      '! python3 "$tool" check "$scratch/qwen3-1.7b-npu" > /dev/null'

# 6. the curve itself, against the RFC 8032 vectors (openssl is not the only judge)
check "the RFC 8032 vectors pass (self-test)" 'python3 "$tool" --self-test'

[ $fail = 0 ] && echo "seal_manifest: PASS" || echo "seal_manifest: FAIL"
exit $fail
