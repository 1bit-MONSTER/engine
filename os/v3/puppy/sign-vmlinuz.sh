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
# sign-vmlinuz.sh <vmlinuz> [--out <file>] [--key <priv>] [--cert <pem>]
#
# Sign the appliance's kernel image with the machine's enrolled MOK, so the shim -> GRUB chain will load it
# with Secure Boot ON. This is the defect the 2 Nov board boot found and no VM row could: the Puppy ISO's
# `vmlinuz` came straight out of `make bindeb-pkg` and carries no signature at all (`sbverify --list`:
# "No signature table present"), so GRUB on the Max+ 395 answered
#   error: bad shim lock signature / you need to load the kernel first
# and the stick stopped there. The out-of-tree modules were MOK-signed the whole time - G3 `--require-mok`
# checked exactly those - which is why the gap survived every check this lane had.
#
# <vmlinuz>  the kernel image, as packaged (boot/vmlinuz-<kver> in the kernel stage)
# --out      where the signed image goes; default: in place, atomically. Re-running on an image that already
#            verifies against this certificate is a no-op: sbsign would happily stack a second signature, and
#            a kernel that grows every time a script runs is a bug that hides in a size difference.
# --key      the private key; default: /var/lib/shim-signed/mok/MOK-Kernel.priv
# --cert     the certificate; default: /var/lib/shim-signed/mok/MOK-Kernel.pem
#
# The key never leaves this machine and is never printed: it is copied into a root-owned 0700 directory,
# converted, used and wiped. Only the certificate (public - it is enrolled in the firmware) is ever read by
# a probe. Two measured traps are handled here rather than by the caller:
#   - `MOK-Kernel.priv` is an OpenSSH private key, which openssl/sbsign refuse ("wrong tag"); a copy is
#     converted with `ssh-keygen -p -m PEM` and the conversion is proved against the certificate before
#     anything is signed.
#   - `sbsign` needs the PEM spelling of a key that `openssl pkey` can read; the match check is what makes a
#     wrong key a refusal instead of a silently different signature.
#
# Exit: 0 signed and verified, 1 usage, 2 refused (not a kernel image, key does not match the certificate,
# the signature does not verify afterwards, or no MOK on this machine), 3 a tool is missing.
set -euo pipefail
prog=sign-vmlinuz
die() {
    printf '%s: REFUSED: %s\n' "$prog" "$*" >&2
    exit 2
}
usage() { sed -n '/^# sign-vmlinuz.sh/,/^# Exit:/p' "$0" | sed 's/^# \{0,1\}//'; }

[ $# -ge 1 ] || { usage >&2; exit 1; }
img=$1
shift
out=""
key=/var/lib/shim-signed/mok/MOK-Kernel.priv
cert=/var/lib/shim-signed/mok/MOK-Kernel.pem
while [ $# -gt 0 ]; do
    case $1 in
    --out) out=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    --key) key=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    --cert) cert=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    -h | --help) usage; exit 0 ;;
    *) printf '%s: unknown option %s\n' "$prog" "$1" >&2; exit 1 ;;
    esac
done
[ -f "$img" ] || die "no such kernel image: $img"
# the key is root-only on every box this runs on; signing IS the privileged step, so escalate here once
# instead of asking each caller to remember
if [ "$(id -u)" -ne 0 ] && { [ ! -r "$key" ] || [ ! -r "$cert" ]; }; then
    exec sudo -n -- "$0" "$img" --out "$out" --key "$key" --cert "$cert"
fi
[ -r "$key" ] || die "cannot read $key (root-owned; run this through sudo)"
[ -r "$cert" ] || die "no MOK certificate at $cert - there is nothing to sign against"
for t in sbsign sbverify ssh-keygen openssl file; do
    command -v "$t" > /dev/null 2>&1 || { printf '%s: %s is missing\n' "$prog" "$t" >&2; exit 3; }
done
# a bootloader signature on something that is not a kernel is 50 KB of pointless signature, and would
# hide a wrong argument; the release string sits in the setup header and `file` reads it
file -b "$img" | grep -q 'Linux kernel x86 boot executable' ||
    die "$img is not an x86 kernel image ($(file -b "$img"))"

out=${out:-$img}
tmp=$(mktemp -d /var/tmp/1bit-sbsign.XXXXXX)
trap 'rm -rf "$tmp"' EXIT
chmod 0700 "$tmp"
cp "$key" "$tmp/k"
cp "$cert" "$tmp/c.pem"
chmod 0600 "$tmp/k"
ssh-keygen -p -m PEM -N "" -f "$tmp/k" > /dev/null 2>&1 ||
    die "cannot convert $key to PEM (an OpenSSH key needs ssh-keygen -p -m PEM)"
k=$(openssl pkey -in "$tmp/k" -pubout 2> /dev/null | sha256sum | cut -d' ' -f1)
c=$(openssl x509 -in "$tmp/c.pem" -pubkey -noout 2> /dev/null | sha256sum | cut -d' ' -f1)
[ -n "$k" ] && [ "$k" = "$c" ] || die "$key is not the private key of $cert - refusing to sign with it"
subject=$(openssl x509 -in "$tmp/c.pem" -noout -subject | sed 's/^subject=//')

# already signed by this certificate: nothing to do (see --out above)
if sbverify --cert "$tmp/c.pem" "$img" > /dev/null 2>&1; then
    if [ "$out" != "$img" ]; then
        cp -f "$img" "$out"
    fi
    printf '%s: %s is already signed by %s - nothing to do\n' "$prog" "$out" "$subject"
    exit 0
fi

# sign beside the target and move it into place, so a failure never leaves a half-written kernel behind
stage=$(mktemp "$out.sbsign.XXXXXX") || die "cannot write beside $out"
rm -f "$stage"
sbsign --key "$tmp/k" --cert "$tmp/c.pem" --output "$stage" "$img" || {
    rm -f "$stage"
    die "sbsign failed"
}
sbverify --cert "$tmp/c.pem" "$stage" > /dev/null 2>&1 || {
    rm -f "$stage"
    die "the signature does not verify against $cert"
}
chmod 0644 "$stage"
# the caller's uid, not root's: the stage belongs to the builder and mk-huge-kernel.sh reads it as them
[ -z "${SUDO_UID:-}" ] || chown "${SUDO_UID}:${SUDO_GID:-$SUDO_UID}" "$stage"
mv -f "$stage" "$out"

printf '%s: signed %s with %s\n' "$prog" "$out" "$subject"
printf '   kernel   %s (%s B)\n' "$(file -b "$out" | sed -n 's/.*version \([^ ]*\) .*/\1/p')" "$(stat -c %s "$out")"
printf '   sha256   %s\n' "$(sha256sum "$out" | cut -d' ' -f1)"
printf '   signer   %s\n' "$(sbverify --list "$out" | sed -n 's/^ - \(.*\)$/\1/p' | head -n 1)"
