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
# sb-vars.sh <out.fd> [--cert <pem>]
#
# OVMF's Secure Boot variable store with THIS machine's MOK certificate enrolled in `db`, so a QEMU boot
# verifies the appliance the way the board does. Why it exists: every VM row this lane ran booted the
# appliance with Secure Boot off, and the 2 Nov board boot then failed at
#   error: bad shim lock signature / you need to load the kernel first
# on a kernel image that had never been signed. A VM that cannot run that check cannot catch that class, so
# the row starts from the firmware the board has instead of from a permissive one.
#
# <out.fd>  the variable store to write (a copy of OVMF_VARS_4M.ms.fd: PK, KEK and Microsoft's db are what
#           let Canonical's Microsoft-signed shim run at all; ours is added beside them, nothing is removed)
# --cert    the certificate to enrol; default: /var/lib/shim-signed/mok/MOK-Kernel.pem. The certificate is
#           public (it is enrolled in this box's firmware); the key is never read here.
#
# With this store, `qemu -machine q35,smm=on -global driver=cfi.pflash01,property=secure,value=on` plus
# `OVMF_CODE_4M.secboot.fd` boots with SetupMode off and our signature required - an unsigned kernel stops
# at GRUB with exactly the board's message, which is what makes the row evidence.
#
# Exit: 0 written and verified, 1 usage, 2 refused, 3 a tool is missing.
set -euo pipefail
prog=sb-vars
die() {
    printf '%s: REFUSED: %s\n' "$prog" "$*" >&2
    exit 2
}
usage() { sed -n '/^# sb-vars.sh/,/^# Exit:/p' "$0" | sed 's/^# \{0,1\}//'; }

[ $# -ge 1 ] || { usage >&2; exit 1; }
out=$1
shift
cert=/var/lib/shim-signed/mok/MOK-Kernel.pem
while [ $# -gt 0 ]; do
    case $1 in
    --cert) cert=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    -h | --help) usage; exit 0 ;;
    *) printf '%s: unknown option %s\n' "$prog" "$1" >&2; exit 1 ;;
    esac
done
for t in virt-fw-vars openssl cp; do
    command -v "$t" > /dev/null 2>&1 || {
        printf '%s: %s is missing (apt install python3-virt-firmware)\n' "$prog" "$t" >&2
        exit 3
    }
done
[ -r /usr/share/OVMF/OVMF_VARS_4M.ms.fd ] || die "no /usr/share/OVMF/OVMF_VARS_4M.ms.fd (apt install ovmf)"
[ -r /usr/share/OVMF/OVMF_CODE_4M.secboot.fd ] || die "no /usr/share/OVMF/OVMF_CODE_4M.secboot.fd (apt install ovmf)"
# the certificate is root-owned 0600 beside the key; the public half is what a probe needs
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if [ -r "$cert" ]; then
    cp "$cert" "$tmp/c.pem" && chmod 0644 "$tmp/c.pem"
else
    sudo -n install -m 0644 "$cert" "$tmp/c.pem" || die "cannot read $cert"
fi
want=$(openssl x509 -in "$tmp/c.pem" -noout -fingerprint -sha256 | sed 's/^.*=//')
[ -n "$want" ] || die "$cert is not an x509 certificate"
subject=$(openssl x509 -in "$tmp/c.pem" -noout -subject | sed 's/^subject=//')

mkdir -p "$(dirname "$out")"
cp /usr/share/OVMF/OVMF_VARS_4M.ms.fd "$out"
virt-fw-vars --inplace "$out" --add-db d719b2cb-3d3a-4596-a3bc-dad00e67656f "$tmp/c.pem" > /dev/null 2>&1 ||
    die "virt-fw-vars could not add the certificate to db"

# verify by reading the store back: the enrolled certificate's fingerprint has to be in its db, or the file
# is just a copy of the template and every VM boot with it proves nothing
mkdir -p "$tmp/x"
(cd "$tmp/x" && virt-fw-vars -i "$out" --extract-certs > /dev/null 2>&1) || die "cannot read $out back"
found=""
for p in "$tmp/x"/db-*.pem; do
    [ -e "$p" ] || continue
    got=$(openssl x509 -in "$p" -noout -fingerprint -sha256 2> /dev/null | sed 's/^.*=//')
    [ "$got" = "$want" ] && found=$p
done
[ -n "$found" ] || die "$subject is not in the db of $out"
ntot=$(ls "$tmp/x"/db-*.pem 2> /dev/null | wc -l)
printf '%s: %s\n' "$prog" "$out"
printf '   enrolled %s\n' "$subject"
printf '   db now holds %s certificate(s), ours among them (%s)\n' "$ntot" "${want:0:16}…"
printf '   boot with OVMF_CODE_4M.secboot.fd, -machine q35,smm=on,\n'
printf '             -global driver=cfi.pflash01,property=secure,value=on\n'
