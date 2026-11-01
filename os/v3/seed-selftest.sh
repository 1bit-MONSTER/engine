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
# seed-selftest.sh
#
# The test for os/v3/seed.sh: no device, no stick, no QEMU, a few seconds. It proves both directions -
# a good seed is rendered and accepted, and every bad one is refused with a message that says why - and
# it proves the four structure regressions this file exists for by breaking the template on purpose
# (no `preserve: true`, a `file://` apt source that would outlive the stick, an unsubstituted
# placeholder, and a password hash that is not SHA-512).
#
# Exit: 0 all checks pass, 1 a check failed.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
seed=$here/seed.sh
w=$(mktemp -d)
trap 'rm -rf "$w"' EXIT
pass=0
fail=0

ok() { pass=$((pass + 1)); printf '  PASS  %s\n' "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL  %s\n' "$*"; }
contains() {   # <text> <label>, in the seed's autoinstall.yaml
    if grep -qF -- "$1" "$2"; then ok "$3"; else bad "$3: no '$1'"; fi
}
same() { if cmp -s "$1" "$2"; then ok "$3"; else bad "$3"; fi; }

# want <expected-rc> <want-text-in-output|-> <label> <command...>
want() {
    local rc=$1 text=$2 label=$3 out
    shift 3
    out=$("$@" 2>&1)
    local got=$?
    if [ "$got" != "$rc" ]; then
        bad "$label: exit $got, wanted $rc :: $(printf '%s' "$out" | head -n 2 | tr '\n' ' ')"
        return 1
    fi
    if [ "$text" != "-" ] && ! printf '%s' "$out" | grep -qF -- "$text"; then
        bad "$label: no '$text' in: $(printf '%s' "$out" | head -n 2 | tr '\n' ' ')"
        return 1
    fi
    ok "$label"
    return 0
}

command -v ssh-keygen >/dev/null 2>&1 || { echo "seed-selftest: no ssh-keygen" >&2; exit 1; }
ssh-keygen -q -t ed25519 -N '' -C 'seed-selftest' -f "$w/testkey" >/dev/null 2>&1 || {
    echo "seed-selftest: ssh-keygen failed" >&2; exit 1; }
printf 'open-sesame\n' >"$w/pw"
printf 'notakey AAAAwhatever\n' >"$w/badkey"

echo "== render a good seed, and what it must contain"
want 0 "rendered" "render: a seeded stick" \
    "$seed" render "$w/good" --disk-serial YMA62T0KA2417407NP --user onebit --hostname 1bit-strixhalo \
    --password-file "$w/pw" --ssh-pubkey "$w/testkey.pub" --timezone UTC
want 0 "OK" "check: the seed it just rendered" "$seed" check "$w/good"
contains 'serial: "YMA62T0KA2417407NP"' "$w/good/autoinstall.yaml" "the YAML matches the disk by serial"
contains 'hostname: "1bit-strixhalo"' "$w/good/autoinstall.yaml" "the YAML carries the hostname"
if [ "$(stat -c %a "$w/good/identity")" = 600 ]; then ok "identity is 0600"
else bad "identity is $(stat -c %a "$w/good/identity"), wanted 0600"; fi
if grep -qx "iso_sha256=$(sed -n 's/^server_iso_sha256=//p' "$here/pin.txt")" "$w/good/identity"; then
    ok "identity records the pinned ISO's sha256"
else bad "identity does not record the ISO's sha256"; fi

echo "== two renders of the same inputs are the same bytes (a stick is reviewable)"
mkdir -p "$w/again"
want 0 - "render: the same inputs into another directory" \
    "$seed" render "$w/again/seed" --disk-serial YMA62T0KA2417407NP --user onebit --hostname 1bit-strixhalo \
    --password-file "$w/pw" --ssh-pubkey "$w/testkey.pub" --timezone UTC
same "$w/good/autoinstall.yaml" "$w/again/seed/autoinstall.yaml" "autoinstall.yaml is byte-identical"

echo "== the refusals, each with the reason it refuses"
want 2 "no --disk-serial" "no disk serial" \
    "$seed" render "$w/r1" --mint-password --ssh-pubkey "$w/testkey.pub"
want 2 "not a serial" "a serial with a space in it" \
    "$seed" render "$w/r2" --disk-serial "not a serial" --mint-password --ssh-pubkey "$w/testkey.pub"
want 2 "not an RFC 1123 label" "an uppercase hostname" \
    "$seed" render "$w/r3" --disk-serial S1 --hostname 1Bit_Host --mint-password --ssh-pubkey "$w/testkey.pub"
want 2 "not a user name" "a user name starting with a digit (1bit: adduser's NAME_REGEX)" \
    "$seed" render "$w/r4a" --disk-serial S1 --user 1bit --mint-password --ssh-pubkey "$w/testkey.pub"
want 2 "not a user name" "a user name that is not one" \
    "$seed" render "$w/r4" --disk-serial S1 --user 'Root!' --mint-password --ssh-pubkey "$w/testkey.pub"
want 2 "not 24G..4096G" "a 16G root (the floor is 24G)" \
    "$seed" render "$w/r5" --disk-serial S1 --root-size 16G --mint-password --ssh-pubkey "$w/testkey.pub"
want 2 "not 24G..4096G" "a 8192G root (the ceiling is 4096G)" \
    "$seed" render "$w/r6" --disk-serial S1 --root-size 8192G --mint-password --ssh-pubkey "$w/testkey.pub"
want 2 "not 24G..4096G" "a root size without G" \
    "$seed" render "$w/r7" --disk-serial S1 --root-size 64 --mint-password --ssh-pubkey "$w/testkey.pub"
want 2 "not a zone name" "a timezone that is a path" \
    "$seed" render "$w/r8" --disk-serial S1 --timezone '../etc' --mint-password --ssh-pubkey "$w/testkey.pub"
want 2 "no password" "neither a password file nor --mint-password" \
    "$seed" render "$w/r9" --disk-serial S1 --ssh-pubkey "$w/testkey.pub"
want 2 "pick one" "--mint-password and --password-file together" \
    "$seed" render "$w/r10" --disk-serial S1 --mint-password --password-file "$w/pw" --ssh-pubkey "$w/testkey.pub"
want 2 "is empty" "an empty password file" \
    bash -c ": >'$w/empty'; exec '$seed' render '$w/r11' --disk-serial S1 --password-file '$w/empty' --ssh-pubkey '$w/testkey.pub'"
want 2 "no SSH key" "no key at all (ssh is keys-only)" \
    "$seed" render "$w/r12" --disk-serial S1 --password-file "$w/pw"
want 2 "not an OpenSSH public key" "a line that is not a key" \
    "$seed" render "$w/r13" --disk-serial S1 --password-file "$w/pw" --ssh-pubkey "$w/testkey.pub" \
    --authorized-keys "$w/badkey"
want 2 "--force to replace" "a second render over an existing seed" \
    "$seed" render "$w/good" --disk-serial YMA62T0KA2417407NP --password-file "$w/pw" --ssh-pubkey "$w/testkey.pub"

echo "== check refuses a seed whose structure regressed (the four this file exists for)"
broken() {   # <dest> <sed-expression> -> a copy of the good seed with one line changed
    rm -rf "$1"; cp -a "$w/good" "$1"
    sed -i "$2" "$1/autoinstall.yaml"
    chmod 600 "$1/identity"
}
broken "$w/bad1" 's/preserve: true/preserve: false/'
want 2 "1BIT-DATA is not preserve: true" "check: no preserve: true" "$seed" check "$w/bad1"
broken "$w/bad2" 's#http://archive.ubuntu.com/ubuntu#file:///cdrom/1bit/repo#'
want 2 "would outlive the stick" "check: a file:// apt source" "$seed" check "$w/bad2"
broken "$w/bad3" 's#timezone: "UTC"#timezone: "@NOPE@"#'
want 2 "unsubstituted placeholder" "check: a placeholder nobody substituted" "$seed" check "$w/bad3"
broken "$w/bad4" 's#size: "64G"#size: "8G"#'
want 2 "root partition is not 64G" "check: a root partition size the identity does not have" "$seed" check "$w/bad4"
broken "$w/bad5" 's#label: 1BIT-DATA#label: 1BIT-DAT#; s#path: /data#path: /date#'
want 2 "no 1BIT-DATA format" "check: a /data mount that is not 1BIT-DATA" "$seed" check "$w/bad5"
rm -rf "$w/bad6"; cp -a "$w/good" "$w/bad6"
# shellcheck disable=SC2016  # $1$ is an md5 hash, not a variable
sed -i 's/^password_hash=.*/password_hash=$1$abcd$efgh/' "$w/bad6/identity"
chmod 600 "$w/bad6/identity"
want 2 "not a crypt(3) SHA-512 hash" "check: an md5 password hash" "$seed" check "$w/bad6"

echo "== check refuses a world-readable identity (it carries the hash)"
chmod 644 "$w/good/identity"
want 2 "wanted 600" "check: identity mode 644" "$seed" check "$w/good"
chmod 600 "$w/good/identity"
want 0 "OK" "check: the seed is accepted again" "$seed" check "$w/good"

echo
printf 'seed selftest: %d pass, %d fail\n' "$pass" "$fail"
[ "$fail" = 0 ] || exit 1
echo "seed selftest PASS $pass/$pass"
