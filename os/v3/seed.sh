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
#   seed.sh render <out-dir> --disk-serial <serial> [options]
#   seed.sh check  <seed-dir>
#   seed.sh selftest
#
# The install seed for 1bit OS v3 (docs/os-v3.md): the two files the stick carries in /1bit/ and the
# installed machine is built from.
#
#   <out-dir>/autoinstall.yaml   subiquity's autoinstall user-data: the target disk (by serial, never by
#                                size), the user, the storage table, the packages, the step hooks
#   <out-dir>/identity           0600, the same facts as key=value for the shell: postinstall and
#                                prepare-stick.sh read it, and postinstall records the ISO's sha256 in
#                                /etc/1bit/release-identity so release-check's identity row answers
#
# render options:
#   --disk-serial <s>      required; the target disk's serial. A stick cannot then wipe another machine,
#                          and a re-run finds the same disk (docs/os-v3.md, "The autoinstall").
#   --password-file <f>    the user's password, one line, read from a file so it never lands in argv or
#                          this box's shell history
#   --mint-password        generate one and print it once (stdout); the hash is all that is written
#   --ssh-pubkey <f>       an authorized key; repeatable
#   --authorized-keys <f>  a file of keys, as /data/ssh/authorized_keys on the stick has them
#   --user <u>             default onebit (a Linux name starts with a letter: adduser's NAME_REGEX)
#   --hostname <h>         default 1bit-os
#   --root-size <nG>       default 64G; floor 24G (the base, /var and a kernel pair), ceil 4096G
#   --timezone <tz>        default America/Halifax (the owner's locale; UTC for a product default)
#   --force                overwrite an existing seed (re-keying a stick is never silent)
#
# `check` validates a rendered seed: the identity's field formats, the mode, every substituted value, and
# the YAML's structure with PyYAML when it is installed (the whole file, not a grep). `selftest` proves
# `check` refuses what it should and accepts what it should, on this machine, with no device - that is the
# test for this script and it runs in a few seconds.
#
# Exit: 0 ok, 1 usage, 2 refused or invalid, 3 a tool is missing.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
prog=seed
die() { printf '%s: REFUSED: %s\n' "$prog" "$*" >&2; exit 2; }
usage() { sed -n '/^#   seed.sh/,/^# Exit:/p' "$0" | sed 's/^# \{0,1\}//'; }

for t in python3 openssl; do
    command -v "$t" >/dev/null 2>&1 || { printf '%s: %s is missing\n' "$prog" "$t" >&2; exit 3; }
done

# ---- the template ---------------------------------------------------------------------------------
# Kept here, not in a second file, so a placeholder and the check that requires it cannot drift apart.
# SEED_TEMPLATE=<file> replaces it - the selftest uses that to prove a broken template is refused.
template() {
    cat <<'TEMPLATE'
#cloud-config
# Written by os/v3/seed.sh. Do not edit on the stick: re-render it (docs/os-v3.md).
autoinstall:
  version: 1
  # Nothing is asked. The two questions that must not be silent - which disk, which user - are answered
  # by the seed's own inputs and shown on the install screen before anything is written (engine#710).
  interactive-sections: []
  refresh-installer: {update: false}          # the medium is pinned; no network fetch of subiquity
  locale: en_US.UTF-8
  keyboard: {layout: us}
  timezone: "@TIMEZONE@"
  identity:
    hostname: "@HOSTNAME@"
    username: "@USER@"
    password: "@PASSWORD_HASH@"               # openssl passwd -6; the plaintext is never written
  ssh:
    install-server: true
    allow-pw: false                           # keys only; the password is for sudo and the console
    authorized-keys: [@SSH_KEYS@]
  storage:
    swap: {size: 0}
    config:
      - {type: disk, id: disk0, ptable: gpt, grub_device: true, preserve: false,
         match: {serial: "@DISK_SERIAL@"}}
      - {type: partition, id: part-esp, device: disk0, number: 1, size: 1G, flag: boot, wipe: superblock}
      - {type: format, id: fmt-esp, volume: part-esp, fstype: fat32, label: 1BIT-OS}
      - {type: mount, id: mnt-esp, device: fmt-esp, path: /boot/efi}
      - {type: partition, id: part-root, device: disk0, number: 2, size: "@ROOT_SIZE@", wipe: superblock}
      - {type: format, id: fmt-root, volume: part-root, fstype: ext4, label: 1BIT-ROOT}
      - {type: mount, id: mnt-root, device: fmt-root, path: /}
      # preserve: true is the v2 -> v3 upgrade path in one key: an existing 1BIT-DATA is adopted with its
      # contents, and on a fresh disk curtin creates it. No separate "upgrade" installer exists.
      - {type: partition, id: part-data, device: disk0, number: 3, size: -1, preserve: true}
      - {type: format, id: fmt-data, volume: part-data, fstype: ext4, label: 1BIT-DATA, preserve: true}
      - {type: mount, id: mnt-data, device: fmt-data, path: /data}
  apt:
    geoip: false
    preserve_sources_list: false
    # The archive, for the stock packages only. Ours come from the medium in postinstall and are *not* a
    # persistent source: a file:///cdrom URI left in the target's sources.list is a dead source the moment
    # the stick is out, so the installed machine's one source is 1bit.gg/apt (1bit-appliance writes it).
    primary:
      - {uri: "http://archive.ubuntu.com/ubuntu", arches: [amd64]}
    fallback: offline-install
  packages: [openssh-server, efibootmgr, mokutil, unattended-upgrades, ca-certificates, curl,
             python3-minimal, rsync, zstd, pciutils, lm-sensors, ethtool]
  early-commands:
    - /cdrom/1bit/steps plan          # the whole step list, before subiquity draws (engine#710)
  late-commands:
    - curtin in-target -- /cdrom/1bit/postinstall   # the steps in docs/os-v3.md
  error-commands:
    - /cdrom/1bit/steps failed        # the failing step stays on screen with its output
  reporting:
    builtin: {type: print}
TEMPLATE
}

# ---- field formats, in one place, used by render, check and the selftest ---------------------------
valid_hostname() { printf '%s' "$1" | grep -qE '^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$'; }
valid_user() { printf '%s' "$1" | grep -qE '^[a-z][a-z0-9_-]{0,31}$'; }
# shellcheck disable=SC2016  # the \$ are the regex's, not a variable: a crypt(3) SHA-512 hash
valid_hash() { printf '%s' "$1" | grep -qE '^\$6\$[A-Za-z0-9./]{1,16}\$[A-Za-z0-9./]{43,}$'; }
valid_serial() { printf '%s' "$1" | grep -qE '^[A-Za-z0-9._-]{2,64}$'; }
    valid_pubkey() {
    printf '%s' "$1" | grep -qE '^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(256|384|521)|sk-ssh-ed25519@openssh.com|sk-ecdsa-sha2-nistp256@openssh.com) AAAA[A-Za-z0-9+/=]+( .{0,200})?$'
}
valid_root_size() {
    local n
    printf '%s' "$1" | grep -qE '^[0-9]{2,4}G$' || return 1
    n=$(printf '%s' "$1" | tr -dc 0-9)
    [ "$n" -ge 24 ] && [ "$n" -le 4096 ]
}
valid_timezone() { printf '%s' "$1" | grep -qE '^[A-Za-z][A-Za-z0-9_+-]*(/[A-Za-z0-9_+-]+)*$'; }

validate_identity_fields() {   # <user> <hostname> <hash> <serial> <root-size> <timezone>
    valid_user "$1" || die "user '$1' is not a user name (^[a-z][a-z0-9_-]{0,31}\$)"
    valid_hostname "$2" || die "hostname '$2' is not an RFC 1123 label (lowercase, no trailing '-')"
    valid_hash "$3" || die "password hash is not a crypt(3) SHA-512 hash (\$6\$salt\$hash)"
    valid_serial "$4" || die "disk serial '$4' is not a serial (letters, digits, . _ -; 2..64)"
    valid_root_size "$5" || die "root size '$5' is not 24G..4096G in whole GiB"
    valid_timezone "$6" || die "timezone '$6' is not a zone name (e.g. UTC, America/Halifax)"
}

read_identity() {   # <seed-dir> -> sets id_* (no eval: key=value, first '=' splits)
    local f=$1
    [ -f "$f" ] || die "no identity at $f"
    id_user=$(sed -n 's/^user=//p' "$f" | head -n 1)
    id_hostname=$(sed -n 's/^hostname=//p' "$f" | head -n 1)
    id_password_hash=$(sed -n 's/^password_hash=//p' "$f" | head -n 1)
    id_disk_serial=$(sed -n 's/^disk_serial=//p' "$f" | head -n 1)
    id_root_size=$(sed -n 's/^root_size=//p' "$f" | head -n 1)
    id_timezone=$(sed -n 's/^timezone=//p' "$f" | head -n 1)
}

# ---- check ----------------------------------------------------------------------------------------
check_seed() {   # <seed-dir>
    local d=$1 yaml mode
    yaml=$d/autoinstall.yaml
    [ -f "$yaml" ] || die "no autoinstall.yaml in $d"
    [ -f "$d/identity" ] || die "no identity in $d"
    mode=$(stat -c %a "$d/identity")
    [ "$mode" = 600 ] || die "identity is mode $mode, wanted 600: it carries the password hash"
    grep -nE '@[A-Z_]{2,}@' "$yaml" >/dev/null 2>&1 &&
        die "unsubstituted placeholder: $(grep -nE '@[A-Z_]{2,}@' "$yaml" | head -n 1)"
    read_identity "$d/identity"
    validate_identity_fields "$id_user" "$id_hostname" "$id_password_hash" "$id_disk_serial" \
        "$id_root_size" "$id_timezone"
    # the YAML, as a document and not as text: PyYAML when this host has it (the build hosts do), and a
    # structural grep when it does not, so a stick built on a bare host is still checked.
    SEED_USER=$id_user SEED_HOST=$id_hostname SEED_HASH=$id_password_hash SEED_SERIAL=$id_disk_serial \
    SEED_ROOT=$id_root_size SEED_TZ=$id_timezone python3 - "$yaml" <<'PY'
import os, re, sys

path = sys.argv[1]
text = open(path).read()
keys = ("user", "host", "hash", "serial", "root", "tz")
want = {k: os.environ["SEED_" + k.upper()] for k in keys}
bad = []

def need(cond, msg):
    if not cond:
        bad.append(msg)

try:
    import yaml
except ImportError:
    print("seed: no PyYAML on this host: checking the document structurally, not by parsing", file=sys.stderr)
    need(re.search(r'^\s*version: 1$', text, re.M), "autoinstall.version is not 1")
    need("interactive-sections: []" in text, "interactive-sections is not []")
    need('username: "%s"' % want["user"] in text, "identity.username is not %s" % want["user"])
    need('hostname: "%s"' % want["host"] in text, "identity.hostname is not %s" % want["host"])
    need('serial: "%s"' % want["serial"] in text, "the disk match serial is not %s" % want["serial"])
    need('size: "%s"' % want["root"] in text, "the root partition size is not %s" % want["root"])
    need("label: 1BIT-ROOT" in text and "path: /" in text, "no 1BIT-ROOT mounted at /")
    need("label: 1BIT-DATA" in text and "path: /data" in text, "no 1BIT-DATA mounted at /data")
    need("preserve: true" in text, "no preserve: true (the /data adoption and the fresh-disk path)")
    need("/cdrom/1bit/postinstall" in text, "late-commands does not run /cdrom/1bit/postinstall")
else:
    doc = yaml.safe_load(text)["autoinstall"]
    need(doc.get("version") == 1, "autoinstall.version is not 1")
    need(doc.get("interactive-sections") == [], "interactive-sections is not []")
    ident = doc.get("identity") or {}
    need(ident.get("username") == want["user"], "identity.username is not %s" % want["user"])
    need(ident.get("hostname") == want["host"], "identity.hostname is not %s" % want["host"])
    need(ident.get("password") == want["hash"], "identity.password is not the identity's hash")
    need(doc.get("timezone") == want["tz"], "timezone is not %s" % want["tz"])
    auth = (doc.get("ssh") or {}).get("authorized-keys") or []
    need(bool(auth), "ssh.authorized-keys is empty")
    for k in auth:
        need(isinstance(k, str) and re.match(r'^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp256|sk-ssh-ed25519@openssh.com) AAAA', k),
             "an authorized key is not an OpenSSH public key: %r" % (k,))
    st = doc.get("storage") or {}
    need((st.get("swap") or {}).get("size") == 0, "swap.size is not 0 (no swap partition, /swap.img)")
    cfg = st.get("config") or []
    disks = [e for e in cfg if e.get("type") == "disk"]
    need(len(disks) == 1, "storage.config has %d disks, wanted exactly 1" % len(disks))
    need(bool(disks) and (disks[0].get("match") or {}).get("serial") == want["serial"],
         "the disk match serial is not %s" % want["serial"])
    labels = {e.get("label"): e for e in cfg if e.get("type") == "format"}
    need("1BIT-ROOT" in labels, "no 1BIT-ROOT format")
    need("1BIT-DATA" in labels, "no 1BIT-DATA format")
    need(labels.get("1BIT-DATA", {}).get("preserve") is True, "1BIT-DATA is not preserve: true")
    mounts = {e.get("path"): e for e in cfg if e.get("type") == "mount"}
    need(set(mounts) == {"/", "/boot/efi", "/data"},
         "mounts are %s, wanted /, /boot/efi, /data" % sorted(mounts))
    roots = [e for e in cfg if e.get("type") == "partition" and e.get("number") == 2]
    need(bool(roots) and roots[0].get("size") == want["root"],
         "the root partition is not %s" % want["root"])
    data = [e for e in cfg if e.get("type") == "partition" and e.get("number") == 3]
    need(bool(data) and data[0].get("size") == -1, "partition 3 does not take the remainder (-1)")
    apt = doc.get("apt") or {}
    prim = apt.get("primary") or []
    need(bool(prim), "apt.primary is empty")
    for p in prim:
        need(not str(p.get("uri", "")).startswith("file://"),
             "apt.primary has a file:// URI (%s): it would outlive the stick" % p.get("uri"))
    need(apt.get("fallback") == "offline-install", "apt.fallback is not offline-install")
    pk = doc.get("packages") or []
    for p in ("openssh-server", "efibootmgr", "mokutil", "unattended-upgrades"):
        need(p in pk, "packages is missing %s" % p)
    need(bool(doc.get("early-commands")), "no early-commands (the step list, #710)")
    need(any("/cdrom/1bit/postinstall" in str(c) for c in doc.get("late-commands") or []),
         "late-commands does not run /cdrom/1bit/postinstall")

if bad:
    for b in bad:
        print("seed: FAIL " + b, file=sys.stderr)
    sys.exit(2)
PY
    printf 'seed: OK   %s (user %s, host %s, disk %s, root %s)\n' \
        "$d" "$id_user" "$id_hostname" "$id_disk_serial" "$id_root_size"
}

# ---- render ---------------------------------------------------------------------------------------
cmd=${1:-}
[ -n "$cmd" ] || { usage >&2; exit 1; }
shift || true

case $cmd in
selftest)
    exec "$here/seed-selftest.sh"
    ;;
template)
    # the template as this script has it; the selftest uses it to build a broken one, and an operator
    # uses it to see what a stick would carry
    template
    exit 0
    ;;
check)
    [ $# -eq 1 ] || { usage >&2; exit 1; }
    check_seed "$1"
    exit 0
    ;;
render) ;;
*)
    usage >&2
    exit 1
    ;;
esac

out=${1:-}
[ -n "$out" ] || { usage >&2; exit 1; }
shift || true

USER_NAME=onebit
HOSTNAME=1bit-os
ROOT_SIZE=64G
TIMEZONE=America/Halifax
SERIAL=""
PASSWORD_FILE=""
MINT=0
FORCE=0
KEYS=()   # file paths, in order
AUTH_KEYS_FILE=""

while [ $# -gt 0 ]; do
    case $1 in
    --disk-serial) SERIAL=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    --password-file) PASSWORD_FILE=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    --mint-password) MINT=1; shift ;;
    --ssh-pubkey) [ -n "${2:-}" ] || { usage >&2; exit 1; }; KEYS+=("$2"); shift 2 ;;
    --authorized-keys) AUTH_KEYS_FILE=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    --user) USER_NAME=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    --hostname) HOSTNAME=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    --root-size) ROOT_SIZE=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    --timezone) TIMEZONE=${2:-}; shift 2 || { usage >&2; exit 1; } ;;
    --force) FORCE=1; shift ;;
    --help | -h) usage; exit 0 ;;
    *) printf '%s: unknown option %s\n' "$prog" "$1" >&2; exit 1 ;;
    esac
done

[ -n "$SERIAL" ] || die "no --disk-serial: the seed names its target disk, or a stick could wipe another machine"
valid_serial "$SERIAL" || die "disk serial '$SERIAL' is not a serial (letters, digits, . _ -; 2..64)"

# the password: a file, or minted and printed exactly once. Never argv (a password in argv is in ps, the
# shell's history and this box's audit log).
if [ "$MINT" = 1 ] && [ -n "$PASSWORD_FILE" ]; then
    die "--mint-password and --password-file are two ways to the same value; pick one"
fi
plain=""
if [ "$MINT" = 1 ]; then
    plain=$(openssl rand -base64 16 | tr -dc 'A-Za-z0-9' | cut -c1-16)
    [ ${#plain} -eq 16 ] || { printf '%s: could not mint a password\n' "$prog" >&2; exit 3; }
elif [ -n "$PASSWORD_FILE" ]; then
    [ -f "$PASSWORD_FILE" ] || die "no password file at $PASSWORD_FILE"
    plain=$(head -n 1 "$PASSWORD_FILE")
    [ -n "$plain" ] || die "$PASSWORD_FILE's first line is empty"
else
    die "no password: --password-file <f> or --mint-password (the user needs sudo and the console)"
fi
# A salt derived from the stick's own identity, not a random one: two renders of the same inputs then
# produce the same bytes, so a seed can be reviewed and diffed. The salt's job (unique per stick) is kept
# by the serial in its input, and a minted 16-character password does not lean on the salt at all.
salt=$(printf '%s:%s:%s' "$SERIAL" "$USER_NAME" "$HOSTNAME" | sha256sum | cut -c1-16)
hash=$(printf '%s' "$plain" | openssl passwd -6 -salt "$salt" -stdin)

# the keys: --ssh-pubkey files, and/or a file of lines (the stick's /data/ssh/authorized_keys)
key_list=()
for f in "${KEYS[@]}"; do
    [ -f "$f" ] || die "no public key at $f"
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        key_list+=("$line")
    done <"$f"
done
if [ -n "$AUTH_KEYS_FILE" ]; then
    [ -f "$AUTH_KEYS_FILE" ] || die "no authorized-keys file at $AUTH_KEYS_FILE"
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        key_list+=("$line")
    done <"$AUTH_KEYS_FILE"
fi
[ ${#key_list[@]} -gt 0 ] || die "no SSH key: ssh is keys-only (allow-pw: false), so the machine would have no way in"
for k in "${key_list[@]}"; do
    valid_pubkey "$k" || die "not an OpenSSH public key: $(printf '%s' "$k" | cut -c1-40)..."
done

validate_identity_fields "$USER_NAME" "$HOSTNAME" "$hash" "$SERIAL" "$ROOT_SIZE" "$TIMEZONE"

if [ -e "$out/autoinstall.yaml" ] && [ "$FORCE" != 1 ]; then
    die "$out/autoinstall.yaml exists; --force to replace it (and $out/identity with it)"
fi
mkdir -p "$out"
out=$(cd "$out" && pwd)

# the medium this seed installs from, from the pin beside this script: postinstall records it, and
# release-check's identity row reads what postinstall wrote
iso_sha=$(sed -n 's/^server_iso_sha256=//p' "$here/pin.txt" 2>/dev/null | head -n 1)
iso_label=$(sed -n 's/^server_iso_label=//p' "$here/pin.txt" 2>/dev/null | head -n 1)
created=$(date -u +%Y-%m-%dT%H:%M:%SZ)

tpl=$(mktemp)
keysfile=$(mktemp)
trap 'rm -f "$tpl" "$keysfile"' EXIT
printf '%s\n' "${key_list[@]}" >"$keysfile"
if [ -n "${SEED_TEMPLATE:-}" ]; then
    [ -f "$SEED_TEMPLATE" ] || die "SEED_TEMPLATE=$SEED_TEMPLATE is not a file"
    cp "$SEED_TEMPLATE" "$tpl"
else
    template >"$tpl"
fi

SEED_USER=$USER_NAME SEED_HOST=$HOSTNAME SEED_HASH=$hash SEED_SERIAL=$SERIAL SEED_ROOT=$ROOT_SIZE \
SEED_TZ=$TIMEZONE python3 - "$tpl" "$out/autoinstall.yaml" "$keysfile" <<'PY'
import os, sys

tpl, dest, keysfile = sys.argv[1], sys.argv[2], sys.argv[3]
keys = [k for k in open(keysfile).read().split("\n") if k]
vals = {
    "@USER@": os.environ["SEED_USER"],
    "@HOSTNAME@": os.environ["SEED_HOST"],
    "@PASSWORD_HASH@": os.environ["SEED_HASH"],
    "@DISK_SERIAL@": os.environ["SEED_SERIAL"],
    "@ROOT_SIZE@": os.environ["SEED_ROOT"],
    "@TIMEZONE@": os.environ["SEED_TZ"],
    # a YAML flow list of quoted keys; a key has no '"', so quoting is exact
    "@SSH_KEYS@": ", ".join('"%s"' % k for k in keys),
}
text = open(tpl).read()
for k, v in vals.items():
    text = text.replace(k, v)
open(dest, "w").write(text)
PY
chmod 644 "$out/autoinstall.yaml"

umask 077
{
    echo "# Written by os/v3/seed.sh $created. Key=value, read by postinstall and prepare-stick.sh."
    echo "user=$USER_NAME"
    echo "hostname=$HOSTNAME"
    echo "password_hash=$hash"
    echo "disk_serial=$SERIAL"
    echo "root_size=$ROOT_SIZE"
    echo "timezone=$TIMEZONE"
    echo "arch=amd64"
    echo "iso_sha256=$iso_sha"
    echo "iso_label=$iso_label"
    echo "created=$created"
} >"$out/identity"
chmod 600 "$out/identity"

check_seed "$out" >/dev/null
if [ "$MINT" = 1 ]; then
    printf 'seed: password (printed once, write it down): %s\n' "$plain"
fi
printf 'seed: rendered %s/{autoinstall.yaml,identity} and checked them\n' "$out"
exit 0
