#!/usr/bin/env python3
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
"""merge_accounts.py <image etc dir> <live etc dir> [--apply --backup DIR] [--known-default-sha256 HEX]...

On 1bit OS the save layer holds its own copy of /etc/passwd, group, shadow and gshadow as soon as an account is added
(07onboard-user's useradd, a package's useradd, a person's): an overlay's upper copy wins over the image, so
  (a) the image's account changes - the locked root password above all - never reach a box that has a saved copy,
  (b) a new system account the image adds stays invisible behind the saved files,
  (c) and the files cannot simply be moved out of the save layer either: that would delete every local user.
This merges instead, one file at a time, into what the box should see:
  * a SYSTEM account or group is one the image's file names: its line comes from the image (new ones appear);
  * a LOCAL one (only in the saved file) is kept, after the image's, in the saved order;
  * in shadow, a password a person set on a system account is kept - but a saved field that IS a base image's known
    default (matched by the sha256 of the field, so the default itself is in no file of this repository), or that is
    empty, or that equals the image's own, takes the image's field (root: '*', no usable password);
  * in an image group (group and gshadow), members that are image accounts follow the image, and local users keep
    the memberships they had (a person's admin user stays in wheel; the image's spot leaves disk, as #3452 meant).
Password fields are never printed, only the names they belong to. Without --apply it only reports what it would
change (exit 0). --apply writes each changed file atomically (same directory, the live file's mode and owner) after
copying the saved originals to --backup (mode 0700). --known-default-sha256 adds a fingerprint (the tests' own). Exit 1, with nothing written, when a file does not parse or the
result would lose an account or root. Run by 1bit-update after an install (against the new image's files, on the save
layer's upper/etc) and at every boot by 06onboard-accounts (against the running image), so a box updated by an
older 1bit-update is merged on its first boot into a release that carries this.
"""
import hashlib
import os
import re
import sys

# sha256 of root's password field in the base image v2026.47 to v2026.49-rc1 were cut from (the base distribution's
# public default, MD5-crypt). A fingerprint of the HASH STRING: it does not reveal the hash, so it cannot be cracked.
KNOWN_DEFAULT_SHA256 = {
    "c3d7234aee49a2723b63aed242c59fb180d78849cecee971a620fb321c80287f",
}
FILES = (("passwd", 7), ("group", 4), ("shadow", 9), ("gshadow", 4))


def is_known_default(field):
    return hashlib.sha256(field.encode()).hexdigest() in KNOWN_DEFAULT_SHA256


class Refused(Exception):
    pass


# Output is built ONLY from account names (through safe_name) and uids (through read_ids, which reads passwd alone):
# no value that came out of a shadow or gshadow line other than its name column can reach a print.
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,31}\$?")


def safe_name(name):
    """an account name fit to print, else a placeholder (never the raw text)"""
    return name if _NAME.fullmatch(name) else "<an account whose name is not printable>"


def read_ids(path):
    """[(name, uid)] of a passwd file, its own reader: nothing but those two columns is ever read into it"""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            cols = line.rstrip("\n").split(":")
            if len(cols) >= 3 and cols[2].isdigit():
                out.append((safe_name(cols[0]), cols[2]))
    return out


def parse(path, nfields):
    """[(name, fields)] in file order; None when the file does not exist."""
    if not os.path.exists(path):
        return None
    out, seen = [], set()
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split(":")
            # shadow lines may stop early (the base has 8-field lines: no trailing reserved field); kept as they are
            lo = 2 if nfields == 9 else nfields
            if not lo <= len(fields) <= nfields or not fields[0]:
                raise Refused("%s line %d: not %d colon-separated fields" % (path, n, nfields))
            if fields[0] in seen:
                raise Refused("%s line %d: %s appears twice" % (path, n, safe_name(fields[0])))
            seen.add(fields[0])
            out.append((fields[0], fields))
    return out


def members(s):
    return [m for m in s.split(",") if m]


def merge_members(img, saved, image_users):
    """image members, then the saved members that are local users (not image accounts), without repeats"""
    out = list(img)
    for m in saved:
        if m not in image_users and m not in out:
            out.append(m)
    return ",".join(out)


def merge(kind, image, saved, image_users, notes):
    """the merged list of (name, fields) for one file"""
    sv = dict(saved)
    out = []
    for name, f in image:
        s = sv.get(name)
        if s is None or s == f:
            if s is None:
                notes.append("%s: %s added from the image" % (kind, safe_name(name)))
            out.append((name, f))
            continue
        if kind == "passwd":
            notes.append("passwd: %s follows the image" % safe_name(name))
            out.append((name, f))
        elif kind == "shadow":
            pw = s[1]
            if is_known_default(pw):
                notes.append("shadow: %s held the base image's DEFAULT password: it takes the image's field" % safe_name(name))
                out.append((name, f))
            elif pw == "" and f[1] != "":
                notes.append("shadow: %s had NO password: it takes the image's field" % safe_name(name))
                out.append((name, f))
            elif pw == f[1]:
                out.append((name, f))
            else:
                out.append((name, s))   # a password a person set, with its own dates
        else:   # group: name:pw:gid:members   gshadow: name:pw:admins:members
            g = list(f)
            if kind == "group":
                g[3] = merge_members(members(f[3]), members(s[3]), image_users)
            else:
                g[2] = merge_members(members(f[2]), members(s[2]), image_users)
                g[3] = merge_members(members(f[3]), members(s[3]), image_users)
            if g != s:
                notes.append("%s: %s takes the image's line, keeping its local members" % (kind, safe_name(name)))
            out.append((name, g))
    names = {n for n, _ in image}
    for name, s in saved:
        if name not in names:
            out.append((name, s))
    return out


def plan(image_dir, live_dir):
    """{file: (merged text, live text, notes)} for the files that exist in both and would change"""
    img = {k: parse(os.path.join(image_dir, k), n) for k, n in FILES}
    live = {k: parse(os.path.join(live_dir, k), n) for k, n in FILES}
    if img["passwd"] is None:
        raise Refused("%s has no passwd: not an image's etc" % image_dir)
    image_users = {n for n, _ in img["passwd"]}
    result = {}
    for kind, _ in FILES:
        if img[kind] is None or live[kind] is None:
            continue
        notes = []
        merged = merge(kind, img[kind], live[kind], image_users, notes)
        got = {n for n, _ in merged}
        lost = ({n for n, _ in live[kind]} | {n for n, _ in img[kind]}) - got
        if lost:
            raise Refused("%s: the merge would lose %s" % (kind, ", ".join(safe_name(n) for n in sorted(lost))))
        if kind in ("passwd", "shadow") and "root" not in got:
            raise Refused("%s: no root after the merge" % kind)
        text = "".join(":".join(f) + "\n" for _, f in merged)
        with open(os.path.join(live_dir, kind), encoding="utf-8", errors="surrogateescape") as fh:
            old = fh.read()
        if text != old:
            result[kind] = (text, old, notes)
    # uid collisions between a local user and an image account are reported, never renumbered; printed from read_ids
    # (passwd's names and uids only), not from the parsed records the shadow merge uses
    img_ids = read_ids(os.path.join(image_dir, "passwd"))
    img_names = {n for n, _ in img_ids}
    by_uid = {u: n for n, u in img_ids}
    for n, u in read_ids(os.path.join(live_dir, "passwd")):
        if n not in img_names and u in by_uid:
            print("merge_accounts: WARNING: local user %s has uid %s, as the image's %s" % (n, u, by_uid[u]))
    return result


def write_atomic(path, text):
    st = os.stat(path)
    tmp = "%s.merge.%d" % (path, os.getpid())
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chown(tmp, st.st_uid, st.st_gid)
        os.chmod(tmp, st.st_mode & 0o7777)
        os.rename(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv):
    args, apply, backup, i = [], False, None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--apply":
            apply = True
        elif a in ("--backup", "--known-default-sha256") and i + 1 < len(argv):
            i += 1
            if a == "--backup":
                backup = argv[i]
            else:
                KNOWN_DEFAULT_SHA256.add(argv[i].lower())
        elif a.startswith("--"):
            print("merge_accounts: unknown option %s" % a, file=sys.stderr)
            return 2
        else:
            args.append(a)
        i += 1
    if len(args) != 2 or (apply and not backup):
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    image_dir, live_dir = args
    try:
        changes = plan(image_dir, live_dir)
    except (Refused, OSError) as e:
        print("merge_accounts: REFUSED, nothing written: %s" % e)
        return 1
    if not changes:
        print("merge_accounts: the account files already agree with the image (local accounts kept)")
        return 0
    for kind in changes:
        for n in changes[kind][2]:
            print("   " + n)
    if not apply:
        print("merge_accounts: would rewrite %s (dry run)" % ", ".join(changes))
        return 0
    os.makedirs(backup, mode=0o700, exist_ok=True)
    os.chmod(backup, 0o700)
    for kind, (_, old, _) in changes.items():
        fd = os.open(os.path.join(backup, kind), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape") as f:
            f.write(old)
    for kind, (text, _, _) in changes.items():
        write_atomic(os.path.join(live_dir, kind), text)
    print("merge_accounts: rewrote %s; the previous copies are in %s" % (", ".join(changes), backup))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
